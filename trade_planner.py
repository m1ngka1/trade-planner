"""
Pluggable daily basket execution planner.

The optimizer builds a daily schedule for signed parent orders. It is designed
so that participation caps, residual-risk adjustments, and transaction-cost
models can be swapped independently.

Required runtime packages:
    pip install cvxpy numpy pandas osqp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd


Array = np.ndarray


class InfeasiblePlanError(ValueError):
    """Raised when hard constraints make full completion impossible."""


def _as_datetime_index(dates: Sequence[pd.Timestamp | str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()


def _make_psd(matrix: Array, jitter: float = 1e-10) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    min_eig = float(np.linalg.eigvalsh(matrix).min())
    if min_eig < jitter:
        matrix = matrix + (jitter - min_eig) * np.eye(matrix.shape[0])
    return matrix


def _safe_numeric(values: Array, floor: float = 1.0) -> Array:
    values = np.asarray(values, dtype=float)
    return np.maximum(values, floor)


def days_to_next_event(
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    event_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str],
) -> pd.DataFrame:
    """
    Return trading/business days from each planner date to each symbol's next event.

    If the event is on one of the planner dates, the distance is the number of
    planner rows until that date. Otherwise it falls back to business-day count.
    Missing future events are set to +inf, so plugin multipliers naturally become
    neutral far from events.
    """
    dates = dates.normalize()
    date_positions = {date: idx for idx, date in enumerate(dates)}
    result = pd.DataFrame(np.inf, index=dates, columns=list(symbols), dtype=float)

    for symbol in symbols:
        raw = event_dates.get(symbol)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        if isinstance(raw, (str, pd.Timestamp)):
            events = [pd.Timestamp(raw).normalize()]
        else:
            events = [pd.Timestamp(value).normalize() for value in raw]
        events = sorted(event for event in events if not pd.isna(event))

        for date in dates:
            future_events = [event for event in events if event >= date]
            if not future_events:
                continue
            event = future_events[0]
            if event in date_positions:
                days = date_positions[event] - date_positions[date]
            else:
                days = np.busday_count(date.date(), event.date())
            result.loc[date, symbol] = max(float(days), 0.0)
    return result


@dataclass(frozen=True)
class PlannerContext:
    """Normalized inputs shared by all model plugins."""

    symbols: list[str]
    dates: pd.DatetimeIndex
    orders: pd.DataFrame
    panel: pd.DataFrame
    price: Array
    adv_shares: Array
    is_open: Array
    base_participation: Array
    event_days: pd.DataFrame


def build_context(
    orders: pd.DataFrame,
    dates: Sequence[pd.Timestamp | str],
    market: pd.DataFrame | None = None,
    event_dates: Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str] | None = None,
    default_participation: float = 0.15,
) -> PlannerContext:
    """
    Normalize order-level and date-symbol-level data into arrays.

    orders index:
        symbols

    required order columns:
        target_shares, price, adv_shares

    optional order columns:
        base_participation, daily_vol, event_vol

    market index:
        MultiIndex(date, symbol)

    optional market columns:
        price, adv_shares, is_open, base_participation
    """
    orders = orders.copy()
    orders.index = orders.index.astype(str)

    required = {"target_shares", "price", "adv_shares"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders is missing required columns: {sorted(missing)}")

    symbols = list(orders.index.astype(str))
    dates_idx = _as_datetime_index(dates)
    idx = pd.MultiIndex.from_product([dates_idx, symbols], names=["date", "symbol"])

    defaults = pd.DataFrame(index=idx)
    defaults["price"] = np.tile(orders["price"].to_numpy(float), len(dates_idx))
    defaults["adv_shares"] = np.tile(orders["adv_shares"].to_numpy(float), len(dates_idx))
    defaults["is_open"] = True
    if "base_participation" in orders:
        base_part = orders["base_participation"].to_numpy(float)
    else:
        base_part = np.full(len(symbols), default_participation, dtype=float)
    defaults["base_participation"] = np.tile(base_part, len(dates_idx))

    if market is None:
        panel = defaults
    else:
        market = market.copy()
        if not isinstance(market.index, pd.MultiIndex):
            raise ValueError("market must be indexed by MultiIndex(date, symbol)")
        market.index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d).normalize(), str(s)) for d, s in market.index],
            names=["date", "symbol"],
        )
        panel = market.reindex(idx).combine_first(defaults)

    t_count, n_names = len(dates_idx), len(symbols)
    price = panel["price"].to_numpy(float).reshape(t_count, n_names)
    adv = panel["adv_shares"].to_numpy(float).reshape(t_count, n_names)
    is_open = panel["is_open"].astype(bool).to_numpy().reshape(t_count, n_names)
    base_participation = panel["base_participation"].to_numpy(float).reshape(t_count, n_names)

    if event_dates is None and "earnings_date" in orders:
        event_dates = {
            symbol: orders.loc[symbol, "earnings_date"]
            for symbol in symbols
            if not pd.isna(orders.loc[symbol, "earnings_date"])
        }
    event_days = days_to_next_event(dates_idx, symbols, event_dates or {})

    return PlannerContext(
        symbols=symbols,
        dates=dates_idx,
        orders=orders,
        panel=panel,
        price=price,
        adv_shares=adv,
        is_open=is_open,
        base_participation=base_participation,
        event_days=event_days,
    )


class ParticipationModifier(Protocol):
    """Plugin that multiplies the base participation cap by a T x N matrix."""

    def multiplier(self, ctx: PlannerContext) -> Array:
        ...


@dataclass(frozen=True)
class ParticipationCapModel:
    """Build per-date, per-symbol share caps."""

    modifiers: Sequence[ParticipationModifier] = ()

    def caps(self, ctx: PlannerContext) -> Array:
        cap = ctx.base_participation * ctx.adv_shares * ctx.is_open.astype(float)
        for modifier in self.modifiers:
            cap = cap * np.clip(modifier.multiplier(ctx), 0.0, 1.0)
        return np.maximum(cap, 0.0)


@dataclass(frozen=True)
class LogisticEarningsParticipation:
    """
    Smoothly reduce participation as earnings approaches.

    h(d) = h_min + (1 - h_min) / (1 + exp(-steepness * (d - midpoint_days)))

    d is trading/business days to the next earnings date. Far from earnings,
    h(d) approaches 1. Near earnings, it approaches h_min.
    """

    h_min: float = 0.25
    midpoint_days: float = 5.0
    steepness: float = 1.0

    def multiplier(self, ctx: PlannerContext) -> Array:
        d = ctx.event_days.to_numpy(float)
        finite = np.isfinite(d)
        z = np.zeros_like(d, dtype=float)
        z[finite] = 1.0 / (1.0 + np.exp(-self.steepness * (d[finite] - self.midpoint_days)))
        z[~finite] = 1.0
        return self.h_min + (1.0 - self.h_min) * z


@dataclass(frozen=True)
class PiecewiseEarningsParticipation:
    """
    Step-rule participation modifier.

    thresholds are interpreted as (max_days_to_event, multiplier), sorted from
    nearest to farthest. Example: ((5, 0.25), (10, 0.5)) means 25% cap within
    five days, 50% cap within ten days, and 100% otherwise.
    """

    thresholds: Sequence[tuple[float, float]] = ((5.0, 0.25), (10.0, 0.5))

    def multiplier(self, ctx: PlannerContext) -> Array:
        d = ctx.event_days.to_numpy(float)
        out = np.ones_like(d, dtype=float)
        for max_days, value in sorted(self.thresholds, key=lambda item: item[0], reverse=True):
            out[d <= max_days] = value
        out[~np.isfinite(d)] = 1.0
        return out


class RiskOverlay(Protocol):
    """Plugin that adds a PSD covariance adjustment for one date."""

    def covariance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        ...


@dataclass(frozen=True)
class StaticCovarianceRiskModel:
    """Return a base covariance plus optional date-dependent overlays."""

    covariance: pd.DataFrame | Array | None = None
    overlays: Sequence[RiskOverlay] = ()

    def base_covariance(self, ctx: PlannerContext) -> Array:
        n_names = len(ctx.symbols)
        if self.covariance is None:
            if "daily_vol" in ctx.orders:
                vol = ctx.orders["daily_vol"].reindex(ctx.symbols).to_numpy(float)
            else:
                vol = np.full(n_names, 0.02, dtype=float)
            return np.diag(vol**2)
        if isinstance(self.covariance, pd.DataFrame):
            matrix = self.covariance.reindex(index=ctx.symbols, columns=ctx.symbols).to_numpy(float)
        else:
            matrix = np.asarray(self.covariance, dtype=float)
        if matrix.shape != (n_names, n_names):
            raise ValueError(f"covariance shape {matrix.shape} does not match ({n_names}, {n_names})")
        return _make_psd(matrix)

    def covariance_for_date(self, ctx: PlannerContext, date_index: int) -> Array:
        matrix = self.base_covariance(ctx).copy()
        for overlay in self.overlays:
            matrix = matrix + overlay.covariance_addition(ctx, date_index)
        return _make_psd(matrix)


@dataclass(frozen=True)
class ExponentialEarningsRiskOverlay:
    """
    Add single-name event variance that grows as earnings approaches.

    addition_i(d) = event_vol_i^2 * exp(-d / tau_days)
    """

    event_vol_column: str = "event_vol"
    tau_days: float = 5.0

    def covariance_addition(self, ctx: PlannerContext, date_index: int) -> Array:
        n_names = len(ctx.symbols)
        if self.event_vol_column in ctx.orders:
            event_vol = ctx.orders[self.event_vol_column].reindex(ctx.symbols).fillna(0.0).to_numpy(float)
        else:
            event_vol = np.zeros(n_names, dtype=float)

        d = ctx.event_days.iloc[date_index].to_numpy(float)
        weight = np.zeros(n_names, dtype=float)
        finite = np.isfinite(d)
        weight[finite] = np.exp(-d[finite] / self.tau_days)
        return np.diag((event_vol**2) * weight)


class CostTerm(Protocol):
    """Plugin that returns the cvxpy cost expression for one date's trades."""

    def objective(
        self,
        trade: cp.Expression,
        ctx: PlannerContext,
        date_index: int,
    ) -> cp.Expression:
        ...


@dataclass(frozen=True)
class CompositeCostModel:
    terms: Sequence[CostTerm]

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        total: cp.Expression | float = 0.0
        for term in self.terms:
            total = total + term.objective(trade, ctx, date_index)
        return total


@dataclass(frozen=True)
class QuadraticParticipationImpact:
    """
    Simple convex impact model.

    The coefficient is calibrated so trading 10% ADV costs roughly
    impact_bps_at_10pct_adv times traded dollars, then represented as a
    quadratic cost in shares.
    """

    impact_bps_at_10pct_adv: float = 5.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        price = ctx.price[date_index]
        adv = _safe_numeric(ctx.adv_shares[date_index])
        eta = (self.impact_bps_at_10pct_adv / 10000.0) * price / _safe_numeric(0.10 * adv)
        return cp.sum(cp.multiply(eta, cp.square(trade)))


@dataclass(frozen=True)
class LinearBpsCost:
    """Linear traded-dollar cost for spread, commission, fees, or soft penalties."""

    bps: float = 1.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        price = ctx.price[date_index]
        return (self.bps / 10000.0) * cp.sum(cp.multiply(price, cp.abs(trade)))


@dataclass(frozen=True)
class EarningsLinearPenalty:
    """
    Optional soft penalty for trading near earnings.

    This is separate from participation caps. Use it when the desk prefers to
    avoid event windows but still wants the optimizer to trade there if needed.
    """

    max_bps: float = 5.0
    tau_days: float = 5.0

    def objective(self, trade: cp.Expression, ctx: PlannerContext, date_index: int) -> cp.Expression:
        d = ctx.event_days.iloc[date_index].to_numpy(float)
        weight = np.zeros(len(ctx.symbols), dtype=float)
        finite = np.isfinite(d)
        weight[finite] = np.exp(-d[finite] / self.tau_days)
        price = ctx.price[date_index]
        return cp.sum(cp.multiply((self.max_bps / 10000.0) * weight * price, cp.abs(trade)))


@dataclass(frozen=True)
class TradePlannerConfig:
    participation_model: ParticipationCapModel
    risk_model: StaticCovarianceRiskModel
    cost_model: CompositeCostModel
    residual_risk_weight: float = 1.0
    terminal_penalty: float = 1e8
    hard_complete: bool = True
    direction_constraint: bool = True
    solver: str = "OSQP"


@dataclass(frozen=True)
class TradePlannerResult:
    schedule: pd.DataFrame
    diagnostics: dict[str, float | str]


class TradePlanner:
    """Core optimizer. Model behavior is controlled by config plugins."""

    def __init__(self, config: TradePlannerConfig):
        self.config = config

    def solve(self, ctx: PlannerContext) -> TradePlannerResult:
        symbols = ctx.symbols
        dates = ctx.dates
        t_count, n_names = len(dates), len(symbols)
        target = ctx.orders["target_shares"].reindex(symbols).to_numpy(float)

        caps = self.config.participation_model.caps(ctx)
        if self.config.hard_complete:
            self._check_completion_capacity(symbols, target, caps)

        trades = cp.Variable((t_count, n_names))
        constraints = [cp.abs(trades) <= caps]

        if self.config.direction_constraint:
            direction = np.sign(target)
            constraints.append(cp.multiply(np.tile(direction, (t_count, 1)), trades) >= 0)

        zero_target_idx = np.flatnonzero(target == 0)
        if len(zero_target_idx):
            constraints.append(trades[:, zero_target_idx] == 0)

        terminal_residual = target - cp.sum(trades, axis=0)
        if self.config.hard_complete:
            constraints.append(cp.sum(trades, axis=0) == target)

        objective_terms: list[cp.Expression] = []
        cumulative = 0
        for date_index in range(t_count):
            trade_t = trades[date_index, :]
            cumulative = cumulative + trade_t
            residual = target - cumulative

            residual_dollars = cp.multiply(ctx.price[date_index], residual)
            sigma = self.config.risk_model.covariance_for_date(ctx, date_index)
            objective_terms.append(
                self.config.residual_risk_weight * cp.quad_form(residual_dollars, sigma)
            )
            objective_terms.append(self.config.cost_model.objective(trade_t, ctx, date_index))

        if not self.config.hard_complete:
            terminal_dollars = cp.multiply(ctx.price[-1], terminal_residual)
            objective_terms.append(self.config.terminal_penalty * cp.sum_squares(terminal_dollars))

        total_objective: cp.Expression | float = 0.0
        for term in objective_terms:
            total_objective = total_objective + term

        problem = cp.Problem(cp.Minimize(total_objective), constraints)
        self._solve_problem(problem)

        trade_values = np.asarray(trades.value, dtype=float)
        residual_after = target[None, :] - np.cumsum(trade_values, axis=0)
        schedule = self._build_schedule(ctx, trade_values, residual_after, caps)

        diagnostics = {
            "status": problem.status,
            "objective": float(problem.value),
            "max_abs_terminal_residual": float(np.max(np.abs(residual_after[-1]))),
        }
        return TradePlannerResult(schedule=schedule, diagnostics=diagnostics)

    @staticmethod
    def _check_completion_capacity(symbols: Sequence[str], target: Array, caps: Array) -> None:
        capacity = caps.sum(axis=0)
        shortfall = np.abs(target) - capacity
        bad = np.flatnonzero(shortfall > 1e-8)
        if len(bad):
            details = {
                symbols[i]: {
                    "required_abs_shares": float(abs(target[i])),
                    "available_capacity": float(capacity[i]),
                    "shortfall": float(shortfall[i]),
                }
                for i in bad
            }
            raise InfeasiblePlanError(f"Insufficient capacity for hard completion: {details}")

    def _solve_problem(self, problem: cp.Problem) -> None:
        try:
            problem.solve(solver=self.config.solver, warm_start=True)
        except cp.SolverError:
            problem.solve(solver="CLARABEL", warm_start=True)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")

    @staticmethod
    def _build_schedule(
        ctx: PlannerContext,
        trades: Array,
        residual_after: Array,
        caps: Array,
    ) -> pd.DataFrame:
        records = []
        for t_idx, date in enumerate(ctx.dates):
            for s_idx, symbol in enumerate(ctx.symbols):
                price = ctx.price[t_idx, s_idx]
                adv = max(ctx.adv_shares[t_idx, s_idx], 1.0)
                trade = trades[t_idx, s_idx]
                records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "trade_shares": trade,
                        "trade_dollars": trade * price,
                        "abs_pct_adv": abs(trade) / adv,
                        "cap_shares": caps[t_idx, s_idx],
                        "cap_pct_adv": caps[t_idx, s_idx] / adv,
                        "days_to_earnings": ctx.event_days.iloc[t_idx, s_idx],
                        "residual_shares_after": residual_after[t_idx, s_idx],
                        "residual_dollars_after": residual_after[t_idx, s_idx] * price,
                    }
                )
        return pd.DataFrame.from_records(records)


def default_earnings_aware_config() -> TradePlannerConfig:
    """Reasonable default config with both requested earnings enhancements."""
    return TradePlannerConfig(
        participation_model=ParticipationCapModel(
            modifiers=[
                LogisticEarningsParticipation(
                    h_min=0.25,
                    midpoint_days=5.0,
                    steepness=1.0,
                )
            ]
        ),
        risk_model=StaticCovarianceRiskModel(
            covariance=None,
            overlays=[
                ExponentialEarningsRiskOverlay(
                    event_vol_column="event_vol",
                    tau_days=5.0,
                )
            ],
        ),
        cost_model=CompositeCostModel(
            terms=[
                QuadraticParticipationImpact(impact_bps_at_10pct_adv=5.0),
                LinearBpsCost(bps=1.0),
            ]
        ),
        residual_risk_weight=1.0,
        hard_complete=True,
        direction_constraint=True,
        solver="OSQP",
    )


def example() -> TradePlannerResult:
    """Run a tiny synthetic example."""
    dates = pd.bdate_range("2026-07-01", periods=8)
    orders = pd.DataFrame(
        {
            "target_shares": [90_000, -65_000, 30_000],
            "price": [50.0, 80.0, 25.0],
            "adv_shares": [1_000_000, 500_000, 250_000],
            "base_participation": [0.20, 0.15, 0.20],
            "daily_vol": [0.025, 0.030, 0.035],
            "event_vol": [0.06, 0.08, 0.05],
            "earnings_date": ["2026-07-08", "2026-07-20", "2026-07-03"],
        },
        index=["AAA", "BBB", "CCC"],
    )
    ctx = build_context(orders=orders, dates=dates)
    planner = TradePlanner(default_earnings_aware_config())
    return planner.solve(ctx)


if __name__ == "__main__":
    result = example()
    print(result.diagnostics)
    print(result.schedule.round(4).to_string(index=False))
