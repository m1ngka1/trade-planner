"""Point-in-time replay and realized economics for rebalance schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .context import PlannerContext
from .downside import weighted_loss_var_cvar
from .types import Array
from .utils import safe_numeric


@dataclass(frozen=True)
class PointInTimeRebalanceEvent:
    """One immutable planning snapshot followed by out-of-sample realizations.

    ``information_cutoff`` certifies when every field in ``ctx`` was observable.
    Realized returns and costs are kept outside ``ctx`` so an optimizer cannot
    accidentally consume them.  Returns are the holding return earned after
    each planner date, matching the accumulated-inventory objective convention.
    """

    event_id: str
    as_of: pd.Timestamp | str
    information_cutoff: pd.Timestamp | str
    ctx: PlannerContext
    realized_returns: Array
    realized_impact_bps_at_10pct_adv: float | Array
    realized_linear_cost_bps: float | Array
    realized_available_at: pd.Timestamp | str
    realized_adv_shares: Array | None = None


@dataclass(frozen=True)
class RealizedRebalanceMetrics:
    gross_holding_pnl_dollars: float
    impact_cost_dollars: float
    linear_cost_dollars: float
    net_pnl_dollars: float
    parent_gross_dollars: float
    net_pnl_bps: float
    within_event_max_drawdown_dollars: float
    within_event_max_drawdown_bps: float
    terminal_completion_error_shares: float
    max_realized_participation_rate: float
    p95_realized_participation_rate: float
    max_realized_participation_excess_shares: float

    def as_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class WalkForwardReplay:
    events: pd.DataFrame
    summary: pd.DataFrame
    daily: pd.DataFrame


ReplayStrategy = Callable[[PointInTimeRebalanceEvent], object]


def validate_point_in_time_event(event: PointInTimeRebalanceEvent) -> None:
    """Reject malformed or chronologically leaked replay inputs."""

    if not str(event.event_id).strip():
        raise ValueError("event_id must be non-empty")
    as_of = pd.Timestamp(event.as_of)
    cutoff = pd.Timestamp(event.information_cutoff)
    realized_at = pd.Timestamp(event.realized_available_at)
    if cutoff > as_of:
        raise ValueError("information_cutoff must be on or before as_of")
    if as_of.normalize() > event.ctx.dates[0]:
        raise ValueError("as_of must be on or before the first planner date")
    if realized_at <= event.ctx.dates[-1]:
        raise ValueError("realized_available_at must be after the final planner date")

    expected_shape = (len(event.ctx.dates), len(event.ctx.symbols))
    returns = np.asarray(event.realized_returns, dtype=float)
    if returns.shape != expected_shape or not np.all(np.isfinite(returns)):
        raise ValueError(
            f"realized_returns must contain finite values with shape {expected_shape}"
        )
    _realized_cost_matrix(
        event.realized_impact_bps_at_10pct_adv,
        event.ctx,
        "realized_impact_bps_at_10pct_adv",
    )
    _realized_cost_matrix(
        event.realized_linear_cost_bps,
        event.ctx,
        "realized_linear_cost_bps",
    )
    _realized_adv_matrix(event)


def evaluate_realized_rebalance_schedule(
    event: PointInTimeRebalanceEvent,
    schedule: pd.DataFrame,
) -> tuple[RealizedRebalanceMetrics, pd.DataFrame]:
    """Measure actual holding P&L and TCA costs without forecast reuse."""

    validate_point_in_time_event(event)
    ctx = event.ctx
    trades = _trade_matrix(ctx, schedule)
    cumulative = np.cumsum(trades, axis=0)
    returns = np.asarray(event.realized_returns, dtype=float)
    impact_bps = _realized_cost_matrix(
        event.realized_impact_bps_at_10pct_adv,
        ctx,
        "realized_impact_bps_at_10pct_adv",
    )
    linear_bps = _realized_cost_matrix(
        event.realized_linear_cost_bps,
        ctx,
        "realized_linear_cost_bps",
    )
    realized_adv = _realized_adv_matrix(event)

    daily_holding = np.sum(cumulative * ctx.price * returns, axis=1)
    eta = (
        (impact_bps / 10_000.0)
        * ctx.price
        / safe_numeric(0.10 * realized_adv)
    )
    daily_impact = np.sum(eta * np.square(trades), axis=1)
    daily_linear = np.sum(
        (linear_bps / 10_000.0) * ctx.price * np.abs(trades),
        axis=1,
    )
    daily_net = daily_holding - daily_impact - daily_linear
    realized_participation = np.abs(trades) / safe_numeric(realized_adv)
    realized_capacity = (
        ctx.base_participation
        * realized_adv
        * np.asarray(ctx.is_open, dtype=float)
    )
    realized_capacity_excess = np.maximum(
        np.abs(trades) - realized_capacity,
        0.0,
    )
    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    parent_gross = float(np.sum(np.abs(target * ctx.price[0])))
    terminal_error = float(np.sum(np.abs(np.sum(trades, axis=0) - target)))
    drawdown = _maximum_drawdown(daily_net)
    net = float(np.sum(daily_net))
    metrics = RealizedRebalanceMetrics(
        gross_holding_pnl_dollars=float(np.sum(daily_holding)),
        impact_cost_dollars=float(np.sum(daily_impact)),
        linear_cost_dollars=float(np.sum(daily_linear)),
        net_pnl_dollars=net,
        parent_gross_dollars=parent_gross,
        net_pnl_bps=10_000.0 * net / max(parent_gross, 1e-12),
        within_event_max_drawdown_dollars=drawdown,
        within_event_max_drawdown_bps=(
            10_000.0 * drawdown / max(parent_gross, 1e-12)
        ),
        terminal_completion_error_shares=terminal_error,
        max_realized_participation_rate=float(np.max(realized_participation)),
        p95_realized_participation_rate=float(
            np.quantile(realized_participation, 0.95)
        ),
        max_realized_participation_excess_shares=float(
            np.max(realized_capacity_excess)
        ),
    )
    daily = pd.DataFrame(
        {
            "date": ctx.dates,
            "holding_pnl_dollars": daily_holding,
            "impact_cost_dollars": daily_impact,
            "linear_cost_dollars": daily_linear,
            "net_pnl_dollars": daily_net,
            "cumulative_net_pnl_dollars": np.cumsum(daily_net),
            "max_realized_participation_rate": np.max(
                realized_participation,
                axis=1,
            ),
            "p95_realized_participation_rate": np.quantile(
                realized_participation,
                0.95,
                axis=1,
            ),
            "max_realized_participation_excess_shares": np.max(
                realized_capacity_excess,
                axis=1,
            ),
        }
    )
    return metrics, daily


def replay_rebalance_events(
    events: list[PointInTimeRebalanceEvent] | tuple[PointInTimeRebalanceEvent, ...],
    strategies: Mapping[str, ReplayStrategy],
) -> WalkForwardReplay:
    """Run multiple strategies on chronological, point-in-time event snapshots."""

    if not events:
        raise ValueError("events must contain at least one point-in-time event")
    if not strategies or any(not str(name).strip() for name in strategies):
        raise ValueError("strategies must contain non-empty names")
    event_ids = [str(event.event_id) for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event_id values must be unique")
    ordered = sorted(events, key=lambda event: (pd.Timestamp(event.as_of), str(event.event_id)))
    if event_ids != [str(event.event_id) for event in ordered]:
        raise ValueError("events must be ordered chronologically by as_of")

    event_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    for event in events:
        validate_point_in_time_event(event)
        for strategy_name, strategy in strategies.items():
            schedule = _extract_schedule(strategy(event))
            metrics, daily = evaluate_realized_rebalance_schedule(event, schedule)
            event_rows.append(
                {
                    "event_id": str(event.event_id),
                    "as_of": pd.Timestamp(event.as_of),
                    "strategy": str(strategy_name),
                    **metrics.as_dict(),
                }
            )
            daily_frames.append(
                daily.assign(
                    event_id=str(event.event_id),
                    strategy=str(strategy_name),
                )
            )
    event_frame = pd.DataFrame(event_rows)
    summaries = [
        _strategy_summary(strategy, rows.sort_values(["as_of", "event_id"]))
        for strategy, rows in event_frame.groupby("strategy", sort=False)
    ]
    return WalkForwardReplay(
        events=event_frame,
        summary=pd.DataFrame(summaries),
        daily=pd.concat(daily_frames, ignore_index=True),
    )


def _strategy_summary(strategy: str, rows: pd.DataFrame) -> dict[str, object]:
    net_bps = rows["net_pnl_bps"].to_numpy(float)
    if len(net_bps) == 1:
        loss_var = loss_cvar = -float(net_bps[0])
    else:
        weights = np.full(len(net_bps), 1.0 / len(net_bps))
        loss_var, loss_cvar = weighted_loss_var_cvar(net_bps, weights)
    return {
        "strategy": strategy,
        "event_count": len(rows),
        "total_net_pnl_dollars": float(rows["net_pnl_dollars"].sum()),
        "mean_net_pnl_bps": float(np.mean(net_bps)),
        "median_net_pnl_bps": float(np.median(net_bps)),
        "pnl_vol_bps": float(np.std(net_bps, ddof=1)) if len(net_bps) > 1 else 0.0,
        "loss_var_95_bps": loss_var,
        "loss_cvar_95_bps": loss_cvar,
        "probability_profitable": float(np.mean(net_bps > 0)),
        "worst_event_pnl_bps": float(np.min(net_bps)),
        "event_sequence_max_drawdown_bps": _maximum_drawdown(net_bps),
        "mean_within_event_max_drawdown_bps": float(
            rows["within_event_max_drawdown_bps"].mean()
        ),
        "max_terminal_completion_error_shares": float(
            rows["terminal_completion_error_shares"].max()
        ),
        "max_realized_participation_rate": float(
            rows["max_realized_participation_rate"].max()
        ),
        "mean_p95_realized_participation_rate": float(
            rows["p95_realized_participation_rate"].mean()
        ),
        "max_realized_participation_excess_shares": float(
            rows["max_realized_participation_excess_shares"].max()
        ),
    }


def _extract_schedule(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    result = getattr(value, "result", value)
    schedule = getattr(result, "schedule", None)
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("a replay strategy must return a schedule, result, or calibrated plan")
    return schedule


def _trade_matrix(ctx: PlannerContext, schedule: pd.DataFrame) -> np.ndarray:
    required = {"date", "symbol", "trade_shares"}
    missing = required.difference(schedule.columns)
    if missing:
        raise ValueError(f"schedule is missing required columns: {sorted(missing)}")
    unknown_dates = set(pd.to_datetime(schedule["date"]).dt.normalize()).difference(ctx.dates)
    unknown_symbols = set(schedule["symbol"].astype(str)).difference(ctx.symbols)
    if unknown_dates or unknown_symbols:
        raise ValueError("schedule contains dates or symbols outside the event context")
    return (
        schedule.assign(
            date=pd.to_datetime(schedule["date"]).dt.normalize(),
            symbol=schedule["symbol"].astype(str),
        )
        .pivot_table(
            index="date",
            columns="symbol",
            values="trade_shares",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=ctx.dates, columns=ctx.symbols, fill_value=0.0)
        .to_numpy(float)
    )


def _realized_cost_matrix(
    value: float | Array,
    ctx: PlannerContext,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim == 0:
        matrix = np.full((len(ctx.dates), len(ctx.symbols)), float(matrix))
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must be scalar or have shape {expected_shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return matrix


def _realized_adv_matrix(event: PointInTimeRebalanceEvent) -> np.ndarray:
    value = (
        event.ctx.adv_shares
        if event.realized_adv_shares is None
        else event.realized_adv_shares
    )
    matrix = np.asarray(value, dtype=float)
    expected_shape = (len(event.ctx.dates), len(event.ctx.symbols))
    if matrix.shape != expected_shape:
        raise ValueError(f"realized_adv_shares must have shape {expected_shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
        raise ValueError("realized_adv_shares must contain finite positive values")
    return matrix


def _maximum_drawdown(increments: np.ndarray) -> float:
    path = np.concatenate(([0.0], np.cumsum(np.asarray(increments, dtype=float))))
    return float(np.max(np.maximum.accumulate(path) - path))
