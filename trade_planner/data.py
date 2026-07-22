"""Provider-driven construction of planner context from minimal orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .context import PlannerContext, as_datetime_index, days_to_next_event
from .types import Array


@dataclass(frozen=True)
class FactorRiskData:
    """
    Barra-style factor risk inputs for the planner date range.

    factor_exposure:
        DataFrame indexed by (date, symbol) with factor columns, or indexed by
        symbol for static exposures.

    factor_covariance:
        Either a date -> factor covariance mapping, a static factor covariance
        DataFrame/array, or a T x K x K array.

    specific_variance:
        Specific return variance by date-symbol, or static symbol Series.
    """

    factor_exposure: pd.DataFrame
    factor_covariance: pd.DataFrame | Mapping[pd.Timestamp | str, pd.DataFrame | Array] | Array
    specific_variance: pd.DataFrame | pd.Series | Array


class PlannerDataProvider:
    """
    Provider adapter for Bloomberg/Barra/Axioma/internal data loaders.

    Subclass this and fill in the placeholders with your data source calls.
    Each method returns one context field or one input used to derive a context
    field, so adding future context data should mean adding one provider method
    plus one alignment call in this module.
    """

    def load_price(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame | pd.Series | Array:
        raise NotImplementedError("load_price must return price by date-symbol or static symbol prices")

    def load_adv_shares(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame | pd.Series | Array:
        raise NotImplementedError("load_adv_shares must return ADV shares by date-symbol or static symbol ADV")

    def load_is_open(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame | pd.Series | Array | bool:
        return True

    def load_base_participation(
        self,
        symbols: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame | pd.Series | Array | float | None:
        return None

    def load_event_days(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
        event_dates = self.load_event_dates(symbols, dates[0], dates[-1])
        return days_to_next_event(dates, symbols, event_dates)

    def load_event_dates(
        self,
        symbols: Sequence[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Mapping[str, Sequence[pd.Timestamp | str] | pd.Timestamp | str]:
        return {}

    def load_factor_exposure(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
        raise NotImplementedError("load_factor_exposure must return security-by-factor exposures")

    def load_factor_covariance(
        self,
        factor_names: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame | Mapping[pd.Timestamp | str, pd.DataFrame | Array] | Array:
        raise NotImplementedError("load_factor_covariance must return factor covariance")

    def load_specific_variance(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame | pd.Series | Array:
        raise NotImplementedError("load_specific_variance must return specific return variance")

    def load_event_volatility(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> pd.Series | Array | None:
        return None

    def load_expected_return(
        self,
        symbols: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame | pd.Series | Array | None:
        """Return probability-weighted expected holding returns, if available."""
        return None

    def load_impact_bps_at_10pct_adv(
        self,
        symbols: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame | pd.Series | Array | None:
        """Return date-by-name TCA impact forecasts, if available."""
        return None

    def load_linear_cost_bps(
        self,
        symbols: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame | pd.Series | Array | None:
        """Return date-by-name spread, fee, and commission forecasts."""
        return None

    def load_return_residual_scenarios(
        self,
        symbols: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> Array | None:
        """Return scenario-by-date-by-name residual holding returns, if available.

        The scenario mean is removed by the downside-risk model, so expected
        rebalance alpha remains exclusively in ``load_expected_return``.
        """
        return None

    def load_return_scenario_weights(
        self,
        symbols: Sequence[str],
        dates: pd.DatetimeIndex,
    ) -> Array | None:
        """Return optional probabilities for residual-return scenarios."""
        return None

    def load_factor_risk_data(self, symbols: Sequence[str], dates: pd.DatetimeIndex) -> FactorRiskData:
        factor_exposure = self.load_factor_exposure(symbols, dates)
        factor_names = list(factor_exposure.columns.astype(str))
        return FactorRiskData(
            factor_exposure=factor_exposure,
            factor_covariance=self.load_factor_covariance(factor_names, dates),
            specific_variance=self.load_specific_variance(symbols, dates),
        )


def build_context_from_provider(
    orders: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    provider: PlannerDataProvider,
    default_participation: float = 0.15,
) -> PlannerContext:
    """
    Build context when the user supplies only symbols and target shares.

    The provider is responsible for querying all context fields for the
    requested symbols and date range.
    """
    orders = normalize_orders(orders)
    dates = build_planner_dates(start_date, end_date)
    symbols = list(orders.index)

    market_panel = build_market_panel_from_provider(
        provider=provider,
        symbols=symbols,
        dates=dates,
        default_participation=default_participation,
    )
    event_days = align_date_symbol_frame(
        provider.load_event_days(symbols, dates),
        dates=dates,
        symbols=symbols,
        name="event_days",
    )
    factor_risk_data = provider.load_factor_risk_data(symbols, dates)
    expected_return_raw = provider.load_expected_return(symbols, dates)
    expected_return = (
        align_date_symbol_field(
            expected_return_raw,
            dates=dates,
            symbols=symbols,
            name="expected_return",
        ).to_numpy(float)
        if expected_return_raw is not None
        else None
    )
    impact_bps_raw = provider.load_impact_bps_at_10pct_adv(symbols, dates)
    impact_bps_at_10pct_adv = (
        align_date_symbol_field(
            impact_bps_raw,
            dates=dates,
            symbols=symbols,
            name="impact_bps_at_10pct_adv",
        ).to_numpy(float)
        if impact_bps_raw is not None
        else None
    )
    linear_cost_raw = provider.load_linear_cost_bps(symbols, dates)
    linear_cost_bps = (
        align_date_symbol_field(
            linear_cost_raw,
            dates=dates,
            symbols=symbols,
            name="linear_cost_bps",
        ).to_numpy(float)
        if linear_cost_raw is not None
        else None
    )
    return_residual_scenarios = provider.load_return_residual_scenarios(symbols, dates)
    return_scenario_weights = provider.load_return_scenario_weights(symbols, dates)

    event_vol = provider.load_event_volatility(symbols, dates)
    if event_vol is not None:
        orders["event_vol"] = align_symbol_series(event_vol, symbols, "event_vol").to_numpy(float)

    return assemble_context(
        orders=orders,
        dates=dates,
        market_panel=market_panel,
        event_days=event_days,
        factor_risk_data=factor_risk_data,
        expected_return=expected_return,
        impact_bps_at_10pct_adv=impact_bps_at_10pct_adv,
        linear_cost_bps=linear_cost_bps,
        return_residual_scenarios=return_residual_scenarios,
        return_scenario_weights=return_scenario_weights,
    )


def normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Normalize user order input. Required user field is target_shares."""
    orders = orders.copy()
    orders.index = orders.index.astype(str)
    if "target_shares" not in orders.columns:
        raise ValueError("orders must contain target_shares")
    return orders


def build_planner_dates(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return as_datetime_index(pd.bdate_range(start, end))


def build_market_panel_from_provider(
    provider: PlannerDataProvider,
    symbols: Sequence[str],
    dates: pd.DatetimeIndex,
    default_participation: float = 0.15,
) -> pd.DataFrame:
    """Load and align the market fields needed by PlannerContext."""
    price = align_date_symbol_field(provider.load_price(symbols, dates), dates, symbols, "price")
    adv = align_date_symbol_field(provider.load_adv_shares(symbols, dates), dates, symbols, "adv_shares")
    is_open = align_date_symbol_field(
        provider.load_is_open(symbols, dates),
        dates,
        symbols,
        "is_open",
        default=True,
        dtype=bool,
    )
    base_participation = align_date_symbol_field(
        provider.load_base_participation(symbols, dates),
        dates,
        symbols,
        "base_participation",
        default=default_participation,
    )

    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    panel = pd.DataFrame(index=idx)
    panel["price"] = price.to_numpy(float).reshape(-1)
    panel["adv_shares"] = adv.to_numpy(float).reshape(-1)
    panel["is_open"] = is_open.to_numpy(bool).reshape(-1)
    panel["base_participation"] = base_participation.to_numpy(float).reshape(-1)
    return panel


def assemble_context(
    orders: pd.DataFrame,
    dates: pd.DatetimeIndex,
    market_panel: pd.DataFrame,
    event_days: pd.DataFrame,
    factor_risk_data: FactorRiskData,
    metadata: dict[str, object] | None = None,
    expected_return: Array | None = None,
    impact_bps_at_10pct_adv: Array | None = None,
    linear_cost_bps: Array | None = None,
    return_residual_scenarios: Array | None = None,
    return_scenario_weights: Array | None = None,
) -> PlannerContext:
    """Assemble the normalized PlannerContext from already-loaded fields."""
    orders = normalize_orders(orders)
    symbols = list(orders.index)
    dates = as_datetime_index(dates)

    market_panel = normalize_market_panel(market_panel, dates, symbols)
    price = field_array(market_panel, "price", dates, symbols, float)
    adv = field_array(market_panel, "adv_shares", dates, symbols, float)
    is_open = field_array(market_panel, "is_open", dates, symbols, bool)
    base_participation = field_array(market_panel, "base_participation", dates, symbols, float)

    event_days = align_date_symbol_frame(event_days, dates, symbols, "event_days")
    factor_names, factor_exposure = align_factor_exposure(
        factor_risk_data.factor_exposure,
        dates,
        symbols,
    )
    factor_covariance = align_factor_covariance(
        factor_risk_data.factor_covariance,
        dates,
        factor_names,
    )
    specific_variance = align_specific_variance(
        factor_risk_data.specific_variance,
        dates,
        symbols,
    )
    aligned_return_scenarios = align_return_scenarios(
        return_residual_scenarios,
        dates,
        symbols,
    )
    aligned_scenario_weights = align_scenario_weights(
        return_scenario_weights,
        None if aligned_return_scenarios is None else len(aligned_return_scenarios),
    )

    return PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=orders,
        panel=market_panel,
        price=price,
        adv_shares=adv,
        is_open=is_open,
        base_participation=base_participation,
        event_days=event_days,
        factor_names=factor_names,
        factor_exposure=factor_exposure,
        factor_covariance=factor_covariance,
        specific_variance=specific_variance,
        expected_return=(
            align_date_symbol_field(
                expected_return,
                dates,
                symbols,
                "expected_return",
            ).to_numpy(float)
            if expected_return is not None
            else None
        ),
        impact_bps_at_10pct_adv=(
            align_date_symbol_field(
                impact_bps_at_10pct_adv,
                dates,
                symbols,
                "impact_bps_at_10pct_adv",
            ).to_numpy(float)
            if impact_bps_at_10pct_adv is not None
            else None
        ),
        linear_cost_bps=(
            align_date_symbol_field(
                linear_cost_bps,
                dates,
                symbols,
                "linear_cost_bps",
            ).to_numpy(float)
            if linear_cost_bps is not None
            else None
        ),
        return_residual_scenarios=aligned_return_scenarios,
        return_scenario_weights=aligned_scenario_weights,
        metadata=metadata or {},
    )


def align_return_scenarios(
    value: Array | None,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> Array | None:
    """Validate already ordered scenario-by-date-by-symbol residual returns."""

    if value is None:
        return None
    scenarios = np.asarray(value, dtype=float)
    expected_tail = (len(dates), len(symbols))
    if scenarios.ndim != 3 or scenarios.shape[1:] != expected_tail:
        raise ValueError(
            "return_residual_scenarios must have shape "
            f"(scenario, date, symbol) with trailing shape {expected_tail}"
        )
    if len(scenarios) < 2 or not np.all(np.isfinite(scenarios)):
        raise ValueError(
            "return_residual_scenarios must contain at least two finite scenarios"
        )
    return scenarios.copy()


def align_scenario_weights(value: Array | None, n_scenarios: int | None) -> Array | None:
    """Validate optional scenario probabilities without forcing normalization."""

    if value is None:
        return None
    if n_scenarios is None:
        raise ValueError(
            "return_scenario_weights requires return_residual_scenarios"
        )
    weights = np.asarray(value, dtype=float)
    if weights.shape != (n_scenarios,):
        raise ValueError(
            "return_scenario_weights must contain one value per return scenario"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError(
            "return_scenario_weights must be finite, non-negative, and nonzero"
        )
    return weights.copy()


def normalize_market_panel(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("market panel must be indexed by MultiIndex(date, symbol)")
    panel = panel.copy()
    panel.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d).normalize(), str(s)) for d, s in panel.index],
        names=["date", "symbol"],
    )
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    panel = panel.reindex(idx)
    required = ["price", "adv_shares", "is_open", "base_participation"]
    for column in required:
        if column not in panel or panel[column].isna().any():
            raise ValueError(f"market panel is missing {column} for at least one date-symbol")
    return panel


def field_array(
    panel: pd.DataFrame,
    column: str,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    dtype: type,
) -> Array:
    return panel[column].to_numpy(dtype).reshape(len(dates), len(symbols))


def align_date_symbol_field(
    value: pd.DataFrame | pd.Series | Array | float | bool | None,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    name: str,
    default: float | bool | None = None,
    dtype: type = float,
) -> pd.DataFrame:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default

    if isinstance(value, (int, float, bool)):
        return pd.DataFrame(value, index=dates, columns=list(symbols), dtype=dtype)

    if isinstance(value, pd.Series):
        return pd.DataFrame(
            np.tile(align_symbol_series(value, symbols, name).to_numpy(dtype)[None, :], (len(dates), 1)),
            index=dates,
            columns=list(symbols),
        )

    if isinstance(value, pd.DataFrame):
        if isinstance(value.index, pd.MultiIndex):
            frame = value.copy()
            frame.index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp(d).normalize(), str(s)) for d, s in frame.index],
                names=["date", "symbol"],
            )
            if name in frame.columns:
                series = frame[name]
            elif frame.shape[1] == 1:
                series = frame.iloc[:, 0]
            else:
                raise ValueError(f"{name} MultiIndex frame must contain column {name}")
            idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
            aligned = series.reindex(idx)
            if aligned.isna().any():
                raise ValueError(f"{name} is missing values for at least one date-symbol")
            return pd.DataFrame(
                aligned.to_numpy(dtype).reshape(len(dates), len(symbols)),
                index=dates,
                columns=list(symbols),
            )

        frame = value.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        frame.columns = frame.columns.astype(str)
        aligned = frame.reindex(index=dates, columns=list(symbols))
        if aligned.isna().any().any():
            raise ValueError(f"{name} is missing values for at least one date-symbol")
        return aligned.astype(dtype)

    array = np.asarray(value, dtype=dtype)
    if array.shape == (len(symbols),):
        return pd.DataFrame(
            np.tile(array[None, :], (len(dates), 1)),
            index=dates,
            columns=list(symbols),
        )
    if array.shape == (len(dates), len(symbols)):
        return pd.DataFrame(array, index=dates, columns=list(symbols))
    raise ValueError(
        f"{name} shape {array.shape} does not match {(len(symbols),)} or {(len(dates), len(symbols))}"
    )


def align_date_symbol_frame(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    name: str,
) -> pd.DataFrame:
    return align_date_symbol_field(frame, dates, symbols, name)


def align_symbol_series(
    value: pd.Series | Array,
    symbols: Sequence[str],
    name: str,
) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
        series.index = series.index.astype(str)
    else:
        array = np.asarray(value, dtype=float)
        if array.shape != (len(symbols),):
            raise ValueError(f"{name} shape {array.shape} does not match {(len(symbols),)}")
        series = pd.Series(array, index=list(symbols))
    aligned = series.reindex(list(symbols))
    if aligned.isna().any():
        raise ValueError(f"{name} is missing values for at least one symbol")
    return aligned


def align_factor_exposure(
    exposure: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> tuple[list[str], Array]:
    exposure = exposure.copy()
    exposure.columns = exposure.columns.astype(str)
    factor_names = list(exposure.columns)

    if isinstance(exposure.index, pd.MultiIndex):
        exposure.index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d).normalize(), str(s)) for d, s in exposure.index],
            names=["date", "symbol"],
        )
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        aligned = exposure.reindex(idx)
        if aligned.isna().any().any():
            raise ValueError("factor_exposure is missing values for at least one date-symbol")
        array = aligned.to_numpy(float).reshape(len(dates), len(symbols), len(factor_names))
        return factor_names, array

    exposure.index = exposure.index.astype(str)
    aligned = exposure.reindex(list(symbols))
    if aligned.isna().any().any():
        raise ValueError("factor_exposure is missing values for at least one symbol")
    static = aligned.to_numpy(float)
    return factor_names, np.tile(static[None, :, :], (len(dates), 1, 1))


def align_factor_covariance(
    covariance: pd.DataFrame | Mapping[pd.Timestamp | str, pd.DataFrame | Array] | Array,
    dates: pd.DatetimeIndex,
    factor_names: Sequence[str],
) -> Array:
    k = len(factor_names)
    if isinstance(covariance, Mapping):
        values = []
        normalized = {pd.Timestamp(key).normalize(): value for key, value in covariance.items()}
        for date in dates:
            if date not in normalized:
                raise ValueError(f"factor_covariance missing date {date.date()}")
            values.append(coerce_factor_covariance_matrix(normalized[date], factor_names))
        return np.stack(values)

    if isinstance(covariance, pd.DataFrame):
        if isinstance(covariance.index, pd.MultiIndex):
            covariance = covariance.copy()
            covariance.index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp(d).normalize(), str(f)) for d, f in covariance.index],
                names=["date", "factor"],
            )
            values = []
            for date in dates:
                if date not in covariance.index.get_level_values("date"):
                    raise ValueError(f"factor_covariance missing date {date.date()}")
                matrix = covariance.loc[date].reindex(index=factor_names, columns=factor_names)
                values.append(matrix.to_numpy(float))
            return np.stack(values)
        return np.tile(
            coerce_factor_covariance_matrix(covariance, factor_names)[None, :, :],
            (len(dates), 1, 1),
        )

    array = np.asarray(covariance, dtype=float)
    if array.shape == (k, k):
        return np.tile(array[None, :, :], (len(dates), 1, 1))
    if array.shape == (len(dates), k, k):
        return array
    raise ValueError(
        f"factor_covariance shape {array.shape} does not match {(k, k)} or {(len(dates), k, k)}"
    )


def coerce_factor_covariance_matrix(value: pd.DataFrame | Array, factor_names: Sequence[str]) -> Array:
    if isinstance(value, pd.DataFrame):
        matrix = value.reindex(index=factor_names, columns=factor_names).to_numpy(float)
    else:
        matrix = np.asarray(value, dtype=float)
    expected = (len(factor_names), len(factor_names))
    if matrix.shape != expected:
        raise ValueError(f"factor covariance matrix shape {matrix.shape} does not match {expected}")
    return matrix


def align_specific_variance(
    specific_variance: pd.DataFrame | pd.Series | Array,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> Array:
    if isinstance(specific_variance, pd.Series):
        return np.tile(
            align_symbol_series(specific_variance, symbols, "specific_variance").to_numpy(float)[None, :],
            (len(dates), 1),
        )

    if isinstance(specific_variance, pd.DataFrame):
        frame = specific_variance.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame.index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp(d).normalize(), str(s)) for d, s in frame.index],
                names=["date", "symbol"],
            )
            column = "specific_variance"
            if column not in frame.columns:
                raise ValueError("MultiIndex specific_variance frame must contain a specific_variance column")
            idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
            aligned = frame.reindex(idx)[column]
            if aligned.isna().any():
                raise ValueError("specific_variance is missing values for at least one date-symbol")
            return aligned.to_numpy(float).reshape(len(dates), len(symbols))

        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        frame.columns = frame.columns.astype(str)
        aligned = frame.reindex(index=dates, columns=list(symbols))
        if aligned.isna().any().any():
            raise ValueError("specific_variance is missing values for at least one date-symbol")
        return aligned.to_numpy(float)

    array = np.asarray(specific_variance, dtype=float)
    if array.shape == (len(symbols),):
        return np.tile(array[None, :], (len(dates), 1))
    if array.shape == (len(dates), len(symbols)):
        return array
    raise ValueError(
        f"specific_variance shape {array.shape} does not match {(len(symbols),)} or {(len(dates), len(symbols))}"
    )
