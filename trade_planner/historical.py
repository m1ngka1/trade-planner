"""Auditable tabular loader for real point-in-time rebalance replays."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import FactorRiskData, assemble_context
from .walkforward import PointInTimeRebalanceEvent, validate_point_in_time_event


EVENT_FILE = "events.csv"
ORDER_FILE = "orders.csv"
PLANNING_FILE = "planning.csv"
FACTOR_COVARIANCE_FILE = "factor_covariance.csv"
REALIZED_FILE = "realized.csv"
SCENARIO_FILE = "scenarios.csv"
FACTOR_COLUMN_PREFIX = "factor:"
ALLOWED_COHORT_ROLES = frozenset({"development", "holdout", "backtest"})
LIQUIDITY_QUANTILE_BY_RISK = {"high": 0.10, "medium": 0.25, "low": 0.50}


@dataclass(frozen=True)
class HistoricalReplayBundle:
    """Loaded replay events plus evidence needed by the research evaluator."""

    role: str
    events: tuple[PointInTimeRebalanceEvent, ...]
    classifications: Mapping[str, pd.DataFrame]
    forecast_adv_quantiles: Mapping[str, Mapping[float, np.ndarray]]
    source_hashes: Mapping[str, str]

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(str(event.event_id) for event in self.events)

    def forecast_adv_for(
        self,
        event_id: str,
        risk_aversion: str,
    ) -> np.ndarray:
        """Return the point-in-time liquidity quantile selected by risk label."""

        risk = str(risk_aversion).strip().lower()
        if risk not in LIQUIDITY_QUANTILE_BY_RISK:
            raise ValueError("risk_aversion must be high, medium, or low")
        quantile = LIQUIDITY_QUANTILE_BY_RISK[risk]
        return self.forecast_adv_quantile(event_id, quantile)

    def forecast_adv_quantile(
        self,
        event_id: str,
        quantile: float,
    ) -> np.ndarray:
        """Return or log-interpolate an available point-in-time ADV quantile."""

        key = str(event_id)
        if key not in self.forecast_adv_quantiles:
            raise KeyError(f"unknown historical event_id: {key}")
        requested = float(quantile)
        available = sorted(float(value) for value in self.forecast_adv_quantiles[key])
        if (
            not np.isfinite(requested)
            or requested < available[0]
            or requested > available[-1]
        ):
            raise ValueError(
                f"liquidity quantile must be between {available[0]:g} and "
                f"{available[-1]:g}"
            )
        for value in available:
            if np.isclose(requested, value, rtol=0.0, atol=1e-12):
                return np.asarray(
                    self.forecast_adv_quantiles[key][value],
                    dtype=float,
                ).copy()
        upper_index = int(np.searchsorted(available, requested, side="right"))
        lower = available[upper_index - 1]
        upper = available[upper_index]
        weight = (requested - lower) / (upper - lower)
        lower_adv = np.asarray(
            self.forecast_adv_quantiles[key][lower],
            dtype=float,
        )
        upper_adv = np.asarray(
            self.forecast_adv_quantiles[key][upper],
            dtype=float,
        )
        return np.exp(
            (1.0 - weight) * np.log(lower_adv) + weight * np.log(upper_adv)
        )


def load_historical_replay_bundle(
    root: str | Path,
    *,
    expected_role: str,
) -> HistoricalReplayBundle:
    """Load one explicitly authorized development, holdout, or backtest bundle.

    The loader reads ``events.csv`` first and rejects a role mismatch before
    opening any detailed planning or realized file. This makes opening a
    holdout an explicit caller action instead of an accidental side effect.
    """

    directory = Path(root)
    role = str(expected_role).strip().lower()
    if role not in ALLOWED_COHORT_ROLES:
        allowed = ", ".join(sorted(ALLOWED_COHORT_ROLES))
        raise ValueError(f"expected_role must be one of: {allowed}")
    event_path = directory / EVENT_FILE
    if not event_path.is_file():
        raise FileNotFoundError(f"historical replay bundle is missing {EVENT_FILE}")
    manifest = pd.read_csv(event_path)
    _require_columns(
        manifest,
        {
            "event_id",
            "cohort_role",
            "as_of",
            "information_cutoff",
            "realized_available_at",
        },
        EVENT_FILE,
    )
    if manifest.empty:
        raise ValueError(f"{EVENT_FILE} must contain at least one event")
    manifest = manifest.copy()
    manifest["event_id"] = manifest["event_id"].astype(str)
    _require_unique(manifest, ["event_id"], EVENT_FILE)
    roles = set(manifest["cohort_role"].astype(str).str.strip().str.lower())
    if roles != {role}:
        raise ValueError(
            f"bundle cohort_role values {sorted(roles)} do not match "
            f"explicit expected_role={role!r}"
        )
    manifest["as_of"] = _timestamp_series(manifest["as_of"], "events.as_of")
    manifest["information_cutoff"] = _timestamp_series(
        manifest["information_cutoff"],
        "events.information_cutoff",
    )
    manifest["realized_available_at"] = _timestamp_series(
        manifest["realized_available_at"],
        "events.realized_available_at",
    )
    if not manifest["as_of"].is_monotonic_increasing:
        raise ValueError("events.csv must be ordered chronologically by as_of")

    required_paths = {
        name: directory / name
        for name in (
            EVENT_FILE,
            ORDER_FILE,
            PLANNING_FILE,
            FACTOR_COVARIANCE_FILE,
            REALIZED_FILE,
        )
    }
    for name, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"historical replay bundle is missing {name}")
    scenario_path = directory / SCENARIO_FILE
    paths = {
        **required_paths,
        **({SCENARIO_FILE: scenario_path} if scenario_path.is_file() else {}),
    }
    source_hashes = {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }

    orders = pd.read_csv(required_paths[ORDER_FILE])
    planning = pd.read_csv(required_paths[PLANNING_FILE])
    covariance = pd.read_csv(required_paths[FACTOR_COVARIANCE_FILE])
    realized = pd.read_csv(required_paths[REALIZED_FILE])
    scenarios = pd.read_csv(scenario_path) if scenario_path.is_file() else None
    _validate_tables(manifest, orders, planning, covariance, realized, scenarios)

    loaded_events: list[PointInTimeRebalanceEvent] = []
    classifications: dict[str, pd.DataFrame] = {}
    forecast_adv: dict[str, Mapping[float, np.ndarray]] = {}
    for manifest_row in manifest.itertuples(index=False):
        event_id = str(manifest_row.event_id)
        cutoff = pd.Timestamp(manifest_row.information_cutoff)
        event_orders = orders.loc[orders["event_id"].astype(str).eq(event_id)].copy()
        event_planning = planning.loc[
            planning["event_id"].astype(str).eq(event_id)
        ].copy()
        event_covariance = covariance.loc[
            covariance["event_id"].astype(str).eq(event_id)
        ].copy()
        event_realized = realized.loc[
            realized["event_id"].astype(str).eq(event_id)
        ].copy()
        event_scenarios = (
            scenarios.loc[scenarios["event_id"].astype(str).eq(event_id)].copy()
            if scenarios is not None
            else None
        )
        event, event_classifications, event_forecast_adv = _build_event(
            event_id=event_id,
            role=role,
            as_of=pd.Timestamp(manifest_row.as_of),
            cutoff=cutoff,
            realized_available_at=pd.Timestamp(manifest_row.realized_available_at),
            orders=event_orders,
            planning=event_planning,
            covariance=event_covariance,
            realized=event_realized,
            scenarios=event_scenarios,
            source_hashes=source_hashes,
        )
        loaded_events.append(event)
        classifications[event_id] = event_classifications
        forecast_adv[event_id] = event_forecast_adv

    return HistoricalReplayBundle(
        role=role,
        events=tuple(loaded_events),
        classifications=classifications,
        forecast_adv_quantiles=forecast_adv,
        source_hashes=source_hashes,
    )


def _validate_tables(
    manifest: pd.DataFrame,
    orders: pd.DataFrame,
    planning: pd.DataFrame,
    covariance: pd.DataFrame,
    realized: pd.DataFrame,
    scenarios: pd.DataFrame | None,
) -> None:
    _require_columns(
        orders,
        {
            "event_id",
            "symbol",
            "target_shares",
            "country",
            "sector",
            "industry",
            "urgency",
            "available_at",
        },
        ORDER_FILE,
    )
    planning_required = {
        "event_id",
        "date",
        "symbol",
        "price",
        "adv_shares",
        "forecast_adv_p10_shares",
        "forecast_adv_p25_shares",
        "forecast_adv_p50_shares",
        "is_open",
        "base_participation",
        "event_days",
        "specific_variance",
        "expected_return",
        "expected_return_uncertainty",
        "impact_bps_at_10pct_adv",
        "linear_cost_bps",
        "available_at",
    }
    _require_columns(planning, planning_required, PLANNING_FILE)
    factor_columns = [
        column for column in planning.columns if str(column).startswith(FACTOR_COLUMN_PREFIX)
    ]
    if not factor_columns:
        raise ValueError(
            f"{PLANNING_FILE} must contain at least one {FACTOR_COLUMN_PREFIX!r} column"
        )
    _require_columns(
        covariance,
        {
            "event_id",
            "date",
            "factor_left",
            "factor_right",
            "covariance",
            "available_at",
        },
        FACTOR_COVARIANCE_FILE,
    )
    _require_columns(
        realized,
        {
            "event_id",
            "date",
            "symbol",
            "realized_return",
            "realized_adv_shares",
            "realized_impact_bps_at_10pct_adv",
            "realized_linear_cost_bps",
            "available_at",
        },
        REALIZED_FILE,
    )
    if scenarios is not None:
        _require_columns(
            scenarios,
            {
                "event_id",
                "scenario",
                "date",
                "symbol",
                "residual_return",
                "scenario_weight",
                "available_at",
            },
            SCENARIO_FILE,
        )
    expected_ids = set(manifest["event_id"].astype(str))
    coverage_tables = [
        (ORDER_FILE, orders),
        (PLANNING_FILE, planning),
        (FACTOR_COVARIANCE_FILE, covariance),
        (REALIZED_FILE, realized),
    ]
    if scenarios is not None:
        coverage_tables.append((SCENARIO_FILE, scenarios))
    for name, frame in coverage_tables:
        actual_ids = set(frame["event_id"].astype(str))
        if actual_ids != expected_ids:
            raise ValueError(
                f"{name} event_id coverage does not match events.csv: "
                f"missing={sorted(expected_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - expected_ids)}"
            )
    _require_unique(orders, ["event_id", "symbol"], ORDER_FILE)
    _require_unique(planning, ["event_id", "date", "symbol"], PLANNING_FILE)
    _require_unique(
        covariance,
        ["event_id", "date", "factor_left", "factor_right"],
        FACTOR_COVARIANCE_FILE,
    )
    _require_unique(realized, ["event_id", "date", "symbol"], REALIZED_FILE)
    if scenarios is not None:
        _require_unique(
            scenarios,
            ["event_id", "scenario", "date", "symbol"],
            SCENARIO_FILE,
        )


def _build_event(
    *,
    event_id: str,
    role: str,
    as_of: pd.Timestamp,
    cutoff: pd.Timestamp,
    realized_available_at: pd.Timestamp,
    orders: pd.DataFrame,
    planning: pd.DataFrame,
    covariance: pd.DataFrame,
    realized: pd.DataFrame,
    scenarios: pd.DataFrame | None,
    source_hashes: Mapping[str, str],
) -> tuple[PointInTimeRebalanceEvent, pd.DataFrame, Mapping[float, np.ndarray]]:
    if cutoff > as_of:
        raise ValueError(f"{event_id}: information_cutoff must be on or before as_of")
    orders = orders.copy()
    planning = planning.copy()
    covariance = covariance.copy()
    realized = realized.copy()
    orders["symbol"] = orders["symbol"].astype(str)
    planning["symbol"] = planning["symbol"].astype(str)
    realized["symbol"] = realized["symbol"].astype(str)
    orders["target_shares"] = _numeric_values(
        orders["target_shares"],
        f"{event_id}.orders.target_shares",
    )
    classification_columns = ["country", "sector", "industry", "urgency"]
    if orders[classification_columns].isna().any().any() or bool(
        (orders[classification_columns].astype(str).apply(lambda column: column.str.strip()) == "")
        .any()
        .any()
    ):
        raise ValueError(f"{event_id}: order classifications must be non-empty")
    planning["date"] = _date_series(planning["date"], f"{event_id}.planning.date")
    covariance["date"] = _date_series(
        covariance["date"],
        f"{event_id}.factor_covariance.date",
    )
    realized["date"] = _date_series(realized["date"], f"{event_id}.realized.date")
    covariance["factor_left"] = covariance["factor_left"].astype(str)
    covariance["factor_right"] = covariance["factor_right"].astype(str)
    _require_unique(planning, ["date", "symbol"], f"{event_id}.{PLANNING_FILE}")
    _require_unique(realized, ["date", "symbol"], f"{event_id}.{REALIZED_FILE}")
    _require_unique(
        covariance,
        ["date", "factor_left", "factor_right"],
        f"{event_id}.{FACTOR_COVARIANCE_FILE}",
    )
    dates = pd.DatetimeIndex(planning["date"].drop_duplicates()).sort_values()
    symbols = orders["symbol"].tolist()
    if not symbols or not len(dates):
        raise ValueError(f"{event_id}: orders and planning dates must be non-empty")
    if as_of.normalize() > dates[0]:
        raise ValueError(f"{event_id}: as_of must be on or before first planner date")
    _validate_pretrade_availability(orders, cutoff, event_id, ORDER_FILE)
    _validate_pretrade_availability(planning, cutoff, event_id, PLANNING_FILE)
    _validate_pretrade_availability(
        covariance,
        cutoff,
        event_id,
        FACTOR_COVARIANCE_FILE,
    )
    if scenarios is not None:
        scenarios = scenarios.copy()
        scenarios["symbol"] = scenarios["symbol"].astype(str)
        scenarios["date"] = _date_series(
            scenarios["date"],
            f"{event_id}.scenarios.date",
        )
        _require_unique(
            scenarios,
            ["scenario", "date", "symbol"],
            f"{event_id}.{SCENARIO_FILE}",
        )
        _validate_pretrade_availability(
            scenarios,
            cutoff,
            event_id,
            SCENARIO_FILE,
        )
    _validate_realized_availability(
        realized,
        realized_available_at,
        event_id,
    )
    if realized_available_at <= dates[-1]:
        raise ValueError(
            f"{event_id}: realized_available_at must be after final planner date"
        )

    _require_complete_grid(planning, dates, symbols, event_id, PLANNING_FILE)
    _require_complete_grid(realized, dates, symbols, event_id, REALIZED_FILE)
    factor_columns = [
        column for column in planning.columns if str(column).startswith(FACTOR_COLUMN_PREFIX)
    ]
    factor_names = [str(column)[len(FACTOR_COLUMN_PREFIX) :] for column in factor_columns]
    if any(not name for name in factor_names) or len(set(factor_names)) != len(factor_names):
        raise ValueError(f"{event_id}: factor column names must be unique and non-empty")
    covariance_matrices = _factor_covariance_matrices(
        covariance,
        dates,
        factor_names,
        event_id,
    )
    scenario_values, scenario_weights = _scenario_arrays(
        scenarios,
        dates,
        symbols,
        event_id,
    )

    planning = planning.set_index(["date", "symbol"]).sort_index()
    realized = realized.set_index(["date", "symbol"]).sort_index()
    desired_index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=["date", "symbol"],
    )
    planning = planning.reindex(desired_index)
    realized = realized.reindex(desired_index)
    _validate_numeric_fields(planning, realized, factor_columns, event_id)
    market_panel = planning[
        ["price", "adv_shares", "is_open", "base_participation"]
    ].copy()
    market_panel["is_open"] = _boolean_values(
        market_panel["is_open"],
        f"{event_id}.planning.is_open",
    )
    event_days = planning["event_days"].unstack("symbol").reindex(
        index=dates,
        columns=symbols,
    )
    exposure = planning[factor_columns].copy()
    exposure.columns = factor_names
    specific_variance = planning["specific_variance"].unstack("symbol").reindex(
        index=dates,
        columns=symbols,
    )
    normalized_orders = orders.drop(columns=["event_id", "available_at"]).set_index("symbol")
    # Preserve every point-in-time order descriptor for downstream conditional
    # alpha calibration. The four required balance fields are validated above;
    # optional fields such as rebalance_type, prediction_confidence, and
    # crowding remain available without becoming planner requirements.
    classifications = normalized_orders.drop(columns=["target_shares"]).copy()
    ctx = assemble_context(
        orders=normalized_orders,
        dates=dates,
        market_panel=market_panel,
        event_days=event_days,
        factor_risk_data=FactorRiskData(
            factor_exposure=exposure,
            factor_covariance=covariance_matrices,
            specific_variance=specific_variance,
        ),
        metadata={
            "historical_bundle_role": role,
            "historical_event_id": event_id,
            "information_cutoff": cutoff.isoformat(),
            "source_hashes": dict(source_hashes),
        },
        expected_return=_field_matrix(planning, "expected_return", dates, symbols),
        expected_return_uncertainty=_field_matrix(
            planning,
            "expected_return_uncertainty",
            dates,
            symbols,
        ),
        impact_bps_at_10pct_adv=_field_matrix(
            planning,
            "impact_bps_at_10pct_adv",
            dates,
            symbols,
        ),
        linear_cost_bps=_field_matrix(
            planning,
            "linear_cost_bps",
            dates,
            symbols,
        ),
        return_residual_scenarios=scenario_values,
        return_scenario_weights=scenario_weights,
    )
    event = PointInTimeRebalanceEvent(
        event_id=event_id,
        as_of=as_of,
        information_cutoff=cutoff,
        ctx=ctx,
        realized_returns=_field_matrix(
            realized,
            "realized_return",
            dates,
            symbols,
        ),
        realized_impact_bps_at_10pct_adv=_field_matrix(
            realized,
            "realized_impact_bps_at_10pct_adv",
            dates,
            symbols,
        ),
        realized_linear_cost_bps=_field_matrix(
            realized,
            "realized_linear_cost_bps",
            dates,
            symbols,
        ),
        realized_available_at=realized_available_at,
        realized_adv_shares=_field_matrix(
            realized,
            "realized_adv_shares",
            dates,
            symbols,
        ),
    )
    validate_point_in_time_event(event)
    return event, classifications, {
        0.10: _field_matrix(planning, "forecast_adv_p10_shares", dates, symbols),
        0.25: _field_matrix(planning, "forecast_adv_p25_shares", dates, symbols),
        0.50: _field_matrix(planning, "forecast_adv_p50_shares", dates, symbols),
    }


def _scenario_arrays(
    scenarios: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    event_id: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if scenarios is None:
        return None, None
    labels = scenarios["scenario"].drop_duplicates().tolist()
    if len(labels) < 2:
        raise ValueError(f"{event_id}: scenarios.csv requires at least two scenarios")
    values = []
    weights = []
    for label in labels:
        rows = scenarios.loc[scenarios["scenario"].eq(label)].copy()
        _require_complete_grid(rows, dates, symbols, event_id, f"scenario={label}")
        rows = rows.set_index(["date", "symbol"]).reindex(
            pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        )
        residual = pd.to_numeric(rows["residual_return"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(residual)):
            raise ValueError(f"{event_id}: scenario {label!r} residual_return must be finite")
        raw_weights = pd.to_numeric(rows["scenario_weight"], errors="coerce")
        unique_weights = raw_weights.drop_duplicates()
        if len(unique_weights) != 1 or not np.isfinite(unique_weights.iloc[0]):
            raise ValueError(
                f"{event_id}: scenario_weight must be one finite value per scenario"
            )
        values.append(residual.reshape(len(dates), len(symbols)))
        weights.append(float(unique_weights.iloc[0]))
    weight_array = np.asarray(weights, dtype=float)
    if np.any(weight_array < 0.0) or np.sum(weight_array) <= 0.0:
        raise ValueError(f"{event_id}: scenario weights must be non-negative and nonzero")
    return np.asarray(values, dtype=float), weight_array / np.sum(weight_array)


def _factor_covariance_matrices(
    covariance: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_names: list[str],
    event_id: str,
) -> np.ndarray:
    left = set(covariance["factor_left"].astype(str))
    right = set(covariance["factor_right"].astype(str))
    expected = set(factor_names)
    if left != expected or right != expected:
        raise ValueError(
            f"{event_id}: factor covariance names do not match planning factors"
        )
    matrices = []
    for date in dates:
        rows = covariance.loc[covariance["date"].eq(date)].copy()
        expected_count = len(factor_names) ** 2
        if len(rows) != expected_count:
            raise ValueError(
                f"{event_id}: factor covariance grid is incomplete on {date.date()}"
            )
        matrix = (
            rows.pivot(index="factor_left", columns="factor_right", values="covariance")
            .reindex(index=factor_names, columns=factor_names)
            .to_numpy(float)
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{event_id}: factor covariance must be finite")
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-9):
            raise ValueError(f"{event_id}: factor covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12:
            raise ValueError(f"{event_id}: factor covariance must be positive semidefinite")
        matrices.append(matrix)
    if set(covariance["date"]) != set(dates):
        raise ValueError(f"{event_id}: factor covariance date coverage is incomplete")
    return np.asarray(matrices, dtype=float)


def _validate_numeric_fields(
    planning: pd.DataFrame,
    realized: pd.DataFrame,
    factor_columns: list[str],
    event_id: str,
) -> None:
    planning_positive = (
        "price",
        "adv_shares",
        "forecast_adv_p10_shares",
        "forecast_adv_p25_shares",
        "forecast_adv_p50_shares",
    )
    planning_nonnegative = (
        "base_participation",
        "event_days",
        "specific_variance",
        "expected_return_uncertainty",
        "impact_bps_at_10pct_adv",
        "linear_cost_bps",
    )
    for field in planning_positive:
        values = _numeric_values(planning[field], f"{event_id}.planning.{field}")
        if np.any(values <= 0.0):
            raise ValueError(f"{event_id}: planning {field} must be positive")
        planning[field] = values
    for field in planning_nonnegative:
        values = _numeric_values(planning[field], f"{event_id}.planning.{field}")
        if np.any(values < 0.0):
            raise ValueError(f"{event_id}: planning {field} must be non-negative")
        planning[field] = values
    if np.any(planning["base_participation"].to_numpy(float) > 1.0):
        raise ValueError(f"{event_id}: base_participation must not exceed one")
    p10 = planning["forecast_adv_p10_shares"].to_numpy(float)
    p25 = planning["forecast_adv_p25_shares"].to_numpy(float)
    p50 = planning["forecast_adv_p50_shares"].to_numpy(float)
    if np.any(p10 > p25) or np.any(p25 > p50):
        raise ValueError(
            f"{event_id}: liquidity forecasts must satisfy P10 <= P25 <= P50"
        )
    planning["expected_return"] = _numeric_values(
        planning["expected_return"],
        f"{event_id}.planning.expected_return",
    )
    for field in factor_columns:
        planning[field] = _numeric_values(
            planning[field],
            f"{event_id}.planning.{field}",
        )
    for field in (
        "realized_return",
        "realized_adv_shares",
        "realized_impact_bps_at_10pct_adv",
        "realized_linear_cost_bps",
    ):
        realized[field] = _numeric_values(
            realized[field],
            f"{event_id}.realized.{field}",
        )
    if np.any(realized["realized_adv_shares"].to_numpy(float) <= 0.0):
        raise ValueError(f"{event_id}: realized_adv_shares must be positive")
    for field in (
        "realized_impact_bps_at_10pct_adv",
        "realized_linear_cost_bps",
    ):
        if np.any(realized[field].to_numpy(float) < 0.0):
            raise ValueError(f"{event_id}: {field} must be non-negative")


def _validate_pretrade_availability(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    event_id: str,
    name: str,
) -> None:
    available = _timestamp_series(frame["available_at"], f"{event_id}.{name}.available_at")
    if bool((available > cutoff).any()):
        leaked = available.loc[available > cutoff].min()
        raise ValueError(
            f"{event_id}: {name} contains future information available at {leaked} "
            f"after cutoff {cutoff}"
        )
    frame["available_at"] = available


def _validate_realized_availability(
    realized: pd.DataFrame,
    realized_available_at: pd.Timestamp,
    event_id: str,
) -> None:
    available = _timestamp_series(
        realized["available_at"],
        f"{event_id}.realized.available_at",
    )
    dates = _date_series(realized["date"], f"{event_id}.realized.date")
    if bool((available <= dates).any()):
        raise ValueError(f"{event_id}: realized rows must be available after their date")
    if bool((available > realized_available_at).any()):
        raise ValueError(
            f"{event_id}: realized_available_at precedes one or more realized rows"
        )
    realized["available_at"] = available


def _require_complete_grid(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    event_id: str,
    name: str,
) -> None:
    actual = set(zip(frame["date"], frame["symbol"].astype(str)))
    expected = set((date, symbol) for date in dates for symbol in symbols)
    if actual != expected:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise ValueError(
            f"{event_id}: {name} date-symbol grid is incomplete; "
            f"missing={missing}, extra={extra}"
        )


def _field_matrix(
    frame: pd.DataFrame,
    field: str,
    dates: pd.DatetimeIndex,
    symbols: list[str],
) -> np.ndarray:
    if not isinstance(frame.index, pd.MultiIndex):
        frame = frame.set_index(["date", "symbol"])
    return (
        frame[field]
        .unstack("symbol")
        .reindex(index=dates, columns=symbols)
        .to_numpy(float)
    )


def _boolean_values(values: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only true/false or 1/0")
    return normalized.map(mapping).to_numpy(bool)


def _numeric_values(values: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite numeric values")
    return numeric


def _timestamp_series(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{name} must contain valid timestamps")
    return parsed.dt.tz_convert(None)


def _date_series(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{name} must contain valid dates")
    return parsed.dt.tz_convert(None).dt.normalize()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, keys].iloc[0].to_dict()
        raise ValueError(f"{name} contains duplicate keys {keys}: {example}")
