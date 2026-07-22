"""Leakage-safe conditional calibration of rebalance holding alpha."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import NormalDist
import numpy as np
import pandas as pd

from .historical import HistoricalReplayBundle
from .walkforward import PointInTimeRebalanceEvent, validate_point_in_time_event


RIDGE_MULTIPLIERS = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_MIN_TRAINING_EVENTS = 4
BASE_NUMERIC_FEATURES = (
    "raw_directional_return",
    "progress",
    "progress_squared",
    "inverse_event_days",
    "event_days_missing",
    "capacity_pressure",
    "log_adv_shares",
    "raw_uncertainty",
    "prediction_confidence",
    "prediction_confidence_missing",
    "crowding",
    "crowding_missing",
)
CATEGORICAL_FEATURES = (
    "side",
    "country",
    "sector",
    "industry",
    "urgency",
    "rebalance_type",
)
REQUIRED_CLASSIFICATIONS = frozenset({"country", "sector", "industry", "urgency"})


@dataclass(frozen=True)
class _FeatureEncoder:
    numeric_columns: tuple[str, ...]
    numeric_means: np.ndarray
    numeric_scales: np.ndarray
    categorical_levels: tuple[tuple[str, tuple[str, ...]], ...]
    feature_names: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = [np.ones((len(frame), 1), dtype=float)]
        numeric = np.column_stack(
            [
                pd.to_numeric(
                    frame.get(column, pd.Series(np.nan, index=frame.index)),
                    errors="coerce",
                ).to_numpy(float)
                for column in self.numeric_columns
            ]
        )
        numeric = np.where(np.isfinite(numeric), numeric, self.numeric_means)
        columns.append((numeric - self.numeric_means) / self.numeric_scales)
        for column, levels in self.categorical_levels:
            values = _categorical_series(frame, column).to_numpy(str)
            columns.append(
                np.column_stack([values == level for level in levels]).astype(float)
            )
        return np.column_stack(columns)


@dataclass(frozen=True)
class ConditionalAlphaDecayModel:
    """One ridge model fitted using events available before a planning cutoff."""

    encoder: _FeatureEncoder
    coefficients: np.ndarray
    precision_inverse: np.ndarray
    ridge_multiplier: float
    ridge_penalty: float
    cross_validated_rmse: float
    residual_standard_error: float
    training_event_ids: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        design = self.encoder.transform(frame)
        mean = design @ self.coefficients
        leverage = np.einsum(
            "ij,jk,ik->i",
            design,
            self.precision_inverse,
            design,
        )
        uncertainty = self.residual_standard_error * np.sqrt(
            1.0 + np.maximum(leverage, 0.0)
        )
        return np.asarray(mean, dtype=float), np.asarray(uncertainty, dtype=float)


@dataclass(frozen=True)
class AlphaDecayWalkForward:
    """Calibrated point-in-time events plus a complete audit trail."""

    events: tuple[PointInTimeRebalanceEvent, ...]
    audit: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    summary: pd.DataFrame


def calibrate_alpha_decay_walk_forward(
    bundle: HistoricalReplayBundle,
    *,
    min_training_events: int = DEFAULT_MIN_TRAINING_EVENTS,
    ridge_multipliers: tuple[float, ...] = RIDGE_MULTIPLIERS,
) -> AlphaDecayWalkForward:
    """Calibrate every event using only earlier outcomes available at its cutoff.

    The current event's realized returns are attached to the returned prediction
    audit only after its forecast has been produced. They never enter feature
    encoding, cross-validation, coefficient fitting, or uncertainty estimation.
    """

    if min_training_events < 2:
        raise ValueError("min_training_events must be at least two")
    candidates = tuple(float(value) for value in ridge_multipliers)
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("ridge_multipliers must contain positive finite values")
    _validate_bundle_order(bundle)

    calibrated_events: list[PointInTimeRebalanceEvent] = []
    audit_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    prior_training_frames: list[tuple[PointInTimeRebalanceEvent, pd.DataFrame]] = []

    for event in bundle.events:
        event_id = str(event.event_id)
        current_cutoff = pd.Timestamp(event.information_cutoff)
        classifications = _classifications_for(bundle, event)
        current_features = _event_feature_frame(
            event,
            classifications,
            include_realized=False,
        )
        eligible = [
            (prior, frame)
            for prior, frame in prior_training_frames
            if pd.Timestamp(prior.realized_available_at) <= current_cutoff
        ]
        training_ids = tuple(str(prior.event_id) for prior, _ in eligible)
        status = "raw_fallback"
        selected_multiplier = np.nan
        selected_penalty = np.nan
        cv_rmse = np.nan
        residual_se = np.nan
        candidate_scores: dict[float, float] = {}

        if len(eligible) >= min_training_events:
            history = pd.concat([frame for _, frame in eligible], ignore_index=True)
            candidate_scores = {
                multiplier: _leave_one_event_out_rmse(history, multiplier)
                for multiplier in candidates
            }
            selected_multiplier = min(
                candidates,
                key=lambda multiplier: (candidate_scores[multiplier], multiplier),
            )
            cv_rmse = candidate_scores[selected_multiplier]
            model = _fit_ridge_model(
                history,
                ridge_multiplier=selected_multiplier,
                cross_validated_rmse=cv_rmse,
                residual_standard_error=max(cv_rmse, np.finfo(float).eps),
                training_event_ids=training_ids,
            )
            modeled_mean, modeled_uncertainty = model.predict(current_features)
            selected_penalty = model.ridge_penalty
            residual_se = model.residual_standard_error
            active = current_features["target_sign"].to_numpy(float) != 0.0
            directional_mean = current_features[
                "raw_directional_return"
            ].to_numpy(float).copy()
            directional_mean[active] = modeled_mean[active]
            predictive_uncertainty = (
                np.zeros(len(current_features), dtype=float)
                if event.ctx.expected_return_uncertainty is None
                else np.asarray(
                    event.ctx.expected_return_uncertainty,
                    dtype=float,
                ).reshape(-1).copy()
            )
            predictive_uncertainty[active] = modeled_uncertainty[active]
            calibrated_return_flat = np.asarray(
                event.ctx.expected_return,
                dtype=float,
            ).reshape(-1).copy()
            calibrated_return_flat[active] = (
                current_features.loc[active, "target_sign"].to_numpy(float)
                * directional_mean[active]
            )
            calibrated_return = calibrated_return_flat.reshape(
                event.ctx.expected_return.shape
            )
            calibrated_uncertainty_flat = predictive_uncertainty.copy()
            calibrated_uncertainty_flat[active] = predictive_uncertainty[active]
            calibrated_uncertainty = calibrated_uncertainty_flat.reshape(
                event.ctx.expected_return.shape
            )
            calibrated_ctx = replace(
                event.ctx,
                expected_return=calibrated_return,
                expected_return_uncertainty=calibrated_uncertainty,
                metadata={
                    **event.ctx.metadata,
                    "alpha_decay_status": "calibrated",
                    "alpha_decay_training_event_ids": training_ids,
                    "alpha_decay_ridge_multiplier": selected_multiplier,
                    "alpha_decay_cv_rmse": cv_rmse,
                },
            )
            calibrated_event = replace(event, ctx=calibrated_ctx)
            status = "calibrated"
            for feature_name, coefficient in zip(
                model.encoder.feature_names,
                model.coefficients,
            ):
                coefficient_rows.append(
                    {
                        "event_id": event_id,
                        "information_cutoff": current_cutoff,
                        "training_event_ids": "|".join(training_ids),
                        "ridge_multiplier": selected_multiplier,
                        "ridge_penalty": selected_penalty,
                        "feature": feature_name,
                        "coefficient": float(coefficient),
                    }
                )
        else:
            if event.ctx.expected_return is None:
                raise ValueError(f"{event_id}: expected_return is required")
            directional_mean = current_features["raw_directional_return"].to_numpy(float)
            if event.ctx.expected_return_uncertainty is None:
                predictive_uncertainty = np.zeros(len(current_features), dtype=float)
            else:
                predictive_uncertainty = np.asarray(
                    event.ctx.expected_return_uncertainty,
                    dtype=float,
                ).reshape(-1)
            calibrated_event = replace(
                event,
                ctx=replace(
                    event.ctx,
                    metadata={
                        **event.ctx.metadata,
                        "alpha_decay_status": "raw_fallback",
                        "alpha_decay_training_event_ids": training_ids,
                    },
                ),
            )

        calibrated_events.append(calibrated_event)
        prediction = current_features[
            [
                "event_id",
                "as_of",
                "information_cutoff",
                "date",
                "symbol",
                "target_sign",
                "progress",
                "side",
                "country",
                "sector",
                "industry",
                "urgency",
                "raw_directional_return",
            ]
        ].copy()
        prediction["status"] = status
        prediction["training_event_count"] = len(training_ids)
        prediction["training_event_ids"] = "|".join(training_ids)
        prediction["ridge_multiplier"] = selected_multiplier
        prediction["ridge_penalty"] = selected_penalty
        prediction["calibrated_directional_return"] = directional_mean
        prediction["predictive_uncertainty"] = predictive_uncertainty
        prediction["realized_directional_return"] = (
            current_features["target_sign"].to_numpy(float)
            * np.asarray(event.realized_returns, dtype=float).reshape(-1)
        )
        prediction_frames.append(prediction)
        audit_row: dict[str, object] = {
            "event_id": event_id,
            "as_of": pd.Timestamp(event.as_of),
            "information_cutoff": current_cutoff,
            "status": status,
            "eligible_training_event_count": len(training_ids),
            "eligible_training_event_ids": "|".join(training_ids),
            "latest_training_realized_available_at": (
                max(pd.Timestamp(prior.realized_available_at) for prior, _ in eligible)
                if eligible
                else pd.NaT
            ),
            "ridge_multiplier": selected_multiplier,
            "ridge_penalty": selected_penalty,
            "cross_validated_rmse": cv_rmse,
            "residual_standard_error": residual_se,
        }
        for multiplier in candidates:
            audit_row[f"cv_rmse_multiplier_{multiplier:g}"] = candidate_scores.get(
                multiplier,
                np.nan,
            )
        audit_rows.append(audit_row)

        training_frame = _event_feature_frame(
            event,
            classifications,
            include_realized=True,
        )
        training_frame = training_frame.loc[
            training_frame["target_sign"].ne(0.0)
        ].reset_index(drop=True)
        if not training_frame.empty:
            prior_training_frames.append((event, training_frame))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_alpha_decay_predictions(predictions)
    return AlphaDecayWalkForward(
        events=tuple(calibrated_events),
        audit=pd.DataFrame(audit_rows),
        predictions=predictions,
        coefficients=pd.DataFrame(coefficient_rows),
        summary=summary,
    )


def summarize_alpha_decay_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-warmup calibration accuracy without selecting on P&L."""

    calibrated = predictions.loc[
        predictions["status"].eq("calibrated")
        & predictions["target_sign"].ne(0.0)
    ].copy()
    columns = {
        "observation_count": 0,
        "event_count": 0,
        "raw_rmse": np.nan,
        "calibrated_rmse": np.nan,
        "rmse_improvement_fraction": np.nan,
        "raw_sign_accuracy": np.nan,
        "calibrated_sign_accuracy": np.nan,
        "predictive_interval_80_coverage": np.nan,
    }
    if calibrated.empty:
        return pd.DataFrame([columns])
    realized = calibrated["realized_directional_return"].to_numpy(float)
    raw = calibrated["raw_directional_return"].to_numpy(float)
    fitted = calibrated["calibrated_directional_return"].to_numpy(float)
    uncertainty = calibrated["predictive_uncertainty"].to_numpy(float)
    raw_rmse = _equal_event_rmse(calibrated, "raw_directional_return")
    calibrated_rmse = _equal_event_rmse(
        calibrated,
        "calibrated_directional_return",
    )
    interval_quantile = NormalDist().inv_cdf(0.90)
    lower = fitted - interval_quantile * uncertainty
    upper = fitted + interval_quantile * uncertainty
    columns.update(
        {
            "observation_count": len(calibrated),
            "event_count": calibrated["event_id"].nunique(),
            "raw_rmse": raw_rmse,
            "calibrated_rmse": calibrated_rmse,
            "rmse_improvement_fraction": (
                (raw_rmse - calibrated_rmse) / raw_rmse if raw_rmse > 0 else 0.0
            ),
            "raw_sign_accuracy": float(np.mean(np.sign(raw) == np.sign(realized))),
            "calibrated_sign_accuracy": float(
                np.mean(np.sign(fitted) == np.sign(realized))
            ),
            "predictive_interval_80_coverage": float(
                np.mean((realized >= lower) & (realized <= upper))
            ),
        }
    )
    return pd.DataFrame([columns])


def _validate_bundle_order(bundle: HistoricalReplayBundle) -> None:
    if not bundle.events:
        raise ValueError("historical bundle must contain at least one event")
    event_ids = [str(event.event_id) for event in bundle.events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("historical bundle event_id values must be unique")
    expected = sorted(
        bundle.events,
        key=lambda event: (pd.Timestamp(event.as_of), str(event.event_id)),
    )
    if event_ids != [str(event.event_id) for event in expected]:
        raise ValueError("historical bundle events must be chronological")
    for event in bundle.events:
        validate_point_in_time_event(event)


def _classifications_for(
    bundle: HistoricalReplayBundle,
    event: PointInTimeRebalanceEvent,
) -> pd.DataFrame:
    event_id = str(event.event_id)
    if event_id not in bundle.classifications:
        raise KeyError(f"{event_id}: missing classifications")
    classifications = bundle.classifications[event_id].copy()
    classifications.index = classifications.index.astype(str)
    classifications = classifications.reindex(event.ctx.symbols)
    missing = REQUIRED_CLASSIFICATIONS.difference(classifications.columns)
    if missing:
        raise ValueError(f"{event_id}: missing classifications {sorted(missing)}")
    if classifications[list(REQUIRED_CLASSIFICATIONS)].isna().any().any():
        raise ValueError(f"{event_id}: classifications must cover every symbol")
    return classifications


def _event_feature_frame(
    event: PointInTimeRebalanceEvent,
    classifications: pd.DataFrame,
    *,
    include_realized: bool,
) -> pd.DataFrame:
    ctx = event.ctx
    if ctx.expected_return is None:
        raise ValueError(f"{event.event_id}: expected_return is required")
    expected = np.asarray(ctx.expected_return, dtype=float)
    expected_shape = (len(ctx.dates), len(ctx.symbols))
    if expected.shape != expected_shape or not np.all(np.isfinite(expected)):
        raise ValueError(
            f"{event.event_id}: expected_return must be finite with shape {expected_shape}"
        )
    uncertainty = (
        np.zeros(expected_shape, dtype=float)
        if ctx.expected_return_uncertainty is None
        else np.asarray(ctx.expected_return_uncertainty, dtype=float)
    )
    if (
        uncertainty.shape != expected_shape
        or not np.all(np.isfinite(uncertainty))
        or np.any(uncertainty < 0)
    ):
        raise ValueError(
            f"{event.event_id}: expected_return_uncertainty must be finite, "
            f"non-negative, and have shape {expected_shape}"
        )

    target = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    target_sign = np.sign(target)
    days = ctx.event_days.reindex(index=ctx.dates, columns=ctx.symbols).to_numpy(float)
    finite_days = np.isfinite(days)
    inverse_days = np.zeros_like(days, dtype=float)
    inverse_days[finite_days] = 1.0 / (1.0 + np.maximum(days[finite_days], 0.0))
    horizon_capacity = np.sum(
        np.asarray(ctx.base_participation, dtype=float)
        * np.asarray(ctx.adv_shares, dtype=float)
        * np.asarray(ctx.is_open, dtype=float),
        axis=0,
    )
    capacity_pressure = np.abs(target) / np.maximum(horizon_capacity, 1e-12)
    n_dates = len(ctx.dates)
    n_symbols = len(ctx.symbols)
    progress = (
        np.zeros(n_dates, dtype=float)
        if n_dates == 1
        else np.arange(n_dates, dtype=float) / float(n_dates - 1)
    )
    frame = pd.DataFrame(
        {
            "event_id": str(event.event_id),
            "as_of": pd.Timestamp(event.as_of),
            "information_cutoff": pd.Timestamp(event.information_cutoff),
            "date": np.repeat(ctx.dates.to_numpy(), n_symbols),
            "symbol": np.tile(np.asarray(ctx.symbols, dtype=object), n_dates),
            "target_sign": np.tile(target_sign, n_dates),
            "raw_directional_return": (expected * target_sign[None, :]).reshape(-1),
            "progress": np.repeat(progress, n_symbols),
            "progress_squared": np.repeat(np.square(progress), n_symbols),
            "inverse_event_days": inverse_days.reshape(-1),
            "event_days_missing": (~finite_days).astype(float).reshape(-1),
            "capacity_pressure": np.tile(capacity_pressure, n_dates),
            "log_adv_shares": np.log1p(
                np.maximum(np.asarray(ctx.adv_shares, dtype=float), 0.0)
            ).reshape(-1),
            "raw_uncertainty": uncertainty.reshape(-1),
            "side": np.tile(
                np.where(target_sign > 0, "buy", np.where(target_sign < 0, "sell", "flat")),
                n_dates,
            ),
        }
    )
    for column in ("country", "sector", "industry", "urgency", "rebalance_type"):
        values = (
            classifications[column]
            if column in classifications
            else pd.Series("__MISSING__", index=classifications.index)
        )
        frame[column] = np.tile(
            values.fillna("__MISSING__").astype(str).to_numpy(),
            n_dates,
        )
    for column in ("prediction_confidence", "crowding"):
        if column in classifications:
            values = pd.to_numeric(classifications[column], errors="coerce").to_numpy(float)
        else:
            values = np.full(n_symbols, np.nan, dtype=float)
        frame[column] = np.tile(values, n_dates)
        frame[f"{column}_missing"] = np.tile((~np.isfinite(values)).astype(float), n_dates)
    if include_realized:
        frame["realized_directional_return"] = (
            target_sign[None, :]
            * np.asarray(event.realized_returns, dtype=float)
        ).reshape(-1)
    return frame.reset_index(drop=True)


def _fit_encoder(frame: pd.DataFrame) -> _FeatureEncoder:
    numeric_columns = tuple(BASE_NUMERIC_FEATURES)
    raw_numeric = np.column_stack(
        [pd.to_numeric(frame[column], errors="coerce").to_numpy(float) for column in numeric_columns]
    )
    numeric_means = np.asarray(
        [
            float(np.nanmean(column)) if np.isfinite(column).any() else 0.0
            for column in raw_numeric.T
        ]
    )
    filled = np.where(np.isfinite(raw_numeric), raw_numeric, numeric_means)
    numeric_scales = np.std(filled, axis=0)
    numeric_scales = np.where(numeric_scales > 1e-12, numeric_scales, 1.0)
    categorical_levels = tuple(
        (
            column,
            tuple(sorted(_categorical_series(frame, column).unique().tolist())),
        )
        for column in CATEGORICAL_FEATURES
    )
    names = ["intercept", *numeric_columns]
    for column, levels in categorical_levels:
        names.extend(f"{column}={level}" for level in levels)
    return _FeatureEncoder(
        numeric_columns=numeric_columns,
        numeric_means=numeric_means,
        numeric_scales=numeric_scales,
        categorical_levels=categorical_levels,
        feature_names=tuple(names),
    )


def _fit_ridge_model(
    frame: pd.DataFrame,
    *,
    ridge_multiplier: float,
    cross_validated_rmse: float,
    residual_standard_error: float,
    training_event_ids: tuple[str, ...],
) -> ConditionalAlphaDecayModel:
    encoder = _fit_encoder(frame)
    design = encoder.transform(frame)
    response = frame["realized_directional_return"].to_numpy(float)
    counts = frame.groupby("event_id")["event_id"].transform("size").to_numpy(float)
    weights = 1.0 / counts
    gram = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * response)
    penalized_dimension = max(design.shape[1] - 1, 1)
    design_scale = max(
        float(np.trace(gram[1:, 1:])) / penalized_dimension,
        np.finfo(float).eps,
    )
    penalty = float(ridge_multiplier) * design_scale
    ridge = np.eye(design.shape[1], dtype=float)
    ridge[0, 0] = 0.0
    precision = gram + penalty * ridge
    coefficients = np.linalg.solve(precision, rhs)
    return ConditionalAlphaDecayModel(
        encoder=encoder,
        coefficients=coefficients,
        precision_inverse=np.linalg.pinv(precision),
        ridge_multiplier=float(ridge_multiplier),
        ridge_penalty=penalty,
        cross_validated_rmse=float(cross_validated_rmse),
        residual_standard_error=float(residual_standard_error),
        training_event_ids=training_event_ids,
    )


def _leave_one_event_out_rmse(frame: pd.DataFrame, ridge_multiplier: float) -> float:
    event_ids = tuple(frame["event_id"].drop_duplicates().astype(str))
    event_mse: list[float] = []
    for held_out in event_ids:
        train = frame.loc[frame["event_id"].ne(held_out)].reset_index(drop=True)
        test = frame.loc[frame["event_id"].eq(held_out)].reset_index(drop=True)
        model = _fit_ridge_model(
            train,
            ridge_multiplier=ridge_multiplier,
            cross_validated_rmse=np.nan,
            residual_standard_error=1.0,
            training_event_ids=tuple(train["event_id"].drop_duplicates().astype(str)),
        )
        predicted, _ = model.predict(test)
        residual = test["realized_directional_return"].to_numpy(float) - predicted
        event_mse.append(float(np.mean(np.square(residual))))
    return float(np.sqrt(np.mean(event_mse)))


def _categorical_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("__MISSING__", index=frame.index, dtype=str)
    return frame[column].fillna("__MISSING__").astype(str)


def _equal_event_rmse(frame: pd.DataFrame, prediction_column: str) -> float:
    event_mse = (
        frame.assign(
            squared_error=np.square(
                frame["realized_directional_return"].to_numpy(float)
                - frame[prediction_column].to_numpy(float)
            )
        )
        .groupby("event_id", sort=False)["squared_error"]
        .mean()
    )
    return float(np.sqrt(event_mse.mean()))
