"""Chronological investment calibration for High/Medium/Low policy vectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import DEFAULT_RISK_PREFERENCES, RiskAversion


PROFILE_ORDER = (RiskAversion.HIGH, RiskAversion.MEDIUM, RiskAversion.LOW)
DEFAULT_CONTEXTUAL_RIDGE_MULTIPLIERS = (0.01, 0.10, 1.0, 10.0, 100.0)
PROFILE_CONFIDENCE: Mapping[RiskAversion, float] = {
    RiskAversion.HIGH: 0.975,
    RiskAversion.MEDIUM: 0.75,
    RiskAversion.LOW: 0.50,
}
FALLBACK_AGGRESSIVENESS: Mapping[RiskAversion, float] = {
    RiskAversion.HIGH: 0.05,
    RiskAversion.MEDIUM: 0.50,
    RiskAversion.LOW: 1.00,
}
POLICY_COEFFICIENT_COLUMNS = (
    "policy_aggressiveness",
    "risk_frontier_fraction",
    "liquidity_quantile",
    "liquidity_shape_fraction",
    "alpha_confidence",
    "factor_stress_fraction",
)
EVENT_COLUMNS = (
    "event_id",
    "as_of",
    "information_cutoff",
    "realized_available_at",
)
TRIAL_COLUMNS = (
    "event_id",
    "policy_id",
    "net_pnl_bps",
    "within_event_max_drawdown_bps",
    "hard_pass",
    "behavior_pass",
)


@dataclass(frozen=True)
class AutomaticRiskProfileCalibration:
    """Walk-forward policy selections and their complete evidence trail."""

    selections: pd.DataFrame
    policy_evaluations: pd.DataFrame
    summary: pd.DataFrame
    policies: pd.DataFrame


@dataclass(frozen=True)
class InvestmentPolicyCoefficients:
    """One complete optimizer policy selected as an indivisible vector."""

    policy_aggressiveness: float
    risk_frontier_fraction: float
    liquidity_quantile: float
    liquidity_shape_fraction: float
    alpha_confidence: float
    factor_stress_fraction: float
    policy_id: str = "selected_policy"

    def __post_init__(self) -> None:
        values = {
            column: float(getattr(self, column))
            for column in POLICY_COEFFICIENT_COLUMNS
        }
        if not 0.0 < values["policy_aggressiveness"] <= 1.0:
            raise ValueError("policy_aggressiveness must be in (0, 1]")
        for column in (
            "risk_frontier_fraction",
            "liquidity_shape_fraction",
            "factor_stress_fraction",
        ):
            if not 0.0 <= values[column] <= 1.0:
                raise ValueError(f"{column} must be between zero and one")
        if not 0.0 < values["liquidity_quantile"] < 1.0:
            raise ValueError("liquidity_quantile must be strictly between zero and one")
        if not 0.50 <= values["alpha_confidence"] < 1.0:
            raise ValueError(
                "alpha_confidence must be between 0.50 inclusive and 1.0 exclusive"
            )
        if not str(self.policy_id).strip():
            raise ValueError("policy_id must be non-empty")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
    ) -> InvestmentPolicyCoefficients:
        return cls(
            policy_id=str(
                values.get(
                    "policy_id",
                    values.get("selected_policy_id", "selected_policy"),
                )
            ),
            **{
                column: float(values[column])
                for column in POLICY_COEFFICIENT_COLUMNS
            },
        )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "policy_id": self.policy_id,
            **{
                column: float(getattr(self, column))
                for column in POLICY_COEFFICIENT_COLUMNS
            },
        }


def build_monotone_policy_ladder(
    aggressiveness: Sequence[float],
) -> pd.DataFrame:
    """Map one economic risk coordinate to a coherent coefficient vector.

    The coefficients move together: a more aggressive policy receives more
    frontier risk and median-like liquidity, while consuming less defensive
    liquidity shape, alpha confidence, and factor stress.
    """

    values = np.asarray(list(aggressiveness), dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or np.any(values > 1.0)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError(
            "aggressiveness must contain unique finite values in (0, 1]"
        )
    values = np.sort(values)
    return pd.DataFrame(
        {
            "policy_id": [f"policy_{round(value * 1000):04d}" for value in values],
            "policy_aggressiveness": values,
            "risk_frontier_fraction": values,
            "liquidity_quantile": 0.10 + 0.40 * values,
            "liquidity_shape_fraction": 1.0 - values,
            "alpha_confidence": 1.0 - 0.50 * values,
            "factor_stress_fraction": 1.0 - values,
        }
    )


def calibrate_risk_profiles_walk_forward(
    events: pd.DataFrame,
    trials: pd.DataFrame,
    policies: pd.DataFrame,
    *,
    min_training_events: int = 8,
    minimum_behavior_pass_rate: float = 0.95,
    materiality_bps: float = 1.0,
) -> AutomaticRiskProfileCalibration:
    """Select complete risk-profile vectors using earlier available outcomes.

    Current-event trial rows are retrieved only after the selection is frozen.
    They are returned for out-of-sample scoring but cannot affect candidate
    eligibility, confidence bounds, realized-risk budgets, or tie breaking.
    """

    if min_training_events < 2:
        raise ValueError("min_training_events must be at least two")
    if not 0.0 <= minimum_behavior_pass_rate <= 1.0:
        raise ValueError("minimum_behavior_pass_rate must be between zero and one")
    if not np.isfinite(materiality_bps) or materiality_bps < 0.0:
        raise ValueError("materiality_bps must be non-negative and finite")
    event_frame, trial_frame, policy_frame = _validated_inputs(
        events,
        trials,
        policies,
    )
    fallback = _fallback_policy_ids(policy_frame)
    selection_rows: list[dict[str, object]] = []
    evaluation_frames: list[pd.DataFrame] = []
    prior_events: list[pd.Series] = []

    for current in event_frame.itertuples(index=False):
        current_id = str(current.event_id)
        current_cutoff = pd.Timestamp(current.information_cutoff)
        eligible_ids = tuple(
            str(prior["event_id"])
            for prior in prior_events
            if pd.Timestamp(prior["realized_available_at"]) <= current_cutoff
        )
        if len(eligible_ids) < min_training_events:
            selected_by_profile = {
                profile: {
                    "selected_policy_id": fallback[profile],
                    "status": "fallback_warmup",
                    "economically_viable": False,
                    "net_pnl_lower_bound_bps": np.nan,
                    "realized_risk_bps": np.nan,
                    "realized_risk_budget_bps": np.nan,
                }
                for profile in PROFILE_ORDER
            }
        else:
            history = trial_frame.loc[
                trial_frame["event_id"].isin(eligible_ids)
            ].copy()
            statistics = _policy_statistics(history, policy_frame)
            selected_by_profile, profile_evaluations = _select_profiles(
                statistics,
                minimum_behavior_pass_rate=minimum_behavior_pass_rate,
                materiality_bps=materiality_bps,
            )
            for profile, evaluation in profile_evaluations.items():
                evaluation_frames.append(
                    evaluation.assign(
                        event_id=current_id,
                        as_of=pd.Timestamp(current.as_of),
                        information_cutoff=current_cutoff,
                        risk_aversion=profile.value,
                        training_event_ids="|".join(eligible_ids),
                    )
                )

        current_outcomes = trial_frame.loc[
            trial_frame["event_id"].eq(current_id)
        ].set_index("policy_id")
        for profile in PROFILE_ORDER:
            selected = selected_by_profile[profile]
            selected_policy_id = str(selected["selected_policy_id"])
            fallback_policy_id = fallback[profile]
            selected_outcome = current_outcomes.loc[selected_policy_id]
            fallback_outcome = current_outcomes.loc[fallback_policy_id]
            policy = policy_frame.set_index("policy_id").loc[selected_policy_id]
            selection_rows.append(
                {
                    "event_id": current_id,
                    "as_of": pd.Timestamp(current.as_of),
                    "information_cutoff": current_cutoff,
                    "risk_aversion": profile.value,
                    "status": selected["status"],
                    "economically_viable": bool(selected["economically_viable"]),
                    "training_event_count": len(eligible_ids),
                    "training_event_ids": "|".join(eligible_ids),
                    "selected_policy_id": selected_policy_id,
                    "fallback_policy_id": fallback_policy_id,
                    "profile_confidence": PROFILE_CONFIDENCE[profile],
                    "net_pnl_lower_bound_bps": selected[
                        "net_pnl_lower_bound_bps"
                    ],
                    "realized_risk_bps": selected["realized_risk_bps"],
                    "realized_risk_budget_bps": selected[
                        "realized_risk_budget_bps"
                    ],
                    **{
                        column: float(policy[column])
                        for column in POLICY_COEFFICIENT_COLUMNS
                    },
                    "selected_net_pnl_bps": float(selected_outcome["net_pnl_bps"]),
                    "fallback_net_pnl_bps": float(fallback_outcome["net_pnl_bps"]),
                    "selected_within_event_drawdown_bps": float(
                        selected_outcome["within_event_max_drawdown_bps"]
                    ),
                    "fallback_within_event_drawdown_bps": float(
                        fallback_outcome["within_event_max_drawdown_bps"]
                    ),
                    "selected_hard_pass": bool(selected_outcome["hard_pass"]),
                    "selected_behavior_pass": bool(
                        selected_outcome["behavior_pass"]
                    ),
                }
            )
        prior_events.append(pd.Series(current._asdict()))

    selections = pd.DataFrame(selection_rows)
    evaluations = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame()
    )
    return AutomaticRiskProfileCalibration(
        selections=selections,
        policy_evaluations=evaluations,
        summary=summarize_risk_profile_selections(selections),
        policies=policy_frame.copy(),
    )


def calibrate_contextual_risk_profiles_walk_forward(
    events: pd.DataFrame,
    trials: pd.DataFrame,
    policies: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    min_training_events: int = 12,
    minimum_behavior_pass_rate: float = 0.95,
    materiality_bps: float = 1.0,
    ridge_multipliers: Sequence[float] = DEFAULT_CONTEXTUAL_RIDGE_MULTIPLIERS,
) -> AutomaticRiskProfileCalibration:
    """Select complete policies from point-in-time event-conditioned P&L.

    Operational eligibility and realized-risk budgets still use eligible
    historical outcomes. Only the expected P&L ranking is conditioned on the
    current event's planning-time features. Current and future outcomes are
    attached after selection and never enter feature scaling or ridge fitting.
    """

    if min_training_events < 4:
        raise ValueError("min_training_events must be at least four")
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one feature")
    feature_names = tuple(str(column) for column in feature_columns)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_columns must be unique")
    multipliers = np.asarray(list(ridge_multipliers), dtype=float)
    if (
        multipliers.ndim != 1
        or len(multipliers) == 0
        or not np.all(np.isfinite(multipliers))
        or np.any(multipliers <= 0.0)
        or len(np.unique(multipliers)) != len(multipliers)
    ):
        raise ValueError(
            "ridge_multipliers must contain unique finite positive values"
        )
    if not 0.0 <= minimum_behavior_pass_rate <= 1.0:
        raise ValueError("minimum_behavior_pass_rate must be between zero and one")
    if not np.isfinite(materiality_bps) or materiality_bps < 0.0:
        raise ValueError("materiality_bps must be non-negative and finite")

    event_frame, trial_frame, policy_frame = _validated_inputs(
        events,
        trials,
        policies,
    )
    missing_features = set(feature_names).difference(event_frame.columns)
    if missing_features:
        raise ValueError(
            f"events is missing contextual features: {sorted(missing_features)}"
        )
    for column in feature_names:
        event_frame[column] = pd.to_numeric(event_frame[column], errors="raise")
        if not np.isfinite(event_frame[column].to_numpy(float)).all():
            raise ValueError(f"events.{column} must be finite")

    fallback = _fallback_policy_ids(policy_frame)
    event_by_id = event_frame.set_index("event_id")
    selection_rows: list[dict[str, object]] = []
    evaluation_frames: list[pd.DataFrame] = []
    prior_events: list[pd.Series] = []

    for current in event_frame.itertuples(index=False):
        current_id = str(current.event_id)
        current_cutoff = pd.Timestamp(current.information_cutoff)
        eligible_ids = tuple(
            str(prior["event_id"])
            for prior in prior_events
            if pd.Timestamp(prior["realized_available_at"]) <= current_cutoff
        )
        current_features = np.asarray(
            [float(getattr(current, column)) for column in feature_names],
            dtype=float,
        )
        if len(eligible_ids) < min_training_events:
            selected_by_profile = {
                profile: {
                    "selected_policy_id": fallback[profile],
                    "status": "fallback_contextual_warmup",
                    "economically_viable": False,
                    "net_pnl_lower_bound_bps": np.nan,
                    "realized_risk_bps": np.nan,
                    "realized_risk_budget_bps": np.nan,
                    "predicted_net_pnl_bps": np.nan,
                    "prediction_standard_error_bps": np.nan,
                    "contextual_ridge_multiplier": np.nan,
                    "contextual_cv_rmse_bps": np.nan,
                }
                for profile in PROFILE_ORDER
            }
        else:
            history = trial_frame.loc[
                trial_frame["event_id"].isin(eligible_ids)
            ].copy()
            statistics = _policy_statistics(history, policy_frame)
            feature_history = event_by_id.loc[
                list(eligible_ids),
                list(feature_names),
            ].to_numpy(float)
            predictions = _contextual_policy_predictions(
                feature_history,
                history,
                policy_frame,
                current_features,
                feature_names=feature_names,
                ridge_multipliers=multipliers,
            )
            statistics = statistics.merge(
                predictions,
                on="policy_id",
                validate="one_to_one",
            )
            selected_by_profile, profile_evaluations = (
                _select_contextual_profiles(
                    statistics,
                    minimum_behavior_pass_rate=minimum_behavior_pass_rate,
                    materiality_bps=materiality_bps,
                )
            )
            for profile, evaluation in profile_evaluations.items():
                evaluation_frames.append(
                    evaluation.assign(
                        event_id=current_id,
                        as_of=pd.Timestamp(current.as_of),
                        information_cutoff=current_cutoff,
                        risk_aversion=profile.value,
                        training_event_ids="|".join(eligible_ids),
                        **{
                            f"current_feature_{column}": value
                            for column, value in zip(
                                feature_names,
                                current_features,
                            )
                        },
                    )
                )

        current_outcomes = trial_frame.loc[
            trial_frame["event_id"].eq(current_id)
        ].set_index("policy_id")
        for profile in PROFILE_ORDER:
            selected = selected_by_profile[profile]
            selected_policy_id = str(selected["selected_policy_id"])
            fallback_policy_id = fallback[profile]
            selected_outcome = current_outcomes.loc[selected_policy_id]
            fallback_outcome = current_outcomes.loc[fallback_policy_id]
            policy = policy_frame.set_index("policy_id").loc[selected_policy_id]
            selection_rows.append(
                {
                    "event_id": current_id,
                    "as_of": pd.Timestamp(current.as_of),
                    "information_cutoff": current_cutoff,
                    "risk_aversion": profile.value,
                    "status": selected["status"],
                    "economically_viable": bool(selected["economically_viable"]),
                    "training_event_count": len(eligible_ids),
                    "training_event_ids": "|".join(eligible_ids),
                    "selected_policy_id": selected_policy_id,
                    "fallback_policy_id": fallback_policy_id,
                    "profile_confidence": PROFILE_CONFIDENCE[profile],
                    "predicted_net_pnl_bps": selected[
                        "predicted_net_pnl_bps"
                    ],
                    "prediction_standard_error_bps": selected[
                        "prediction_standard_error_bps"
                    ],
                    "net_pnl_lower_bound_bps": selected[
                        "net_pnl_lower_bound_bps"
                    ],
                    "realized_risk_bps": selected["realized_risk_bps"],
                    "realized_risk_budget_bps": selected[
                        "realized_risk_budget_bps"
                    ],
                    "contextual_ridge_multiplier": selected[
                        "contextual_ridge_multiplier"
                    ],
                    "contextual_cv_rmse_bps": selected[
                        "contextual_cv_rmse_bps"
                    ],
                    **{
                        f"current_feature_{column}": value
                        for column, value in zip(feature_names, current_features)
                    },
                    **{
                        column: float(policy[column])
                        for column in POLICY_COEFFICIENT_COLUMNS
                    },
                    "selected_net_pnl_bps": float(selected_outcome["net_pnl_bps"]),
                    "fallback_net_pnl_bps": float(fallback_outcome["net_pnl_bps"]),
                    "selected_within_event_drawdown_bps": float(
                        selected_outcome["within_event_max_drawdown_bps"]
                    ),
                    "fallback_within_event_drawdown_bps": float(
                        fallback_outcome["within_event_max_drawdown_bps"]
                    ),
                    "selected_hard_pass": bool(selected_outcome["hard_pass"]),
                    "selected_behavior_pass": bool(
                        selected_outcome["behavior_pass"]
                    ),
                }
            )
        prior_events.append(pd.Series(current._asdict()))

    selections = pd.DataFrame(selection_rows)
    evaluations = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame()
    )
    return AutomaticRiskProfileCalibration(
        selections=selections,
        policy_evaluations=evaluations,
        summary=summarize_risk_profile_selections(selections),
        policies=policy_frame.copy(),
    )


def summarize_risk_profile_selections(selections: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-sample calibrated policy economics by risk label."""

    calibrated = selections.loc[
        selections["status"].str.startswith(("calibrated", "contextual"))
    ].copy()
    rows: list[dict[str, object]] = []
    for profile in PROFILE_ORDER:
        group = calibrated.loc[calibrated["risk_aversion"].eq(profile.value)]
        if group.empty:
            rows.append(
                {
                    "risk_aversion": profile.value,
                    "event_count": 0,
                    "mean_selected_net_pnl_bps": np.nan,
                    "mean_fallback_net_pnl_bps": np.nan,
                    "mean_net_pnl_delta_bps": np.nan,
                    "selected_pnl_vol_bps": np.nan,
                    "fallback_pnl_vol_bps": np.nan,
                    "selected_loss_cvar_95_bps": np.nan,
                    "fallback_loss_cvar_95_bps": np.nan,
                    "mean_selected_drawdown_bps": np.nan,
                    "mean_policy_aggressiveness": np.nan,
                    "all_selected_hard_pass": False,
                    "selected_behavior_pass_rate": np.nan,
                    "economically_viable_selection_rate": np.nan,
                }
            )
            continue
        selected_pnl = group["selected_net_pnl_bps"].to_numpy(float)
        fallback_pnl = group["fallback_net_pnl_bps"].to_numpy(float)
        rows.append(
            {
                "risk_aversion": profile.value,
                "event_count": len(group),
                "mean_selected_net_pnl_bps": float(np.mean(selected_pnl)),
                "mean_fallback_net_pnl_bps": float(np.mean(fallback_pnl)),
                "mean_net_pnl_delta_bps": float(
                    np.mean(selected_pnl - fallback_pnl)
                ),
                "selected_pnl_vol_bps": _sample_std(selected_pnl),
                "fallback_pnl_vol_bps": _sample_std(fallback_pnl),
                "selected_loss_cvar_95_bps": _loss_cvar_95(selected_pnl),
                "fallback_loss_cvar_95_bps": _loss_cvar_95(fallback_pnl),
                "mean_selected_drawdown_bps": float(
                    group["selected_within_event_drawdown_bps"].mean()
                ),
                "mean_policy_aggressiveness": float(
                    group["policy_aggressiveness"].mean()
                ),
                "all_selected_hard_pass": bool(group["selected_hard_pass"].all()),
                "selected_behavior_pass_rate": float(
                    group["selected_behavior_pass"].mean()
                ),
                "economically_viable_selection_rate": float(
                    group["economically_viable"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _contextual_policy_predictions(
    feature_history: np.ndarray,
    history: pd.DataFrame,
    policies: pd.DataFrame,
    current_features: np.ndarray,
    *,
    feature_names: Sequence[str],
    ridge_multipliers: np.ndarray,
) -> pd.DataFrame:
    event_order = list(dict.fromkeys(history["event_id"].astype(str)))
    policy_order = policies["policy_id"].astype(str).tolist()
    outcomes = (
        history.pivot(index="event_id", columns="policy_id", values="net_pnl_bps")
        .reindex(index=event_order, columns=policy_order)
        .to_numpy(float)
    )
    features = np.asarray(feature_history, dtype=float)
    current = np.asarray(current_features, dtype=float)
    if features.shape != (len(event_order), len(feature_names)):
        raise ValueError("feature history must align with eligible event history")
    if current.shape != (len(feature_names),):
        raise ValueError("current features must align with feature_columns")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("contextual policy outcomes must be finite and complete")

    cv_rows: list[tuple[float, float]] = []
    for multiplier in ridge_multipliers:
        squared_errors: list[np.ndarray] = []
        for held_out in range(len(features)):
            keep = np.arange(len(features)) != held_out
            prediction, _, _, _, _, _ = _fit_contextual_ridge(
                features[keep],
                outcomes[keep],
                features[held_out],
                float(multiplier),
            )
            squared_errors.append(np.square(prediction - outcomes[held_out]))
        cv_rows.append(
            (
                float(multiplier),
                float(np.sqrt(np.mean(np.vstack(squared_errors)))),
            )
        )
    selected_multiplier, cv_rmse = min(cv_rows, key=lambda row: (row[1], row[0]))
    (
        prediction,
        prediction_error,
        feature_mean,
        feature_scale,
        leverage,
        residual_std,
    ) = _fit_contextual_ridge(
        features,
        outcomes,
        current,
        selected_multiplier,
    )
    rows: list[dict[str, object]] = []
    for policy_index, policy_id in enumerate(policy_order):
        rows.append(
            {
                "policy_id": policy_id,
                "predicted_net_pnl_bps": float(prediction[policy_index]),
                "prediction_standard_error_bps": float(
                    prediction_error[policy_index]
                ),
                "contextual_ridge_multiplier": selected_multiplier,
                "contextual_cv_rmse_bps": cv_rmse,
                "contextual_leverage": leverage,
                "contextual_residual_std_bps": float(
                    residual_std[policy_index]
                ),
                **{
                    f"training_feature_mean_{name}": float(value)
                    for name, value in zip(feature_names, feature_mean)
                },
                **{
                    f"training_feature_scale_{name}": float(value)
                    for name, value in zip(feature_names, feature_scale)
                },
            }
        )
    return pd.DataFrame(rows)


def _fit_contextual_ridge(
    features: np.ndarray,
    outcomes: np.ndarray,
    prediction_features: np.ndarray,
    multiplier: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
]:
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    standardized = (features - feature_mean) / feature_scale
    current = (prediction_features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(features)), standardized])
    current_design = np.concatenate([[1.0], current])
    penalty = np.diag(np.concatenate([[0.0], np.ones(features.shape[1])]))
    information = design.T @ design + multiplier * penalty
    inverse = np.linalg.pinv(information, rcond=1e-12)
    coefficients = inverse @ design.T @ outcomes
    prediction = current_design @ coefficients
    residuals = outcomes - design @ coefficients
    degrees_of_freedom = max(len(features) - design.shape[1], 1)
    residual_std = np.sqrt(np.sum(np.square(residuals), axis=0) / degrees_of_freedom)
    leverage = float(max(current_design @ inverse @ current_design, 0.0))
    prediction_error = residual_std * np.sqrt(1.0 + leverage)
    return (
        np.asarray(prediction, dtype=float),
        np.asarray(prediction_error, dtype=float),
        np.asarray(feature_mean, dtype=float),
        np.asarray(feature_scale, dtype=float),
        leverage,
        np.asarray(residual_std, dtype=float),
    )


def _select_contextual_profiles(
    statistics: pd.DataFrame,
    *,
    minimum_behavior_pass_rate: float,
    materiality_bps: float,
) -> tuple[
    dict[RiskAversion, dict[str, object]],
    dict[RiskAversion, pd.DataFrame],
]:
    selected: dict[RiskAversion, dict[str, object]] = {}
    evaluations: dict[RiskAversion, pd.DataFrame] = {}
    minimum_aggressiveness = -np.inf
    for profile in PROFILE_ORDER:
        confidence = PROFILE_CONFIDENCE[profile]
        z_score = NormalDist().inv_cdf(confidence)
        evaluation = statistics.copy()
        evaluation["risk_aversion"] = profile.value
        evaluation["profile_confidence"] = confidence
        evaluation["net_pnl_lower_bound_bps"] = (
            evaluation["predicted_net_pnl_bps"]
            - z_score * evaluation["prediction_standard_error_bps"]
        )
        evaluation["monotonic_policy_eligible"] = evaluation[
            "policy_aggressiveness"
        ].ge(minimum_aggressiveness - 1e-12)
        evaluation["operationally_eligible"] = (
            evaluation["hard_pass_all"]
            & evaluation["behavior_pass_rate"].ge(minimum_behavior_pass_rate)
            & evaluation["monotonic_policy_eligible"]
        )
        evaluation["profit_eligible"] = (
            evaluation["operationally_eligible"]
            & evaluation["net_pnl_lower_bound_bps"].ge(0.0)
        )
        profitable = evaluation.loc[evaluation["profit_eligible"]].copy()
        if not profitable.empty:
            pool = profitable
            status = "contextual_profitable"
            economically_viable = True
        else:
            pool = evaluation.loc[evaluation["operationally_eligible"]].copy()
            status = "contextual_no_profitable_policy"
            economically_viable = False
        if pool.empty:
            pool = evaluation.loc[evaluation["monotonic_policy_eligible"]].copy()
            status = "contextual_no_operational_policy"
            economically_viable = False
        min_risk = float(pool["realized_risk_bps"].min())
        max_risk = float(pool["realized_risk_bps"].max())
        risk_fraction = DEFAULT_RISK_PREFERENCES[profile].risk_frontier_fraction
        risk_budget = min_risk + risk_fraction * (max_risk - min_risk)
        inside_budget = pool.loc[
            pool["realized_risk_bps"].le(
                risk_budget + max(1e-12, abs(risk_budget) * 1e-12)
            )
        ].copy()
        best_lower_bound = float(inside_budget["net_pnl_lower_bound_bps"].max())
        tied = inside_budget.loc[
            inside_budget["net_pnl_lower_bound_bps"].ge(
                best_lower_bound - materiality_bps
            )
        ]
        chosen = tied.sort_values(
            [
                "realized_risk_bps",
                "policy_aggressiveness",
                "net_pnl_lower_bound_bps",
                "predicted_net_pnl_bps",
            ],
            ascending=[True, True, False, False],
        ).iloc[0]
        policy_id = str(chosen["policy_id"])
        evaluation["realized_risk_budget_bps"] = risk_budget
        evaluation["within_realized_risk_budget"] = (
            evaluation["realized_risk_bps"]
            <= risk_budget + max(1e-12, abs(risk_budget) * 1e-12)
        )
        evaluation["selected"] = evaluation["policy_id"].eq(policy_id)
        evaluations[profile] = evaluation
        selected[profile] = {
            "selected_policy_id": policy_id,
            "status": status,
            "economically_viable": economically_viable,
            "predicted_net_pnl_bps": float(chosen["predicted_net_pnl_bps"]),
            "prediction_standard_error_bps": float(
                chosen["prediction_standard_error_bps"]
            ),
            "net_pnl_lower_bound_bps": float(
                chosen["net_pnl_lower_bound_bps"]
            ),
            "realized_risk_bps": float(chosen["realized_risk_bps"]),
            "realized_risk_budget_bps": risk_budget,
            "contextual_ridge_multiplier": float(
                chosen["contextual_ridge_multiplier"]
            ),
            "contextual_cv_rmse_bps": float(chosen["contextual_cv_rmse_bps"]),
        }
        minimum_aggressiveness = float(chosen["policy_aggressiveness"])
    return selected, evaluations


def _select_profiles(
    statistics: pd.DataFrame,
    *,
    minimum_behavior_pass_rate: float,
    materiality_bps: float,
) -> tuple[
    dict[RiskAversion, dict[str, object]],
    dict[RiskAversion, pd.DataFrame],
]:
    selected: dict[RiskAversion, dict[str, object]] = {}
    evaluations: dict[RiskAversion, pd.DataFrame] = {}
    minimum_aggressiveness = -np.inf
    for profile in PROFILE_ORDER:
        confidence = PROFILE_CONFIDENCE[profile]
        z_score = NormalDist().inv_cdf(confidence)
        evaluation = statistics.copy()
        evaluation["risk_aversion"] = profile.value
        evaluation["profile_confidence"] = confidence
        evaluation["net_pnl_lower_bound_bps"] = (
            evaluation["mean_net_pnl_bps"]
            - z_score * evaluation["net_pnl_standard_error_bps"]
        )
        evaluation["monotonic_policy_eligible"] = evaluation[
            "policy_aggressiveness"
        ].ge(minimum_aggressiveness - 1e-12)
        evaluation["operationally_eligible"] = (
            evaluation["hard_pass_all"]
            & evaluation["behavior_pass_rate"].ge(minimum_behavior_pass_rate)
            & evaluation["monotonic_policy_eligible"]
        )
        evaluation["profit_eligible"] = (
            evaluation["operationally_eligible"]
            & evaluation["net_pnl_lower_bound_bps"].ge(0.0)
        )
        profitable = evaluation.loc[evaluation["profit_eligible"]].copy()
        if not profitable.empty:
            pool = profitable
            status = "calibrated_profitable"
            economically_viable = True
        else:
            pool = evaluation.loc[evaluation["operationally_eligible"]].copy()
            status = "calibrated_no_profitable_policy"
            economically_viable = False
        if pool.empty:
            pool = evaluation.loc[evaluation["monotonic_policy_eligible"]].copy()
            status = "calibrated_no_operational_policy"
            economically_viable = False
        min_risk = float(pool["realized_risk_bps"].min())
        max_risk = float(pool["realized_risk_bps"].max())
        risk_fraction = DEFAULT_RISK_PREFERENCES[profile].risk_frontier_fraction
        risk_budget = min_risk + risk_fraction * (max_risk - min_risk)
        inside_budget = pool.loc[
            pool["realized_risk_bps"].le(
                risk_budget + max(1e-12, abs(risk_budget) * 1e-12)
            )
        ].copy()
        best_lower_bound = float(inside_budget["net_pnl_lower_bound_bps"].max())
        tied = inside_budget.loc[
            inside_budget["net_pnl_lower_bound_bps"].ge(
                best_lower_bound - materiality_bps
            )
        ]
        chosen = tied.sort_values(
            [
                "realized_risk_bps",
                "policy_aggressiveness",
                "net_pnl_lower_bound_bps",
                "mean_net_pnl_bps",
            ],
            ascending=[True, True, False, False],
        ).iloc[0]
        policy_id = str(chosen["policy_id"])
        evaluation["realized_risk_budget_bps"] = risk_budget
        evaluation["within_realized_risk_budget"] = (
            evaluation["realized_risk_bps"]
            <= risk_budget + max(1e-12, abs(risk_budget) * 1e-12)
        )
        evaluation["selected"] = evaluation["policy_id"].eq(policy_id)
        evaluations[profile] = evaluation
        selected[profile] = {
            "selected_policy_id": policy_id,
            "status": status,
            "economically_viable": economically_viable,
            "net_pnl_lower_bound_bps": float(
                chosen["net_pnl_lower_bound_bps"]
            ),
            "realized_risk_bps": float(chosen["realized_risk_bps"]),
            "realized_risk_budget_bps": risk_budget,
        }
        minimum_aggressiveness = float(chosen["policy_aggressiveness"])
    return selected, evaluations


def _policy_statistics(
    history: pd.DataFrame,
    policies: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy_id, group in history.groupby("policy_id", sort=False):
        pnl = group["net_pnl_bps"].to_numpy(float)
        pnl_std = _sample_std(pnl)
        drawdown = float(group["within_event_max_drawdown_bps"].mean())
        loss_cvar = _loss_cvar_95(pnl)
        rows.append(
            {
                "policy_id": str(policy_id),
                "training_event_count": len(group),
                "mean_net_pnl_bps": float(np.mean(pnl)),
                "net_pnl_standard_error_bps": pnl_std / sqrt(len(pnl)),
                "pnl_vol_bps": pnl_std,
                "loss_cvar_95_bps": loss_cvar,
                "mean_within_event_drawdown_bps": drawdown,
                "realized_risk_bps": max(pnl_std, loss_cvar, drawdown),
                "probability_profitable": float(np.mean(pnl > 0.0)),
                "hard_pass_all": bool(group["hard_pass"].all()),
                "behavior_pass_rate": float(group["behavior_pass"].mean()),
            }
        )
    return policies.merge(pd.DataFrame(rows), on="policy_id", validate="one_to_one")


def _loss_cvar_95(pnl_bps: np.ndarray) -> float:
    losses = -np.asarray(pnl_bps, dtype=float)
    tail_count = max(1, int(ceil(0.05 * len(losses))))
    return max(0.0, float(np.mean(np.sort(losses)[-tail_count:])))


def _sample_std(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _fallback_policy_ids(
    policies: pd.DataFrame,
) -> dict[RiskAversion, str]:
    return {
        profile: str(
            policies.iloc[
                np.argmin(
                    np.abs(
                        policies["policy_aggressiveness"].to_numpy(float)
                        - target
                    )
                )
            ]["policy_id"]
        )
        for profile, target in FALLBACK_AGGRESSIVENESS.items()
    }


def _validated_inputs(
    events: pd.DataFrame,
    trials: pd.DataFrame,
    policies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_frame = events.copy()
    trial_frame = trials.copy()
    policy_frame = policies.copy()
    _require_columns(event_frame, EVENT_COLUMNS, "events")
    _require_columns(trial_frame, TRIAL_COLUMNS, "trials")
    _require_columns(
        policy_frame,
        ("policy_id", *POLICY_COEFFICIENT_COLUMNS),
        "policies",
    )
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    trial_frame["event_id"] = trial_frame["event_id"].astype(str)
    trial_frame["policy_id"] = trial_frame["policy_id"].astype(str)
    policy_frame["policy_id"] = policy_frame["policy_id"].astype(str)
    if event_frame["event_id"].duplicated().any():
        raise ValueError("events.event_id values must be unique")
    if policy_frame["policy_id"].duplicated().any():
        raise ValueError("policies.policy_id values must be unique")
    if trial_frame.duplicated(["event_id", "policy_id"]).any():
        raise ValueError("trials must contain one row per event and policy")
    for column in ("as_of", "information_cutoff", "realized_available_at"):
        event_frame[column] = pd.to_datetime(event_frame[column], errors="raise")
    if not event_frame["as_of"].is_monotonic_increasing:
        raise ValueError("events must be ordered chronologically by as_of")
    if (event_frame["information_cutoff"] > event_frame["as_of"]).any():
        raise ValueError("information_cutoff must be on or before as_of")
    if (event_frame["realized_available_at"] <= event_frame["as_of"]).any():
        raise ValueError("realized_available_at must be after as_of")
    event_ids = set(event_frame["event_id"])
    policy_ids = set(policy_frame["policy_id"])
    if set(trial_frame["event_id"]) != event_ids:
        raise ValueError("trials event_id values must exactly match events")
    if set(trial_frame["policy_id"]) != policy_ids:
        raise ValueError("trials policy_id values must exactly match policies")
    expected_rows = len(event_frame) * len(policy_frame)
    if len(trial_frame) != expected_rows:
        raise ValueError("trials must contain a complete event-policy grid")
    numeric_trial_columns = ("net_pnl_bps", "within_event_max_drawdown_bps")
    for column in numeric_trial_columns:
        trial_frame[column] = pd.to_numeric(trial_frame[column], errors="raise")
        if not np.isfinite(trial_frame[column].to_numpy(float)).all():
            raise ValueError(f"trials.{column} must be finite")
    if (trial_frame["within_event_max_drawdown_bps"] < 0.0).any():
        raise ValueError("within_event_max_drawdown_bps must be non-negative")
    for column in ("hard_pass", "behavior_pass"):
        if not trial_frame[column].isin((True, False)).all():
            raise ValueError(f"trials.{column} must be boolean")
        trial_frame[column] = trial_frame[column].astype(bool)
    for column in POLICY_COEFFICIENT_COLUMNS:
        policy_frame[column] = pd.to_numeric(policy_frame[column], errors="raise")
        values = policy_frame[column].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"policies.{column} must be finite and between zero and one")
    if (
        (policy_frame["liquidity_quantile"] <= 0.0).any()
        or (policy_frame["liquidity_quantile"] >= 1.0).any()
    ):
        raise ValueError("liquidity_quantile must be strictly between zero and one")
    if (
        (policy_frame["alpha_confidence"] < 0.50).any()
        or (policy_frame["alpha_confidence"] >= 1.0).any()
    ):
        raise ValueError("alpha_confidence must be between 0.50 inclusive and 1.0 exclusive")
    if policy_frame["policy_aggressiveness"].duplicated().any():
        raise ValueError("policy_aggressiveness values must be unique")
    if (policy_frame["policy_aggressiveness"] <= 0.0).any():
        raise ValueError("policy_aggressiveness must be positive")
    ordered = policy_frame.sort_values("policy_aggressiveness").reset_index(drop=True)
    _require_monotone(ordered, "risk_frontier_fraction", increasing=True)
    _require_monotone(ordered, "liquidity_quantile", increasing=True)
    _require_monotone(ordered, "liquidity_shape_fraction", increasing=False)
    _require_monotone(ordered, "alpha_confidence", increasing=False)
    _require_monotone(ordered, "factor_stress_fraction", increasing=False)
    return (
        event_frame.reset_index(drop=True),
        trial_frame.reset_index(drop=True),
        ordered,
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _require_monotone(
    policies: pd.DataFrame,
    column: str,
    *,
    increasing: bool,
) -> None:
    differences = np.diff(policies[column].to_numpy(float))
    valid = np.all(differences >= -1e-12) if increasing else np.all(differences <= 1e-12)
    if not valid:
        direction = "non-decreasing" if increasing else "non-increasing"
        raise ValueError(f"policies.{column} must be {direction} with aggressiveness")
