"""Small, solver-independent analytics for trade schedules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .context import PlannerContext


def cumulative_side_completion(
    ctx: PlannerContext,
    schedule: pd.DataFrame,
    *,
    reference_date_index: int = 0,
) -> pd.DataFrame:
    """Return daily and cumulative long/short gross-notional completion.

    All notionals use each symbol's price on ``reference_date_index`` so price
    moves cannot make terminal completion differ from 100%.  Output percentage
    columns use the 0-to-100 scale.
    """
    required = {"date", "symbol", "trade_shares"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule is missing required columns: {sorted(missing)}")
    if not 0 <= reference_date_index < len(ctx.dates):
        raise IndexError("reference_date_index is outside the planner horizon")

    targets = ctx.orders["target_shares"].reindex(ctx.symbols).to_numpy(float)
    reference_prices = np.asarray(ctx.price[reference_date_index], dtype=float)
    symbol_index = {symbol: index for index, symbol in enumerate(ctx.symbols)}

    frame = schedule.loc[:, ["date", "symbol", "trade_shares"]].copy()
    frame["date"] = pd.DatetimeIndex(pd.to_datetime(frame["date"])).normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    unknown = sorted(set(frame["symbol"]) - set(symbol_index))
    if unknown:
        raise ValueError(f"schedule contains symbols outside the context: {unknown}")

    indices = frame["symbol"].map(symbol_index).to_numpy(int)
    frame["side"] = np.where(targets[indices] > 0, "long", np.where(targets[indices] < 0, "short", "flat"))
    frame["reference_notional"] = (
        np.abs(frame["trade_shares"].to_numpy(float)) * reference_prices[indices]
    )
    frame = frame[frame["side"] != "flat"]

    daily = frame.pivot_table(
        index="date",
        columns="side",
        values="reference_notional",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(ctx.dates, fill_value=0.0)
    for side in ("long", "short"):
        if side not in daily:
            daily[side] = 0.0

    total_long = float(np.sum(np.abs(targets[targets > 0]) * reference_prices[targets > 0]))
    total_short = float(np.sum(np.abs(targets[targets < 0]) * reference_prices[targets < 0]))
    total_gross = total_long + total_short

    result = pd.DataFrame(index=ctx.dates)
    result.index.name = "date"
    result["daily_long_pct"] = _percentage(daily["long"].to_numpy(float), total_long)
    result["daily_short_pct"] = _percentage(daily["short"].to_numpy(float), total_short)
    result["cumulative_long_pct"] = _percentage(
        daily["long"].cumsum().to_numpy(float),
        total_long,
    )
    result["cumulative_short_pct"] = _percentage(
        daily["short"].cumsum().to_numpy(float),
        total_short,
    )
    result["cumulative_gross_pct"] = _percentage(
        (daily["long"] + daily["short"]).cumsum().to_numpy(float),
        total_gross,
    )
    if total_long > 0 and total_short > 0:
        result["long_short_gap_pp"] = (
            result["cumulative_long_pct"] - result["cumulative_short_pct"]
        )
    else:
        result["long_short_gap_pp"] = np.nan
    return result


def _percentage(values: np.ndarray, denominator: float) -> np.ndarray:
    if denominator <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return 100.0 * values / denominator
