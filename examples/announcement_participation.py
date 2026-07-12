"""Day 1-15 announcement participation example from the project brief."""

import numpy as np
import pandas as pd

from trade_planner import AnnouncementParticipationCurve


dates = pd.date_range("2026-07-01", periods=15, freq="D")
announcement_date = dates[9]  # Day 10; announcement day stays in the pre regime.

step_rates = AnnouncementParticipationCurve(
    pre_rate=0.025,
    post_rate=0.15,
).rates(dates, announcement_date)

volatility = np.linspace(0.40, 0.15, len(dates))
dynamic_rates = AnnouncementParticipationCurve(
    pre_rate=0.025,
    post_rate=0.15,
    transition="logistic",
    transition_days=2,
    reference_volatility=0.20,
    pre_volatility_sensitivity=0.5,
    post_volatility_sensitivity=0.5,
).rates(dates, announcement_date, volatility=volatility)

print(pd.DataFrame({"step_rate": step_rates, "volatility_adjusted_rate": dynamic_rates}))
