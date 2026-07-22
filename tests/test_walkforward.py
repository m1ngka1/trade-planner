from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from trade_planner import (
    PlannerContext,
    PointInTimeRebalanceEvent,
    evaluate_realized_rebalance_schedule,
    replay_rebalance_events,
    validate_point_in_time_event,
)


class WalkForwardTests(unittest.TestCase):
    def test_realized_economics_use_only_realized_returns_and_costs(self) -> None:
        event, schedule = _event_and_schedule()

        metrics, daily = evaluate_realized_rebalance_schedule(event, schedule)

        self.assertAlmostEqual(metrics.gross_holding_pnl_dollars, 50.0)
        self.assertAlmostEqual(metrics.impact_cost_dollars, 0.0)
        self.assertAlmostEqual(metrics.linear_cost_dollars, 0.0)
        self.assertAlmostEqual(metrics.net_pnl_dollars, 50.0)
        self.assertAlmostEqual(metrics.net_pnl_bps, 250.0)
        self.assertAlmostEqual(metrics.terminal_completion_error_shares, 0.0)
        np.testing.assert_allclose(daily["net_pnl_dollars"], [10.0, 40.0])

    def test_replay_returns_event_and_strategy_level_evidence(self) -> None:
        event, schedule = _event_and_schedule()

        replay = replay_rebalance_events(
            [event],
            {
                "candidate": lambda _: schedule,
                "control": lambda _: schedule,
            },
        )

        self.assertEqual(len(replay.events), 2)
        self.assertEqual(len(replay.summary), 2)
        self.assertEqual(len(replay.daily), 4)
        self.assertTrue((replay.summary["mean_net_pnl_bps"] == 250.0).all())

    def test_point_in_time_contract_rejects_future_information(self) -> None:
        event, _ = _event_and_schedule()
        leaked = PointInTimeRebalanceEvent(
            **{
                **event.__dict__,
                "information_cutoff": "2026-01-03",
            }
        )

        with self.assertRaisesRegex(ValueError, "information_cutoff"):
            validate_point_in_time_event(leaked)

    def test_realized_adv_drives_impact_and_participation_audit(self) -> None:
        event, schedule = _event_and_schedule()
        realized_adv = np.full((2, 2), 500.0)
        event = replace(
            event,
            realized_adv_shares=realized_adv,
            realized_impact_bps_at_10pct_adv=10.0,
        )

        metrics, daily = evaluate_realized_rebalance_schedule(event, schedule)

        self.assertAlmostEqual(metrics.impact_cost_dollars, 0.2)
        self.assertAlmostEqual(metrics.max_realized_participation_rate, 0.01)
        self.assertAlmostEqual(metrics.p95_realized_participation_rate, 0.01)
        self.assertAlmostEqual(
            metrics.max_realized_participation_excess_shares,
            0.0,
        )
        np.testing.assert_allclose(
            daily["max_realized_participation_rate"],
            [0.01, 0.01],
        )

    def test_point_in_time_contract_rejects_invalid_realized_adv(self) -> None:
        event, _ = _event_and_schedule()
        invalid = replace(event, realized_adv_shares=np.zeros((2, 2)))

        with self.assertRaisesRegex(ValueError, "realized_adv_shares"):
            validate_point_in_time_event(invalid)


def _event_and_schedule() -> tuple[PointInTimeRebalanceEvent, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-02", periods=2)
    symbols = ["BUY", "SELL"]
    shape = (2, 2)
    ctx = PlannerContext(
        symbols=symbols,
        dates=dates,
        orders=pd.DataFrame({"target_shares": [10.0, -10.0]}, index=symbols),
        panel=pd.DataFrame(index=pd.MultiIndex.from_product([dates, symbols])),
        price=np.full(shape, 100.0),
        adv_shares=np.full(shape, 1_000.0),
        is_open=np.ones(shape, dtype=bool),
        base_participation=np.ones(shape),
        event_days=pd.DataFrame(np.inf, index=dates, columns=symbols),
    )
    schedule = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "symbol": symbols * 2,
            "trade_shares": [5.0, -5.0, 5.0, -5.0],
        }
    )
    event = PointInTimeRebalanceEvent(
        event_id="event-1",
        as_of="2026-01-01 16:00",
        information_cutoff="2026-01-01 16:00",
        ctx=ctx,
        realized_returns=np.array([[0.01, -0.01], [0.02, -0.02]]),
        realized_impact_bps_at_10pct_adv=0.0,
        realized_linear_cost_bps=0.0,
        realized_available_at="2026-01-06 18:00",
    )
    return event, schedule


if __name__ == "__main__":
    unittest.main()
