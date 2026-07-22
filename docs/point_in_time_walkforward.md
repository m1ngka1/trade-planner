# Point-in-time rebalance replay

## Decision

Keep the replay infrastructure. Do not automatically apply either tested
forecast-uncertainty adjustment.

The repository previously evaluated schedules with independent synthetic return
scenarios, but it had no event-level contract that physically separated data
available to the optimizer from later realized returns and costs. The new
`PointInTimeRebalanceEvent` closes that gap:

- `ctx` contains only the planning snapshot;
- `information_cutoff <= as_of <= first planner date` is checked;
- realized returns and TCA costs live outside `ctx`;
- `realized_available_at` must be after the final planner date; and
- replay reports event P&L, costs, hit rate, volatility, loss CVaR, drawdown,
  and completion error in dollars and basis points of parent gross.

This is necessary infrastructure for the final production gate. It is not the
gate itself because no proprietary historical rebalance baskets or realized
fills are stored in this repository.

## Realized P&L convention

For cumulative dollar inventory `w_t`, realized holding return `r_t`, executed
shares `q_t`, realized impact surface `eta_t`, and realized linear costs `c_t`,
the replay measures:

```text
daily realized P&L_t
    = r_t' w_t
    - eta_t' square(q_t)
    - c_t' abs(q_t)
```

The return at date `t` is earned after that planner date, matching the
accumulated-inventory objective. Event P&L is never calculated from
`ctx.expected_return`.

## Usage

```python
from trade_planner import (
    PointInTimeRebalanceEvent,
    calibrate_rebalance_plan,
    replay_rebalance_events,
)

event = PointInTimeRebalanceEvent(
    event_id="2026-06-hk-rebalance",
    as_of="2026-05-25 16:00:00",
    information_cutoff="2026-05-25 16:00:00",
    ctx=point_in_time_ctx,
    realized_returns=realized_holding_returns,
    realized_impact_bps_at_10pct_adv=realized_impact,
    realized_linear_cost_bps=realized_spread_fees,
    realized_available_at="2026-06-02 18:00:00",
)

replay = replay_rebalance_events(
    [event],
    {
        "medium": lambda item: calibrate_rebalance_plan(
            item.ctx,
            risk_aversion="medium",
        ),
    },
)

print(replay.events)
print(replay.summary)
print(replay.daily)
```

The data provider can optionally load `expected_return_uncertainty`, defined as
the point-in-time standard error of each probability-weighted holding-return
forecast. The core production policy does not use it automatically; it is
available for controlled research through
`ConfidenceAdjustedExpectedReturnAlphaModel`.

## Development experiment

The development replay uses 12 chronological synthetic planning snapshots. A
disjoint 2,000-draw pre-replay sample estimates forecast standard errors; event
returns, optimizer scenarios, and realized TCA shocks use separate seeds. This
checks the replay and adjustment mechanics without pretending to be historical
evidence.

Acceptance requires:

- no more than 1 bp average realized-P&L sacrifice where applicable;
- lower event P&L volatility;
- no higher 95% loss CVaR;
- urgent flow never starts later;
- small flow never starts earlier;
- early factor balance is preserved; and
- at least 90% of the baseline late/early volume ratio is retained.

### Idea 1: confidence-adjusted expected alpha

The candidate subtracts a one-sided forecast-error hurdle from expected alpha.
It was tested at 60%, 65%, 70%, and 75% confidence, both by reselecting the
risk frontier and by holding the baseline risk coefficient fixed. The 60% and
70% fixed-risk variants were also combined with the retained hybrid tail model.

| Variant | Realized P&L delta | Volatility delta | Loss-CVaR delta | Early-factor delta | Decision |
|---|---:|---:|---:|---:|---|
| Variance, reselect, 60% | +$177k | -0.74 bp | +1.09 bp | +0.23 pp | Discard |
| Variance, reselect, 65% | -$795k | +2.34 bp | +23.88 bp | +0.66 pp | Discard |
| Variance, reselect, 70% | +$1,204k | +3.30 bp | +3.48 bp | +0.85 pp | Discard |
| Variance, reselect, 75% | +$624k | +0.92 bp | +4.85 bp | +1.04 pp | Discard |
| Variance, fixed risk, 60% | +$210k | -0.92 bp | +1.09 bp | +0.34 pp | Discard |
| Variance, fixed risk, 65% | +$296k | -1.28 bp | +2.19 bp | +0.59 pp | Discard |
| Variance, fixed risk, 70% | +$308k | -1.66 bp | +3.48 bp | +0.87 pp | Discard |
| Variance, fixed risk, 75% | +$260k | -1.96 bp | +4.85 bp | +1.16 pp | Discard |
| Hybrid, fixed risk, 60% | +$103k | -0.90 bp | +1.09 bp | +0.52 pp | Discard |
| Hybrid, fixed risk, 70% | +$123k | -1.50 bp | +3.48 bp | +1.18 pp | Discard |

The fixed-risk variants improved average P&L and volatility, but every one made
the worst event—and therefore empirical 95% loss CVaR—worse. Reselecting the
frontier also caused discontinuous small-order timing in some events. The
production default therefore remains the probability-weighted raw expected
return, which is already supposed to incorporate call confidence.

### Idea 2: reliability-scaled medium risk budget

The second candidate leaves expected alpha untouched and contracts the medium
risk-frontier fraction when signal dollars are small relative to forecast-error
dollars:

```text
reliability = positive directional alpha dollars
              / (positive directional alpha dollars + uncertainty dollars)

effective medium fraction = 50% * (50% + 50% * reliability)
```

The mean reliability was 44.4%, producing a mean 36.1% risk fraction instead of
the fixed 50%.

| Risk model | P&L delta | Volatility delta | Loss-CVaR delta | Factor delta | Small start | Decision |
|---|---:|---:|---:|---:|---:|---|
| Variance | -$170k (-0.29 bp/event) | +0.21 bp | 0.00 bp | -0.05 pp | +0.67 day | Discard |
| Hybrid downside | $0 | 0.00 bp | 0.00 bp | 0.00 pp | 0.00 day | Discard: no schedule change |

The variance version improved early balance and delayed small flow, but did not
reduce realized volatility. On the hybrid frontier both budgets selected the
same discrete candidate, so the rule added complexity without changing the
schedule. It is not promoted.

The chronological holdout events were deliberately left unopened because no
candidate passed development gates. This avoids turning the holdout into
another tuning sample.

### Idea 3: uncertainty-aware frontier ties

The third candidate leaves the optimizer, scenarios, medium 50% risk budget,
and expected alpha unchanged. It only treats a lower-risk frontier point as tied
with the highest-P&L eligible point when the forecast P&L gap is smaller than a
one-sided confidence charge on their paired inventory difference. Confidence
levels of 60%, 75%, and 90% were screened.

This produced no schedule changes across 12 development events. Paired alpha
uncertainty averaged $5.9k and peaked at $13.1k. Even the 90% one-sided charge
remained below the existing $48.7k threshold, which is one basis point of the
$487.2m parent basket. The current economic-materiality rule therefore already
absorbed more forecast noise than the proposed statistical rule. All three
variants are recorded as inconclusive/redundant, the production selector is
unchanged, and the chronological holdout remains unopened.

## Reproduce

```bash
env PYTHONPATH=. python experiments/alpha_confidence_walkforward.py \
  --risk-measure variance \
  --event-start 0 \
  --n-events 12 \
  --alpha-confidence 0.60 \
  --selection-policy fixed_risk \
  --output-prefix artifacts/alpha_confidence_fixed_dev_60

env PYTHONPATH=. python experiments/uncertainty_budget_walkforward.py \
  --risk-measure variance \
  --event-start 0 \
  --n-events 12 \
  --output-prefix artifacts/uncertainty_budget_dev

env PYTHONPATH=. python experiments/frontier_uncertainty_selection.py \
  --risk-measure variance \
  --event-start 0 \
  --n-events 12 \
  --confidences 0.60 0.75 0.90 \
  --output-prefix artifacts/frontier_uncertainty_dev
```

Each prefix produces a trial ledger, paired event deltas, summary, complete
schedules, daily P&L, and a four-panel PNG. The hybrid confirmation prefixes are
`alpha_confidence_hybrid_dev_*` and `uncertainty_budget_hybrid_dev_*`.
`artifacts/walkforward_research_ledger.csv` is the compact cross-trial index of
all 15 tested variants, their comparable deltas, decisions, reasons, and source
artifact prefixes.

## Production completion gate

Populate `PointInTimeRebalanceEvent` objects from actual rebalance prediction
snapshots and realized close/VWAP, fills, spread, impact, FX, financing, and
borrow data. Then compare the current automatic high/medium/low policy against
the desk baseline across enough independent events for stable tail estimates.
Until that replay exists, synthetic expected P&L and `economically_viable` are
model outputs, not proof that the strategy makes money.
