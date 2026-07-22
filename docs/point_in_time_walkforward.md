# Point-in-time rebalance replay

## Decision

Keep the replay infrastructure. Do not automatically apply any tested
forecast-uncertainty adjustment without real historical confirmation.

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
`ConfidenceAdjustedExpectedReturnAlphaModel` and the recorded forecast-error
path-risk experiment.

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

### Idea 4: receding-horizon execution

The next hypothesis replaced the one-snapshot plan with model-predictive
execution. The optimizer still chooses every trade: on each morning the
experiment observes a new point-in-time forecast vintage, solves the remaining
order's medium-risk economic frontier, executes only that day's slice, and
rolls the actual remaining shares forward. It does not prescribe a volume
curve, an urgent-name start date, or a small-order delay.

Synthetic forecast revisions are generated from a latent alpha state only in
the replay data-generating process. The optimizer receives the noisy vintage,
never latent alpha or realized returns. Forecast error retains 65% per day and
converges toward a 20% uncertainty floor. These values describe a synthetic
information environment; production must estimate them from stored forecast
vintages rather than expose them as user inputs.

Daily fills are converted to whole shares. Rounding is constrained by today's
participation cap, the remaining parent order, and the minimum current trade
needed to fit within future whole-share capacity. This is the discrete version
of the existing hard-completion constraint, not a pacing rule. A false
infeasibility from one numerical backend triggers the other backend on the
identical objective and constraints, and the coefficient audit records the
solver actually used.

Unconditional daily re-optimization failed development. Although empirical
loss CVaR improved 2.51 bp, total P&L fell $1.114m, event volatility increased
0.46 bp, mean within-event drawdown increased 2.77 bp, and the late/early ratio
fell 0.149. The shrinking horizon repeatedly made near-term trading look more
urgent and consumed the late-volume reserve. This is a structural reason not
to deploy naïve receding-horizon control.

### Idea 5: commitment-aware recourse

Commitment-aware recourse compares the newly selected frontier plan with the
active prior plan under the current forecast. A replan is accepted only if one
of these investment cases clears an automatic basis-point materiality hurdle:

```text
profit case:
    expected P&L gain >= hurdle
    and forecast volatility/CVaR degradation <= hurdle

defensive case:
    forecast volatility or CVaR reduction >= hurdle
    and expected P&L sacrifice <= hurdle
```

The hurdle is measured in basis points of remaining parent gross. It is a desk
materiality policy, not a CVXPY coefficient and not a user-entered schedule.
Every accepted frontier still obtains its inventory and path-risk coefficients
automatically from the selected `medium` risk profile and the remaining
basket's economics.

| Development policy | P&L delta | Volatility delta | Loss-CVaR delta | Within-event DD delta | Factor delta | Ramp delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Always replan | -$1,114k | +0.464 bp | -2.508 bp | +2.767 bp | -0.019 pp | -0.149 | Discard |
| Materiality 0.5 bp | -$449k | +0.136 bp | -2.167 bp | +0.024 bp | -0.437 pp | -0.039 | Discard |
| Materiality 1.0 bp | -$449k | +0.161 bp | -2.225 bp | -0.110 bp | -0.438 pp | -0.032 | Discard |
| Materiality 2.0 bp | -$242k | -0.021 bp | -4.610 bp | -0.450 bp | -0.202 pp | +0.004 | Discard |
| Defensive-only 2.0 bp | -$242k | -0.021 bp | -4.610 bp | -0.450 bp | -0.202 pp | +0.004 | Discard: same schedule |
| Defensive-only 4.0 bp | -$579k | +0.417 bp | -4.535 bp | +0.250 bp | -0.022 pp | +0.020 | Discard |

The 2 bp schedule passed every gate except the requirement that volatility fall
by at least 0.05 bp rather than solver/noise scale: it improved volatility by
only 0.0207 bp. Raising the defensive hurdle to 4 bp then worsened both
volatility and drawdown, so the near-pass was not a monotonic risk-control
effect. The result is useful enough to retain for historical replay, but not
strong enough to change production or open the chronological holdout. The
current static high/medium/low policy remains the production default.

### Idea 6: economically proximal recourse

The next candidate moved commitment inside the optimizer. Each daily solve
adds an economic charge for departing from the still-active schedule, so a
forecast revision must be valuable enough to pay for changing the execution
plan. The charge is calibrated automatically from the basket's remaining
forecast uncertainty and the selected high/medium/low profile. The corrected
version prices the difference between two noisy forecast vintages and applies
a simultaneous one-sided confidence hurdle across the remaining solve dates;
it is not a user-entered trading rule.

The trade-proximal objective is an L1 charge on changed dollars. A shifted
dollar is divided by two because moving it from one date to another appears
once as a removal and once as an addition. The experiment still leaves all
trade dates and quantities to the optimizer.

| Development policy | P&L delta | Volatility delta | Loss-CVaR delta | Within-event DD delta | Factor delta | Ramp delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 95% single-comparison hurdle | -$571k | +0.269 bp | -5.581 bp | +0.069 bp | -0.554 pp | -0.013 | Discard |
| 95% simultaneous hurdle | -$532k | +0.191 bp | -5.689 bp | +0.101 bp | -0.386 pp | -0.004 | Discard |

Both versions improved empirical loss CVaR and early factor balance while
preserving hard capacity, direction, completion, urgency, and broad volume-ramp
gates. Neither reduced event P&L volatility; both increased within-event
drawdown and moved one small order two days earlier. The simultaneous
correction reduced but did not remove those failures.

A quadratic inventory-path formulation was then added because an L1
dollar-day formulation was numerically unstable. The quadratic model solved
cleanly, but a two-event mechanical screen immediately failed P&L, volatility,
loss-CVaR, drawdown, and rank-ramp gates. It was stopped before the 12-event
development sweep. No proximal variant reached the chronological holdout, and
none changes the production selector.

### Idea 7: forecast-error path risk

The seventh candidate treats expected-return estimation error as another
source of accumulated P&L variance. Independent forecast errors contribute the
sum of squared uncertainty-dollar exposures. A persistent error in the parent
order direction contributes one squared horizon exposure, capturing an event
call that makes additions and deletions jointly look too attractive.

This is an investment-risk decomposition, not another free coefficient. A
disjoint synthetic forecast history estimates the persistent directional scale
at `0.552305`. Market and forecast-error variance are both dollars squared and
receive the same automatically selected inventory-risk coefficient. The user
still chooses only high, medium, or low risk aversion.

| Split | P&L delta | Volatility delta | Loss-CVaR delta | Within-event DD delta | Factor delta | Ramp delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Development, events 1-12 | +$38.7k | -0.161 bp | -0.214 bp | -0.104 bp | +0.032 pp | +0.012 | Pass to holdout |
| Untouched holdout, events 13-24 | -$33.7k | +0.022 bp | -0.038 bp | -0.037 bp | +0.026 pp | +0.010 | Fail volatility gate |
| Combined, descriptive only | +$5.0k | -0.086 bp | -0.280 bp | -0.071 bp | +0.029 pp | +0.011 | Not a promotion test |

Development passed all 13 predeclared gates without parameter tuning, so the
chronological holdout was opened exactly once. The holdout preserved every
operational and shape gate and slightly improved loss-CVaR, drawdown, and the
volume ramp, but realized volatility rose by 0.022 bp instead of falling by at
least 0.05 bp. That one failure rejects production promotion. The combined
sample is useful description, not a substitute decision, and the spent holdout
must not be tuned against. See [Forecast-error path risk](forecast_error_risk.md)
for the full contract, automatic calibration, commands, and artifact map.

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

env PYTHONPATH=. python experiments/rolling_horizon_walkforward.py \
  --solver OSQP \
  --daily-solver CLARABEL \
  --event-start 0 \
  --n-events 12 \
  --replan-policy materiality \
  --replan-threshold-bps 2.0 \
  --output-prefix artifacts/commitment_aware_dev_200

env PYTHONPATH=. python experiments/rolling_horizon_walkforward.py \
  --solver OSQP \
  --daily-solver OSQP \
  --event-start 0 \
  --n-events 12 \
  --replan-policy proximal \
  --proximal-basis trade \
  --risk-aversion medium \
  --output-prefix artifacts/proximal_rolling_dev_medium_simultaneous

env PYTHONPATH=. python experiments/forecast_error_risk_walkforward.py \
  --solver OSQP \
  --event-start 0 \
  --n-events 12 \
  --risk-aversion medium \
  --output-prefix artifacts/forecast_error_risk_dev

env PYTHONPATH=. python experiments/forecast_error_risk_walkforward.py \
  --solver OSQP \
  --event-start 12 \
  --n-events 12 \
  --risk-aversion medium \
  --output-prefix artifacts/forecast_error_risk_holdout
```

Each prefix produces a trial ledger, paired event deltas, summary, complete
schedules, daily P&L, and a PNG. Rolling-horizon prefixes additionally record
every forecast vintage, accepted/rejected replan, active and candidate risk
coefficient, complete factor exposure path, and every acceptance gate. The
hybrid confirmation prefixes are
`alpha_confidence_hybrid_dev_*` and `uncertainty_budget_hybrid_dev_*`.
`artifacts/walkforward_research_ledger.csv` is the compact cross-trial index of
25 recorded development/holdout trials, their comparable deltas, decisions,
reasons, and source artifact prefixes.

## Production completion gate

Populate `PointInTimeRebalanceEvent` objects from actual rebalance prediction
snapshots and realized close/VWAP, fills, spread, impact, FX, financing, and
borrow data. Then compare the current automatic high/medium/low policy against
the desk baseline across enough independent events for stable tail estimates.
Until that replay exists, synthetic expected P&L and `economically_viable` are
model outputs, not proof that the strategy makes money.
