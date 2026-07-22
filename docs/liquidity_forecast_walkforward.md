# Point-in-time event-liquidity forecast experiment

## Predeclared hypothesis

The strongest result in the economic fixture came from a date-varying forecast
of rebalance liquidity, not from a larger risk penalty. If expected ADV and
market impact improve near the event, the optimizer should naturally reserve
more flow for those dates, reduce premature inventory, and pay less impact. No
daily trade amount or volume curve is constrained directly.

Test the current flat-ADV planner against one candidate that receives a
point-in-time lower-quantile ADV forecast estimated from disjoint synthetic
history. Both strategies retain identical orders, alpha, covariance, TCA,
participation rules, hard completion, and medium risk aversion.

## Fresh cohort and protected surfaces

- Use 24 new event, residual-scenario, and liquidity seeds that do not overlap
  the spent forecast-risk events 1-24.
- Fix the liquidity population and high/medium/low mapping before running any
  event. Do not change realized returns, realized costs, scoring, or gates after
  inspecting results.
- Events 25-36 are development; events 37-48 are a sealed holdout.
- Open the holdout only if every development gate passes without tuning.
- Latent liquidity and realized ADV stay outside the optimizer context.

## Synthetic liquidity population

The population median daily ADV multiplier is the previously recorded moderate
event-liquidity curve:

```text
0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.80
```

This describes the replay population, not a schedule constraint. Event-common,
date-common, and name-specific log-volume shocks have standard deviations
`0.10`, `0.08`, and `0.10`. A separate 2,000-event calibration history
estimates the date/name log mean and standard deviation.

The user still chooses only a risk label. The planner receives this automatic
liquidity quantile:

| Risk aversion | Forecast ADV quantile | Investment meaning |
|---|---:|---|
| High | 10th percentile | Protect completion and participation against weak liquidity |
| Medium | 25th percentile | Use likely liquidity while retaining a downside buffer |
| Low | 50th percentile | Use median liquidity and accept more forecast risk |

The forecast ADV drives both physical capacity and the dollar impact curve.
Existing economic calibration continues to set covariance, tail-risk, alpha,
and cost coefficients.

## Realized-liquidity scoring

`PointInTimeRebalanceEvent` must keep realized ADV outside `ctx`, just like
realized returns and costs. Realized impact is evaluated using actual ADV, and
the replay records maximum and 95th-percentile actual participation plus any
excess over the forecast-time participation policy. Existing events without
realized ADV retain their current behavior exactly.

## Acceptance gates

The candidate must pass all existing P&L, volatility, loss-CVaR, drawdown,
urgent-start, small-start, factor-balance, ramp, direction, forecast-cap, and
completion gates. It must also:

- reduce realized impact cost;
- not increase mean event-level 95th-percentile realized participation by more
  than 0.5 percentage point; and
- not increase maximum realized participation by more than 1 percentage point
  versus the flat-ADV baseline.

## Result

Both development candidates produced the desired optimizer-derived rising
volume curve and materially reduced P&L swing, downside, drawdown, and realized
impact. Neither preserved enough realized P&L, and both caused at least one
small order to start earlier than the flat-ADV baseline.

| Medium-risk forecast | P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Predeclared 25th percentile | -$1.333m | -9.182 bp | -15.935 bp | -7.521 bp | -$391k | +0.712 pp | +1.595 | Discard |
| One median sensitivity | -$2.508m | -9.949 bp | -13.648 bp | -7.587 bp | -$334k | -0.068 pp | +2.147 | Discard |

The 25th-percentile candidate earned 5.26 bp per event versus 7.54 bp for the
baseline, outside the one-basis-point preservation band. The median candidate
earned only 3.25 bp. Their minimum paired small-order start differences were
both `-7` days. All volatility, loss-CVaR, drawdown, urgent-start, factor,
volume-shape, participation, direction, completion, and added realized-
liquidity gates passed.

The median run was the single post-failure sensitivity permitted on
development data. It tested whether the downside quantile, rather than the
event-liquidity shape, caused the P&L sacrifice. Its larger P&L loss rejects
that explanation and leaves the production high/medium/low quantile map
unchanged. Events 37-48 remain sealed and were not run.

The six-panel plots and full ledgers are stored under
`artifacts/liquidity_forecast_dev*` and
`artifacts/liquidity_forecast_dev_q50*`.

## Follow-up hypothesis: invariant investment risk price

Inspection of the development frontier found a second mechanism: changing ADV
also caused the discrete frontier selector to change the medium-risk
coefficient in four of twelve 25th-percentile events. The two largest
small-order timing violations occurred where the selected coefficient fell to
roughly one third of the baseline value. Liquidity information should change
capacity and expected impact, not silently redefine the desk's risk appetite.

Test one new candidate on the same development set:

1. automatically select the event's medium coefficient from the existing
   investment-risk frontier;
2. reuse that exact coefficient when solving with the predeclared 25th-
   percentile ADV forecast; and
3. leave every objective term, order, return forecast, TCA surface, constraint,
   gate, and event seed unchanged.

This is not a numerical user input or a hard-coded schedule. The user still
chooses only high, medium, or low, and the optimizer still determines every
trade. It isolates the economic price of risk from a point-in-time liquidity
forecast. No quantile sweep is permitted. The existing development gates
control whether the still-sealed events 37-48 may be opened.

The invariant-price candidate passed 15 of 16 development gates:

| P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -$580k | -0.992 bp/event | -10.069 bp | -15.935 bp | -8.442 bp | -$394k | +0.205 pp | +1.596 | Discard |

Locking the economic risk price recovered about $752k relative to the original
25th-percentile candidate and brought realized P&L just inside the predeclared
one-basis-point band. It also improved every swing, downside, volume-shape,
liquidity, and hard-execution gate. The only failure was
`small_never_earlier`: three events moved the first nonzero small-name trade one
day earlier, often through an economically tiny opening trade. The strict gate
is preserved rather than redefined after observing the result. The sealed
holdout therefore remains unopened.

Artifacts: `artifacts/liquidity_forecast_locked_dev*`.

## Next hypothesis: capacity-slack option value

The near-miss shows that risk-price stability solves the large discontinuity,
but a continuous QP can still leak a small amount into non-urgent names when
forecast alpha or factor hedging makes that marginal trade attractive. A
small order with ample remaining capacity has more option value in waiting for
new information than a large order that must use most of its capacity.

Predeclare one combination candidate:

- retain the 25th-percentile medium liquidity forecast and invariant automatic
  risk price;
- compute each name's capacity utilization from target shares divided by total
  point-in-time forecast capacity;
- apply the existing point-in-time expected-return standard error only to the
  unused-capacity fraction, so the alpha hurdle is strongest for small names
  and fades automatically for urgent names; and
- derive the one-sided confidence from the existing risk-profile fraction:
  `confidence = 1 - 0.5 * risk_frontier_fraction`, giving 97.5%, 75%, and 50%
  for high, medium, and low respectively.

This produces an expected-P&L-dollar objective term, not a fixed start day,
minimum trade size, post-processing clip, or new free coefficient. First run a
two-event mechanical screen because event 26 already contains the timing
failure. Run the full development set only if the screen preserves urgent
timing, hard feasibility, and removes the small-order regression. Do not open
events 37-48 unless every full-development gate passes.

The two-event mechanical screen removed event 26's small-order regression,
kept urgent starts unchanged, and passed every hard constraint, so the full
development run was permitted. The full result was:

| P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -$872k | -1.491 bp/event | -12.300 bp | -25.138 bp | -9.426 bp | -$436k | +1.349 pp | +2.101 | +1.5 days | Discard |

The term completely fixed `small_never_earlier` and strengthened every swing,
downside, cost, and volume-shape result. It failed P&L preservation and exceeded
the early-factor tolerance by 0.349 percentage point. Event-level exposure
inspection shows that delaying optional small flow left IT and financials as
the most frequent worst early factors, with country US worst in two events.
This is evidence that small orders had also been serving as portfolio hedges.
The result is discarded and the holdout remains sealed.

Artifacts: `artifacts/liquidity_option_value_smoke*` and
`artifacts/liquidity_option_value_dev*`.

## Next hypothesis: equal-factor correlation-break stress

The standard Barra objective already prices factor covariance, but the stated
execution goal is stricter: country, sector, and industry dollar exposures
should remain balanced even when their estimated correlations or relative
volatilities make one hedge look cheap. Add one correlation-break overlay to
the option-value candidate:

```text
factor stress risk
    = stress fraction * median supplied factor variance
      * sum(square(country/sector/industry dollar exposure))
```

The stress fraction is `1 - risk_frontier_fraction`, so high, medium, and low
profiles use 95%, 50%, and 0% respectively. The supplied covariance determines
the variance scale, and the already selected inventory-risk price multiplies
the total dollar-squared risk. Thus the user still supplies only a risk label;
there is no balance coefficient, exposure target, or daily trade schedule.

Screen events 28-29 first because event 28 had the largest factor regression.
Proceed to all development events only if the screen keeps urgent starts and
hard constraints intact, keeps small names no earlier, and holds early-factor
imbalance within one percentage point of baseline. No parameter sweep is
permitted. Events 37-48 remain sealed unless every full-development gate
passes.

The fixed events 28-29 screen solved and preserved P&L, volatility, loss-CVaR,
drawdown, urgent timing, small timing, the volume ramp, liquidity, and every
hard constraint. Mean early factor imbalance was 7.056% versus 6.018% for the
baseline: a `+1.038 pp` change, missing the one-percentage-point gate by 0.038
pp. The result is discarded at the mechanical screen; its stress strength is
not tuned and the full development set is not run.

Artifacts: `artifacts/liquidity_factor_stress_smoke*`.

## Next hypothesis: minimax factor stress

The equal-factor L2 overlay spreads its charge across every exposure, while
the actual desk objective and gate are dominated by the single worst country,
sector, or industry imbalance. Test one norm change, not a coefficient change:
replace the equal-factor sum of squares with

```text
factor_count * square(max(abs(factor dollar exposure)))
```

This is the dollar-variance stress if the largest current exposure represents
the basket's factor concentration and all factor slots are budgeted against
that worst exposure. Keep the same median factor variance, automatic
high/medium/low stress fraction, baseline-locked risk price, liquidity
forecast, option-value term, events, and gates. Screen events 28-29 once. Run
full development only if all mechanical screen criteria pass; no multiplier
or norm sweep is permitted.

The minimax screen passed every gate. Full development then passed all 16
predeclared economics, behavior, liquidity, and hard-execution gates without
tuning, which authorized the first and only opening of events 37-48.

| Split | P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Focused screen, events 28-29 | +$674k | +6.920 bp/event | -4.483 bp | -10.090 bp | -16.445 bp | -$78k | -1.011 pp | +1.489 | +1.0 day | Pass to development |
| Development, events 25-36 | -$480k | -0.822 bp/event | -15.741 bp | -26.925 bp | -10.974 bp | -$410k | -1.534 pp | +1.922 | +2.67 days | Pass to holdout |
| Untouched holdout, events 37-48 | -$1.041m | -1.780 bp/event | -14.031 bp | -23.316 bp | -10.717 bp | -$504k | -1.715 pp | +1.663 | +2.58 days | Fail promotion |

The holdout confirms that the combined objectives create the desired optimizer-
derived execution shape and materially reduce swing, downside, cost, and early
factor imbalance while keeping urgent names on day one and delaying small
orders. It rejects profitability promotion: mean P&L fell 1.780 bp per event,
outside the one-basis-point band. Event 45 also contained the only numerical
hard-audit misses, a 0.128-share cap excess and a 0.00121-share wrong-direction
residual under OSQP. Completion remained exact. Those sub-share issues do not
change the economic rejection.

The candidate is retained as the strongest synthetic research model, not a
production default. Events 37-48 are now spent and must not be tuned or reused
for promotion. Artifacts are stored under
`artifacts/liquidity_minimax_factor_smoke*`,
`artifacts/liquidity_minimax_factor_dev*`, and
`artifacts/liquidity_minimax_factor_holdout*`.

## Next hypothesis: automatic forecast-profit floor

The minimax candidate's holdout forecast P&L was already $1.147m below the
baseline, and realized P&L was $1.041m lower. The risk model is accepting a
known alpha sacrifice to buy lower swing. Preserve its successful risk and
behavior terms, but add a convex expected-net-P&L floor:

```text
candidate forecast net P&L
    >= baseline forecast net P&L - 1 bp of parent gross
```

One basis point is the desk materiality unit already used by the automatic
frontier selector and the realized promotion gate, not a new coefficient. The
constraint includes raw point-in-time expected holding alpha, forecast impact,
and linear cost. It limits economic sacrifice without specifying any symbol's
trade date or daily volume.

Use a completely fresh cohort and fixed seeds: events 49-60 are validation and
events 61-72 are sealed holdout. Do not reuse events 37-48. Use CLARABEL for
both baseline and candidate because the minimax epigraph is conic and the spent
OSQP holdout exposed sub-share residuals. The same 16 gates apply. Open events
61-72 once only if all validation gates pass; no floor or solver sweep is
permitted. First screen events 49-50 only for solve status, profit-floor
satisfaction, urgent timing, and hard feasibility; full validation remains the
economic decision.

The event-49 mechanical screen crashed before applying the profit floor.
CLARABEL solved only the zero-risk point of the flat-ADV frontier and declared
every forecast-liquidity frontier point infeasible. This is a solver/scaling
failure rather than an economic result, but the predeclared no-solver-sweep rule
is preserved: events 49-72 will not be retried with OSQP and produce no
promotion evidence.

Before allocating more fresh data, screen the profit-floor constraint with the
already-spent development events 25-26 and the previously validated OSQP
formulation. The screen must solve, satisfy the floor, preserve urgent timing,
and pass all hard constraints. If it passes, use an entirely new OSQP cohort:
events 73-84 validation and events 85-96 sealed holdout. Events 49-72 remain
abandoned rather than recycled. No floor or solver sweep is permitted within
the new cohort.

Failure record: `artifacts/profit_floor_smoke_failure.txt`.

The OSQP formulation screen also reported the net-P&L constraint infeasible on
event 25. This is demonstrably a numerical false infeasibility: the already
solved event-25 minimax schedule has forecast net P&L of `-$220,907` versus a
floor of `-$272,029`, clearing the inequality by `$51,122`. Scaling the exact
same constraint by one-basis-point materiality did not fix the backend result.
The quadratic net-P&L-floor formulation is therefore rejected as numerically
unreliable, and no new cohort is allocated to it.

The realized holdout loss came from `$1.558m` less gross holding P&L, partially
offset by `$504k` lower impact. Test a linear holding-alpha floor next:

```text
candidate forecast holding alpha
    >= baseline forecast holding alpha - 1 bp of parent gross
```

Impact and linear cost remain in the optimizer objective and realized gates;
the new constraint targets the observed alpha sacrifice without a quadratic
cost epigraph. Screen it with spent events 25-26 under OSQP. Allocate fresh
events 73-96 only if it solves, satisfies the alpha floor, preserves urgency,
and passes hard feasibility. No threshold sweep is permitted.

Failure records: `artifacts/profit_floor_smoke_failure.txt` and
`artifacts/profit_floor_osqp_mechanics_failure.txt`.

The linear holding-alpha floor solved on events 25-26, but it did not satisfy
its own contract: event 26 finished `$21.5k` below the floor despite an
`optimal` solver status. Event 25 also had a 0.180-share cap excess and a
0.00156-share wrong-direction residual, and the two-event volatility was 7.225
bp higher. The formulation is discarded and no new cohort is allocated.

Next test a numerically stable plan-selection gate rather than another solver
constraint. Solve both complete optimizer models as already validated:

1. the current flat-ADV automatic medium plan; and
2. the forecast-liquidity, invariant-price, capacity-option, minimax-factor
   plan.

Select the forecast-liquidity plan only when its raw point-in-time forecast net
P&L is no worse than the baseline by one basis point of parent gross and its
participation, direction, and completion audit passes. Otherwise select the
baseline plan. Both outcomes are optimizer-derived complete schedules; the
gate contains no start date, symbol target, trade amount, or fitted coefficient.

Screen this fixed policy on spent events 25-36. It must pass all 16 gates before
allocating a fresh events 73-96 cohort. The spent events may validate mechanics
but cannot support promotion. No materiality or audit-tolerance sweep is
permitted.

Artifacts: `artifacts/alpha_floor_osqp_mechanics*`.

The plan-selection gate chose the minimax schedule in only three of twelve
spent development events and fell back to the baseline in nine. It nevertheless
lost `$856k` or `1.464 bp/event`, while volatility rose `0.063 bp`; forecast
P&L did not identify the realized winning events well enough. It is discarded
and no fresh cohort is allocated.

The next candidate keeps one optimizer solve rather than selecting or blending
completed plans. Shrink the point-in-time event-liquidity shape toward flat
liquidity using the already defined desk risk-budget fraction:

```text
log(planning ADV / flat ADV)
    = (1 - risk_frontier_fraction)
      * log(lower-quantile forecast ADV / flat ADV)
```

High, medium, and low profiles therefore use 95%, 50%, and 0% of the event
shape automatically. Quantiles remain 10%, 25%, and 50%. Medium retains a
rising liquidity curve but reduces the alpha-timing sacrifice of the full
shape. Keep the invariant risk price, capacity-option alpha, minimax factor
stress, orders, costs, and all gates unchanged. No schedule is blended or
post-processed.

Screen the fixed medium policy on spent events 25-36. Allocate fresh events
73-96 only if all 16 gates pass. No shape-weight sweep is permitted.

Artifacts: `artifacts/plan_selection_gate_dev*`.

The fixed medium risk-scaled shape passed all 16 gates on spent events 25-36:

| P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +$67k | +0.115 bp/event | -10.706 bp | -22.617 bp | -7.108 bp | -$264k | -1.384 pp | +0.766 | +3.0 days |

Urgent names remained on day one and all numerical hard audits passed. Because
these development events informed earlier model choices, this is a mechanics
and direction screen only, not promotion evidence. Freeze the policy and run a
completely fresh cohort with new event, residual-scenario, and realized-
liquidity seeds: events 73-84 validation and events 85-96 sealed holdout. Use
OSQP for both strategies, the same 16 gates, and no shape, quantile, coefficient,
or solver changes. Open events 85-96 once only if all fresh-validation gates
pass.

Artifacts: `artifacts/risk_scaled_liquidity_dev*`.

Fresh events 73-84 passed all 16 gates with the policy frozen, authorizing the
first and only opening of events 85-96. The untouched holdout rejected
promotion:

| Split | P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Fresh validation, events 73-84 | -$167k | -0.286 bp/event | -5.494 bp | -16.079 bp | -4.994 bp | -$293k | -1.323 pp | +0.700 | +2.50 days | Pass to holdout |
| Untouched holdout, events 85-96 | -$1.896m | -3.242 bp/event | -8.081 bp | -14.754 bp | -3.651 bp | -$304k | -1.506 pp | +0.733 | +1.33 days | Fail promotion |

The risk-scaled shape again produces the desired optimizer-derived rising
volume, keeps urgent names on day one, delays small names, lowers early factor
imbalance, and reduces every swing/downside/cost metric. It does not preserve
realized profitability. Events 91 and 92 contributed `-27.92` and `-13.94` bp
of paired P&L respectively, showing that timing-alpha forecast error remains
the dominant failure. Event 93 had the only hard-audit miss, a 0.087-share cap
excess; direction and completion passed.

The holdout is spent and the model is not promoted. More synthetic penalty or
shape tuning is now lower priority than estimating point-in-time alpha decay
and liquidity jointly from real rebalance histories. The next research cycle
should consume stored predictions, fills, close/VWAP, realized ADV, TCA, FX,
borrow, and classifications, then calibrate the High/Medium/Low policy on
chronologically disjoint events.

Artifacts: `artifacts/risk_scaled_liquidity_fresh_dev*` and
`artifacts/risk_scaled_liquidity_fresh_holdout*`.

Status: fresh validation passed; untouched holdout failed P&L and one numerical
cap audit; no production promotion.

## Next hypothesis: raw-alpha opportunity-cost insurance

Post-failure analysis uses only the now-spent events 25-36 and 73-96.  The
forecast-liquidity plan saved realized impact in every cohort, but its paired
P&L had little linear relationship with observable event-level forecast
features: correlation was `0.13` with target-weighted directional forecast
alpha, `0.14` with median alpha signal-to-noise, and `-0.10` with the
candidate-versus-baseline forecast-net-P&L difference.  A forecast-based
completed-plan selector is therefore still unsupported.

The current challenger also applies a confidence haircut to optional alpha at
the same time that the risk-scaled liquidity curve and minimax factor risk
encourage waiting.  That double conservatism may over-insure P&L swing.  Test
the smallest attributable combination change: retain the baseline-locked risk
price, 50% risk-scaled liquidity shape, 50% minimax factor stress, orders,
costs, liquidity limits, and all 16 gates, but restore the raw point-in-time
expected-return objective.  This remains one optimizer solve and does not
prescribe any symbol's start date or daily volume.

First run the fixed policy on spent events 25-36 as a mechanics and direction
screen.  Proceed only if all 16 gates pass, including small names never
starting earlier.  If it passes, freeze the policy and allocate completely new
event, residual-scenario, and realized-liquidity seeds to events 97-108 for
development and events 109-120 for sealed holdout.  Open the holdout once only
if development passes every gate.  No alpha coefficient, shape fraction,
factor multiplier, solver, or gate threshold sweep is permitted.

The spent-event screen passed all 16 gates:

| P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +$284k | +0.485 bp/event | -8.262 bp | -14.817 bp | -5.834 bp | -$209k | -2.129 pp | +0.519 | +0.33 day |

Urgent names remained on day one and every hard audit passed.  The result is
stronger than the confidence-haircut policy on these already-spent events, so
the raw-alpha policy is now frozen for the predeclared fresh cohort.  This
screen is not promotion evidence.

Artifacts: `artifacts/risk_scaled_raw_alpha_spent*`.

Fresh events 97-108 rejected the raw-alpha policy before holdout:

| P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -$1.690m | -2.890 bp/event | -2.810 bp | +1.876 bp | -4.356 bp | -$269k | -2.124 pp | +0.485 | 0.0 day | Discard |

The optimizer still created the requested exposure, urgency, and rising-volume
behavior and reduced volatility, drawdown, factor imbalance, impact, and
participation.  It did not preserve profitability or tail loss.  Events
109-120 remain sealed and will not be reassigned to this failed candidate.

Artifacts: `artifacts/raw_alpha_opportunity_fresh_dev*`.

## Next hypothesis: baseline-relative tail-regret insurance

Raw alpha protects forecast mean opportunity cost, but fresh validation shows
that it does not protect the downside of being late when the alpha or residual
path is unexpectedly favorable.  Add a scenario objective for the candidate's
holding-P&L regret relative to the already solved flat-ADV optimizer plan:

```text
relative P&L(scenario)
    = sum over days of
      (candidate inventory - baseline optimizer inventory)
      * price * (point-in-time expected return + centered residual scenario)

regret objective
    = automatic regret weight * CVaR95(-relative P&L)
```

The baseline is an optimizer-derived schedule, not a hard-coded start date or
volume curve.  The candidate remains free to depart from it wherever saved
impact, factor balance, and inventory-risk reduction pay for the scenario
regret.  The dimensionless regret weight is
`1 - risk_frontier_fraction`, giving High/Medium/Low values of 95%, 50%, and
0%.  A dollar of forecast tail regret is therefore priced directly against
the existing dollar alpha and cost terms, with no manually entered
coefficient.  The 95% confidence matches the desk's existing loss-CVaR gate.

Screen this one fixed formulation on the now-spent events 97-108 using raw
alpha, baseline-locked inventory risk, 50% risk-scaled liquidity, and 50%
minimax factor stress.  Proceed only if all 16 gates pass.  If it passes,
freeze the model and allocate new events 121-132 for validation and events
133-144 for sealed holdout; events 109-120 remain unused.  Do not sweep regret
weight, confidence, scenarios, shape, solver, or gates.

The fixed screen crashed with OSQP status `user_limit`.  Initializing the CVaR
threshold so the planner could include it in reference-objective normalization
did not change the result on the one permitted implementation retry.  No
complete event comparison or visualization was produced.  The formulation is
discarded without a solver, confidence, weight, or scenario-count sweep, and
events 121-144 were not allocated to it.

Failure record: `artifacts/baseline_regret_spent_failure.txt`.

## Next hypothesis: baseline-relative tracking-risk insurance

Retain the investment idea but replace the numerically unreliable CVaR hinge
formulation with a pure quadratic relative tracking-risk objective:

```text
relative residual P&L(scenario)
    = sum over days of
      (candidate inventory - baseline optimizer inventory)
      * price * centered residual scenario

tracking-risk objective
    = automatic relative-risk price
      * weighted mean(square(relative residual P&L))
```

Raw expected alpha continues to price the mean opportunity cost, so the new
term uses centered residual scenarios only and cannot become a hidden alpha
forecast.  Its coefficient is the automatically selected baseline inventory-
risk price multiplied by `1 - risk_frontier_fraction`; High/Medium/Low thus
charge 95%, 50%, and 0% of the desk's existing dollar-variance price.  This
keeps units consistent and introduces no manual number.

Screen the fixed quadratic formulation on spent events 97-108 with the same
raw alpha, baseline-locked inventory risk, risk-scaled liquidity, minimax
factor stress, OSQP solver, and 16 gates.  If and only if all gates pass,
allocate new seeds to events 121-132 and sealed events 133-144.  Do not reuse
events 109-120 or sweep any coefficient, scenario set, solver, or gate.

The quadratic formulation solved cleanly and improved the failed raw-alpha
candidate, but it did not clear the two decisive gates:

| P&L delta | Mean P&L delta | Volatility delta | Loss-CVaR delta | Mean DD delta | Impact delta | Factor delta | Ramp delta | Small-start delta | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -$1.482m | -2.535 bp/event | -2.977 bp | +0.532 bp | -4.213 bp | -$266k | -1.714 pp | +0.468 | 0.0 day | Discard |

Relative tracking risk reduced the raw-alpha candidate's P&L sacrifice by
`0.355 bp/event` and narrowed its loss-CVaR regression from `+1.876` to
`+0.532 bp`, while preserving urgency, small-order timing, factor balance,
rising volume, impact, participation, and hard feasibility.  That is not enough
to justify a coefficient sweep or new data.  Events 121-144 remain unallocated.

Artifacts: `artifacts/baseline_tracking_risk_spent*`.

## Reproduce

```bash
env PYTHONPATH=. python experiments/liquidity_forecast_walkforward.py \
  --solver OSQP --event-start 0 --n-events 12 \
  --risk-aversion medium \
  --output-prefix artifacts/liquidity_forecast_dev

env PYTHONPATH=. python experiments/liquidity_forecast_walkforward.py \
  --solver OSQP --event-start 0 --n-events 12 \
  --risk-aversion medium --liquidity-quantile 0.50 \
  --output-prefix artifacts/liquidity_forecast_dev_q50

env PYTHONPATH=. python experiments/liquidity_forecast_walkforward.py \
  --solver OSQP --event-start 0 --n-events 12 \
  --risk-aversion medium --coefficient-policy baseline_locked \
  --output-prefix artifacts/liquidity_forecast_locked_dev

env PYTHONPATH=. python experiments/liquidity_forecast_walkforward.py \
  --solver OSQP --event-start 0 --n-events 12 \
  --risk-aversion medium --coefficient-policy baseline_locked \
  --alpha-policy capacity_slack_confidence \
  --output-prefix artifacts/liquidity_option_value_dev
```
