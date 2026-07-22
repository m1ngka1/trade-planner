# Automatic investment-policy calibration for High/Medium/Low

## Investment hypothesis

The optimizer already selects the inventory-risk price from an economic
frontier, but liquidity quantile, event-liquidity shape, alpha confidence, and
factor stress still use a permanent fixed mapping. Synthetic holdouts showed
that stable coefficients do not guarantee profit when the holding-alpha path
is wrong. Once point-in-time alpha is calibrated, the remaining policy vector
should be selected from earlier realized net P&L and downside—not entered by a
user or fitted to the current event.

Use one predeclared monotone policy ladder. Every row has a scalar
`policy_aggressiveness` between zero and one and the complete coefficient
vector:

- feasible frontier fraction;
- liquidity forecast quantile;
- event-liquidity shape fraction;
- optional-alpha confidence; and
- factor-stress fraction.

The user still chooses only High, Medium, or Low. The selector chooses a whole
coherent vector; it never combines the best coefficient from different
policies and never inserts a start date or daily trade percentage.

## Chronological contract

For current event `j`, policy outcomes from event `i` are usable only when:

```text
i appears earlier in events.csv
i.realized_available_at <= j.information_cutoff
```

The current event's result and every future result are attached for scoring
only after the policy has been selected. The first eight eligible events use a
predeclared fallback mapping nearest to aggressiveness 0.05, 0.50, and 1.00 for
High, Medium, and Low. Development and holdout bundles remain physically
separate.

## Investment-first selection rule

For each candidate on eligible history, compute equal-event statistics:

- mean realized net P&L in basis points and its standard error;
- one-sided net-P&L lower confidence bound;
- event-level P&L volatility;
- empirical 95% loss CVaR;
- mean within-event drawdown; and
- hard-constraint and behavior pass rates.

High, Medium, and Low use the existing 97.5%, 75%, and 50% one-sided
confidence semantics. A policy is investment-eligible only when every hard
audit passed, at least 95% of behavior audits passed, and its net-P&L lower
confidence bound is non-negative. If no policy clears the profit condition,
the selector must still return an executable low-risk fallback but mark it
`no_profitable_policy`; it may not describe the result as economically viable.

Define one coefficient-free realized-risk measure in basis points:

```text
max(event P&L volatility, positive loss CVaR, mean within-event drawdown)
```

Within the investment-eligible set, High, Medium, and Low allow 5%, 50%, and
100% of the observed risk range. Inside that budget, choose the highest P&L
lower bound. Candidates within one basis point are economically tied, so the
lower realized-risk policy wins. Select profiles in High-to-Low order and
require non-decreasing `policy_aggressiveness`; a less risk-averse profile may
never receive a more conservative vector than the preceding profile.

## Protected surfaces

- Do not change optimizer objectives, constraints, risk labels, realized
  scoring, alpha calibration, liquidity forecasts, TCA, or the existing 16
  promotion gates.
- Do not use the current event to select its own coefficient vector.
- Do not optimize a weighted mathematical score with arbitrary unit-mixing
  coefficients.
- Do not promote a policy from controlled data or open a real holdout before a
  frozen real development replay passes.

## Historical policy-panel contract

The production bridge solves the predeclared aggressiveness ladder
`0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00` for every event using the same
point-in-time alpha calibration, solver, realized scoring, and source hashes.
Each event-policy row separates:

- `hard_pass`: forecast participation-cap excess at most 0.05 shares,
  wrong-direction flow at most 0.001 shares, and terminal error at most 0.001
  shares; and
- `behavior_pass`: urgent flow never later, small flow never earlier, early
  factor imbalance no more than one percentage point worse, late/early ramp at
  least 90% of baseline and at least one, rank ramp within 0.10, non-decreasing
  steps within one, mean P95 realized participation within 0.5 percentage point,
  and maximum realized participation within one percentage point.

After the panel is complete, the chronological selector produces one vector per
event and risk label. The requested profile is converted to an event-policy
mapping and solved again as the selected replay. Candidate outcomes exist for
post-event scoring, but selection for event `j` still sees only earlier events
whose outcomes were available by `j`'s cutoff.

The two-event synthetic historical smoke is plumbing evidence only. Keep the
bridge only if it produces a complete 14-row panel, uses the Medium fallback
policy during warm-up, matches the final replay's coefficient audit to the
selected IDs, preserves source hashes and alpha audit, and passes the full
repository suite.

## Controlled mechanics acceptance criteria

Use a deterministic policy-result panel with profitable efficient policies,
one hard-failing high-P&L policy, and one behavior-failing high-P&L policy. Keep
the calibration infrastructure only if:

1. changing current and future outcomes cannot change the current selection,
   training-event list, or estimated policy statistics;
2. hard-failing and behavior-failing policies are never selected after warm-up;
3. selected aggressiveness is ordered `High <= Medium <= Low` for every event;
4. every calibrated profile has positive mean selected net P&L;
5. no calibrated profile loses more than one basis point per event versus its
   predeclared fallback mapping;
6. the first eight events use only the fallback mapping; and
7. the full repository test suite passes.

Passing this screen proves chronological selection and investment semantics,
not real profitability. The production decision remains a frozen real
development replay followed, if successful, by one sealed historical holdout.

## Implementation

`trade_planner/policy_calibration.py` provides two production-facing pieces:

1. `build_monotone_policy_ladder(...)` converts one aggressiveness coordinate
   into the complete frontier/liquidity/alpha/factor vector; and
2. `calibrate_risk_profiles_walk_forward(...)` selects High, Medium, and Low
   policies from a complete event-policy result panel using only outcomes
   available before each information cutoff.

The output records every training event, candidate mean and standard error,
confidence-adjusted P&L, realized-risk budget, feasibility flag, selection,
coefficient, and current-event realized score. An all-negative history still
returns a feasible schedule policy but marks every result
`calibrated_no_profitable_policy`; this prevents compulsory execution from
being mislabeled as an investment opportunity.

## Controlled result

The fixed 28-event panel used eight fallback events and evaluated the following
20 events out of sample. Two deliberately attractive candidates were excluded:
aggressiveness 0.35 had a hard failure and 0.65 passed fewer than 95% of
behavior audits. All seven predeclared gates passed.

| Profile | Mean selected P&L | P&L volatility | Fixed fallback P&L | Mean selected aggressiveness |
|---|---:|---:|---:|---:|
| High | 1.42 bp/event | 0.24 bp | 1.42 bp/event | 0.05 |
| Medium | 3.32 bp/event | 0.75 bp | 3.32 bp/event | 0.50 |
| Low | 4.48 bp/event | 1.28 bp | 4.15 bp/event | 0.79 |

The Low profile gained 0.33 bp/event versus its fixed fallback while reducing
event P&L volatility from 1.77 to 1.28 bp. High and Medium retained their
fallback economics. Current/future outcome perturbations left the probe
selection and every estimated statistic exactly unchanged; every selected
schedule passed hard and behavior audits; and `High <= Medium <= Low` held for
all 20 events. Repository verification passed 110 tests plus 2 subtests.

The selected row is directly consumable by `run_historical_experiment(...)`
through `InvestmentPolicyCoefficients.from_mapping(...)`. The replay uses its
frontier fraction for the shared inventory-risk price, log-interpolates the
stored P10/P25/P50 ADV forecasts to its liquidity quantile, and applies its
shape, alpha confidence, and factor stress to the challenger solve. Default
calls still use the existing fixed map, so learned policy deployment is always
explicit and auditable.

Decision: **keep the automatic policy-calibration infrastructure for real
development replay**. The controlled data-generating process is known, so this
does not establish real profitability or authorize a production default.

Reproduce the evidence with:

```bash
env PYTHONPATH=. python experiments/automatic_risk_profile_walkforward.py \
  --full-suite-verified \
  --output-prefix artifacts/automatic_risk_profile_mechanics
```

Artifacts include the policy ladder, full controlled trial panel, every
chronological selection, all candidate evaluations, profile summaries, seven
gates, and a four-panel PNG at `artifacts/automatic_risk_profile_mechanics*`.

## Historical replay bridge smoke

`experiments/historical_policy_panel.py` now performs the complete workflow:
calibrate alpha once, solve all seven policy vectors, build event-level
hard/behavior evidence, select profiles chronologically, convert the requested
profile to per-event `InvestmentPolicyCoefficients`, and run the final selected
baseline-versus-challenger replay.

The two-event historical smoke produced the required 14 policy trials. Every
candidate passed cap, direction, and completion audits; policies below 0.50
failed the predeclared behavior composite because they delayed small flow too
aggressively, while policies 0.50 and above passed. All events remained inside
the eight-event warm-up, so Medium selected policy 0.50, and the final replay's
coefficient IDs exactly matched those chronological selections. Source hashes
and the alpha audit were preserved.

The selected smoke challenger was discarded by the existing promotion gates:
it did not lower event volatility, loss CVaR, or impact cost. This is the
correct outcome for a plumbing fixture with two identical realized events and
is not used to change the ladder, selector, or optimizer. The full repository
passed 111 tests plus 2 subtests.

Reproduce both the selected replay and policy-panel visual with:

```bash
env PYTHONPATH=. python experiments/historical_policy_panel.py \
  --bundle artifacts/historical_replay_smoke_input \
  --role development \
  --risk-aversion medium \
  --solver CLARABEL \
  --output-prefix artifacts/historical_policy_smoke
```

The prefix writes the standard selected-replay artifacts plus the policy
events, ladder, trials, selections, candidate evaluations, summaries,
schedules, profiles, exposures, coefficients, frontiers, and a separate
`historical_policy_smoke_policy.png`.
