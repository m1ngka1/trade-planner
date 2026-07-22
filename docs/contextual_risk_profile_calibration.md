# Contextual High/Medium/Low policy calibration

## Status

Predeclared on 2026-07-23 before implementation or results were generated.

This experiment improves automatic coefficient selection mechanics. It cannot
establish real profitability because no frozen real development bundle is
available locally. The only local historical bundle is the two-event plumbing
fixture, which is too small to fit a contextual model.

## Investment hypothesis

The existing chronological selector estimates one unconditional P&L and risk
distribution for each complete optimizer policy. That is deliberately safer
than hand-entered coefficients, but it assumes the same policy is appropriate
for every rebalance. The historical evidence already shows that the
risk-scaled challenger wins on some events and loses on others.

An investment desk should be more aggressive when point-in-time alpha is
strong relative to cost or execution capacity is tight, and more defensive
when alpha is uncertain, the basket is factor-concentrated, or forecast
liquidity is expected to improve sharply near the event. The user should still
choose only High, Medium, or Low risk aversion. The planner should infer the
complete coefficient vector from the current event and prior realized policy
outcomes.

## Point-in-time event features

The contextual selector accepts five dimensionless features available at the
event's `information_cutoff`:

1. `capacity_pressure`: parent shares relative to total executable capacity;
2. `alpha_cost_ratio`: forecast directional holding alpha relative to forecast
   impact and linear cost;
3. `liquidity_ramp_strength`: forecast late-horizon ADV divided by early ADV;
4. `factor_concentration`: target country/sector/industry exposure relative to
   parent gross; and
5. `forecast_uncertainty_ratio`: forecast-alpha uncertainty relative to the
   absolute forecast-alpha opportunity.

Features may influence the chosen policy but never set a start date, daily
percentage, or target volume curve. Once selected, the complete policy vector
still enters the same optimizer and all liquidity, participation, direction,
completion, and behavior audits remain unchanged.

## Chronological model

For current event `j`, training event `i` is eligible only when:

```text
i appears earlier than j
i.realized_available_at <= j.information_cutoff
```

The first 12 eligible events use the existing fixed High/Medium/Low fallback.
Thereafter:

- standardize features using training events only;
- fit one ridge P&L model per complete policy using a shared ridge multiplier;
- select the multiplier from the fixed grid `0.01, 0.1, 1, 10, 100` by nested
  leave-one-event-out prediction error on eligible history only;
- estimate a current-event P&L prediction interval from training residuals and
  ridge leverage;
- exclude policies with any historical hard failure or less than 95% behavior
  pass rate;
- require the profile-specific predicted P&L lower confidence bound to be
  non-negative when a profitable policy exists; and
- apply the existing High/Medium/Low realized-risk budget, one-basis-point
  materiality tie, and monotone `High <= Medium <= Low` ordering.

Current-event and future outcomes are attached only after selection for
out-of-sample scoring. The model must fall back to the unconditional selector
when feature history is missing, non-finite, or insufficient; it may never
silently use current realized outcomes.

## Frozen controlled mechanics

- 48 chronological events; first 12 are fallback warm-up and 36 are scored out
  of sample.
- Seven complete policies at aggressiveness
  `0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00`.
- Five pre-event features drawn once from seed `20260725`.
- Policy P&L has a known feature-dependent opportunity component, plus common
  event shocks and policy-specific noise.
- Aggressiveness 0.35 has periodic hard failures and 0.65 has periodic behavior
  failures despite attractive P&L; both must remain ineligible.
- Comparator: the existing unconditional chronological selector on the exact
  same event-policy panel.
- Protected surfaces: policy outcomes, feature values, risk budgets, fallback
  map, confidence levels, scoring, and current/future leakage test cannot be
  changed after the first run.

## Acceptance gates

Keep the contextual infrastructure only if all of the following pass:

1. Perturbing current and future outcomes cannot change the current policy,
   training-event list, feature scaling, ridge multiplier, prediction, or P&L
   lower bound.
2. The first 12 events use only the existing fallback policies.
3. Hard-failing and behavior-failing policies are never selected after warm-up.
4. Every event preserves `High <= Medium <= Low` aggressiveness.
5. Across Medium and Low together, contextual mean P&L improves by at least
   0.50 bp/event versus the unconditional selector.
6. No individual profile loses more than 0.25 bp/event versus unconditional.
7. Contextual P&L volatility is no more than 0.50 bp above unconditional for
   every profile.
8. Every profile has positive mean selected P&L and 100% hard-pass rate.
9. For Low risk aversion, high alpha/capacity opportunity selects at least
   0.15 more aggressiveness on average than low opportunity, proving that the
   event features affect the policy rather than merely reproducing a global
   lookup.
10. The complete repository test suite passes.

Passing proves chronology, conditioning, and investment semantics only. It
authorizes an opt-in path in the historical policy-panel runner, while the
existing unconditional selector remains the default until a frozen real
development replay and untouched holdout pass every economic gate.

## Required evidence

Write the controlled events/features, full policy panel, contextual and
unconditional selections, model evaluations, summaries, gates, and a
reviewable visualization showing:

- cumulative P&L improvement versus unconditional by profile;
- contextual policy aggressiveness through time;
- policy choice versus alpha/capacity opportunity;
- selected P&L and volatility; and
- the ridge multiplier and prediction uncertainty through time.

## Result

The fixed 48-event controlled panel completed. Twelve events used the fallback
map and 36 events were scored out of sample. Current/future outcome
perturbations left the probe event's policies, training IDs, feature values,
ridge multiplier, predictions, prediction errors, and confidence bounds
unchanged. Unsafe policies were excluded and `High <= Medium <= Low` held for
every event.

| Profile | Contextual P&L | Unconditional P&L | P&L change | Contextual volatility | Unconditional volatility | Volatility change |
|---|---:|---:|---:|---:|---:|---:|
| High | 1.75 bp/event | 1.75 bp/event | 0.00 | 0.37 bp | 0.37 bp | 0.00 |
| Medium | 2.59 bp/event | 2.12 bp/event | +0.46 | 1.03 bp | 0.39 bp | +0.63 |
| Low | 3.17 bp/event | 2.18 bp/event | +0.99 | 1.88 bp | 0.54 bp | +1.35 |

The average Medium/Low P&L improvement was 0.73 bp/event, above the fixed
0.50-bp gate. Low aggressiveness was 0.54 higher in high-opportunity events
than low-opportunity events, proving the selector used the features. Every
selected policy was profitable on average and passed the controlled hard
audit.

The volatility gate failed. Context conditioning monetized the opportunity
signal by choosing more aggressive policies precisely when opportunity was
high, but the resulting event P&L dispersion rose materially. Medium exceeded
its unconditional volatility by 0.63 bp and Low by 1.35 bp, both above the
predeclared 0.50-bp maximum.

## Decision

**Discard this P&L-only contextual selector.** It demonstrates a leakage-safe
way to improve policy timing and P&L, but it does not satisfy the user's joint
money-making and swing-reduction objective. The code and artifacts are kept to
make the negative result reproducible; the unconditional selector remains the
historical and production default, and no real holdout is authorized.

The evidence suggests a distinct next hypothesis: condition both expected P&L
and downside risk on event features, then select on a profile-specific
certainty equivalent rather than contextual P&L alone. That must be a new,
separately predeclared experiment; this trial's gates and result will not be
changed.

Reproduce with:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/contextual_risk_profile_walkforward.py \
  --full-suite-verified \
  --output-prefix artifacts/contextual_risk_profile_mechanics
```

Artifacts are stored at `artifacts/contextual_risk_profile_mechanics*`.
