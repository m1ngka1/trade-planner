# Systematic-first inventory-risk experiment

## Status

Predeclared on 2026-07-23 before implementation or optimizer runs.

This is a spent-development mechanics test, not a fresh validation and not
evidence of real profitability. Synthetic events 109-120 remain sealed.

## Investment hypothesis

The current candidate prices factor risk and specific risk with the same
inventory-risk coefficient. On already-spent events 97-108, specific variance
accounts for about 70% of average accumulated-inventory variance for both the
flat-ADV baseline and the risk-scaled-liquidity challenger. That can make the
optimizer delay a diversified long/short basket even after country, sector,
and industry exposure is balanced. The desk is then giving up rebalance alpha
to avoid specific risk that is substantially diversified across the basket.

The proposed objective keeps the complete Barra factor covariance and the
existing minimax categorical-factor stress, but scales only the specific-risk
component. Strong systematic protection should still encourage balanced early
inventory, while a lower specific-risk price should let profitable,
factor-offsetting names trade earlier. Liquidity, participation, direction,
and completion constraints are unchanged. No start date, daily percentage, or
volume curve is prescribed.

## Automatic investment coefficient

Let `a` be the existing risk-frontier fraction associated with the user's
High/Medium/Low profile. The specific-risk fraction is fixed before the run:

```text
specific_risk_fraction = max(0.25, 1 - a)
```

This gives:

| Risk aversion | Frontier fraction `a` | Specific-risk fraction | Investment meaning |
|---|---:|---:|---|
| High | 0.05 | 0.95 | Keep nearly all single-name risk protection. |
| Medium | 0.50 | 0.50 | Diversify half of specific risk while retaining full factor protection. |
| Low | 1.00 | 0.25 | Prioritize expected P&L, but never treat single-name risk as free. |

There is no coefficient sweep. The 25% floor is a desk guardrail against an
optimizer treating idiosyncratic risk as zero even for the least risk-averse
profile.

## Frozen comparison

- Cohort: already-spent synthetic events 97-108 only.
- Baseline: current flat-ADV, raw-alpha, medium-risk optimizer.
- Control challenger: current risk-scaled liquidity, raw alpha, baseline-locked
  inventory-risk price, and minimax factor stress with 100% specific risk.
- New challenger: identical to the control except the automatic Medium
  specific-risk fraction is 50%.
- Solver, forecasts, event seeds, scenario seeds, realized-liquidity seeds,
  TCA, and evaluation outcomes are identical across the comparison.
- Protected surface: scoring, economic gates, data-generating process, and
  events 109-120 cannot change.

## Acceptance gates

The new challenger is retained as a real-data replay candidate only if all of
the following hold:

1. It passes every existing P&L, volatility, loss-CVaR, drawdown, behavior,
   liquidity, participation, direction, and completion gate versus the
   flat-ADV baseline.
2. Mean realized net P&L improves by at least 0.50 bp per event versus the
   100%-specific-risk control challenger.
3. Event P&L volatility is no more than 1.00 bp above the control and remains
   at least 0.05 bp below the flat-ADV baseline.
4. Mean early factor imbalance is no more than 0.75 percentage point above the
   control and remains within the existing baseline tolerance.
5. The late/early volume ratio retains at least 90% of the control's ratio,
   urgent names never start later than the baseline, small names never start
   earlier than the baseline, and all hard share certificates pass.
6. Realized impact cost remains below the flat-ADV baseline.

A pass authorizes inclusion in the frozen real historical policy panel. It
does not authorize another synthetic holdout or production promotion. A fail
is recorded and the production/history default remains unchanged.

## Required artifacts

The experiment must write trial, paired, summary, gate, schedule,
coefficient, and risk-decomposition CSVs plus a reviewable PNG showing:

- cumulative realized P&L for baseline, control, and challenger;
- P&L recovery versus the control by event;
- volatility, loss-CVaR, and drawdown;
- factor versus specific inventory variance; and
- optimizer-derived volume profiles and early factor balance.

## Result

The fixed 12-event spent-development run completed with OSQP. The automatic
Medium coefficient priced 50% of specific variance and left factor covariance,
minimax stress, forecasts, costs, and hard constraints unchanged.

| Metric | Flat-ADV baseline | 100% specific-risk control | 50% systematic-first |
|---|---:|---:|---:|
| Mean realized net P&L | 16.94 bp/event | 14.05 bp/event | 13.26 bp/event |
| Event P&L volatility | 55.74 bp | 52.93 bp | 54.04 bp |
| Loss-CVaR 95% | 61.88 bp | 63.75 bp | 65.67 bp |
| Mean within-event drawdown | 45.87 bp | 41.52 bp | 43.17 bp |
| Mean early factor imbalance | 5.23% | 3.10% | 2.52% |
| Mean late/early volume ratio | 1.20x | 1.69x | 1.41x |
| Total realized impact cost | $4.06m | $3.79m | $3.98m |

The new objective did improve early factor balance by another 0.59 percentage
point versus the control, and volatility remained 1.71 bp below the flat-ADV
baseline. It did not recover alpha. Mean P&L fell another 0.78 bp/event versus
the control, loss-CVaR worsened by 1.91 bp, volatility worsened by 1.10 bp, the
late/early ramp weakened, and impact savings shrank. OSQP also produced a
0.261-share participation-cap excess, above the fixed 0.05-share hard
certificate. The direction and terminal-completion certificates passed.

## Decision

**Discard.** The evidence contradicts the hypothesis that reducing the
specific-risk price would recover holding alpha while retaining the existing
risk improvement. It concentrated the optimizer more tightly around factor
neutrality but did not select more profitable timing. The default Barra risk
model, automatic policy ladder, and historical replay remain unchanged.

No solver retry, coefficient sweep, fresh synthetic cohort, or sealed event was
used. The reusable multiplier is retained only to reproduce the documented
ablation; it defaults to 100% and changes no existing plan.

Reproduce with:

```bash
env PYTHONPATH=. PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
  MPLCONFIGDIR=/private/tmp/trade_planner_mpl \
  XDG_CACHE_HOME=/private/tmp/trade_planner_cache \
  ./.conda-env/bin/python experiments/systematic_first_risk_walkforward.py \
  --solver OSQP --n-events 12 --event-start 0 --risk-aversion medium \
  --output-prefix artifacts/systematic_first_risk_spent
```

The evidence is in `artifacts/systematic_first_risk_spent*`, including the
predeclared gates, all schedules, per-date risk decomposition, coefficients,
and the six-panel review chart.
