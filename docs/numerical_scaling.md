# Numerically scaled execution formulation

## Predeclared hypothesis

The failed forecast-profit constraints were already divided by one basis point
of parent gross, and the planner already multiplies the whole objective by one
positive scalar. The remaining numerical mismatch is in the decision system:
raw share variables, million-share completion equalities, small normalized
economic inequalities, and dollar objectives coexist in one canonical model.

Keep each name's decision variable and every economic objective in raw shares.
Build a separate unit expression by dividing shares by that name's absolute
parent target; zero-target names use their largest daily cap or one share. This
avoids worsening the dollar-objective Hessian while normalizing the hard
system. Keep `state.trades`, accumulated inventory, all objective plugins, and
every public schedule in shares. Express the built-in participation,
direction, zero-target, completion, and milestone constraints in per-name
units. Express notional and factor limits relative to their own dollar bound.
This is an algebraic change of variables, not a new investment preference or a
hard-coded trading rule.

Every solve will also report an independent raw-share certificate:

- maximum participation-cap excess;
- maximum wrong-direction shares; and
- maximum terminal completion error.

When strict verification is enabled, the planner must reject an `optimal`
solver result that breaches desk tolerances instead of passing an invalid
schedule downstream.

## Protected surfaces

- Do not change any forecast, realized outcome, event seed, cohort split,
  acceptance gate, risk coefficient, liquidity coefficient, or scoring rule.
- Do not reopen unused events 109–120 or allocate events 121–144.
- Do not use historical holdout data; none is available yet.
- Use CLARABEL as the single explicit backend for the fixed mechanics
  comparison. The exact expected-net-P&L floor is a convex quadratic
  constraint and therefore is not an OSQP problem. The older command requested
  OSQP but the planner silently fell back to CLARABEL; correcting that label is
  not a solver sweep.
- Preserve economic objective units and plugin-facing share expressions.

## Acceptance criteria

The implementation is kept only if all of the following hold:

1. On deterministic heterogeneous-order fixtures, scaled and unscaled models
   produce the same economic optimum within solver tolerance, while the scaled
   schedule satisfies raw-share cap, direction, and completion certificates.
2. Existing custom constraints that read `state.trades` continue to receive
   share expressions.
3. On already-spent synthetic events 25–26 with CLARABEL, the exact quadratic
   expected-net-P&L floor that previously returned false infeasibility solves
   for both events.
4. Each candidate clears its recorded forecast floor with at least `-$1` of
   numerical tolerance, starts urgent names no later than the flat-ADV
   baseline, and satisfies cap excess `<= 0.05` share, wrong-direction flow
   `<= 0.001` share, and terminal error `<= 0.001` share.
5. The full repository test suite passes.

Passing this screen keeps the scaling formulation for real-data replay. It
does not revive the discarded profit-floor investment policy, authorize a new
synthetic cohort, or establish profitability.

## Implementation finding recorded before the fixed rerun

The first implementation substituted parent-order units into both constraints
and dollar objectives. On event 25, OSQP stalled at `user_limit` or issued a
false infeasibility certificate even though the proportional-cap reference is
exactly feasible. A constraint-only version keeps the original objective
matrix and passed the deterministic CLARABEL equivalence fixture. The fixed
spent-event screen therefore uses this narrower algebraic transformation.

## Fixed screen result

The corrected formulation passed every predeclared mechanics criterion on
spent events 25–26:

| Check | Event 25 | Event 26 | Limit |
|---|---:|---:|---:|
| Forecast net-P&L floor slack | +$61,972.86 | +$0.04 | at least -$1.00 |
| Maximum cap excess | 0.000011 share | 0 share | at most 0.05 share |
| Maximum wrong-direction flow | 0 share | 0 share | at most 0.001 share |
| Terminal completion error | 0.000000003 share | 0.0000000003 share | at most 0.001 share |
| Urgent-start delta versus baseline | 0 day | 0 day | at most 0 day |

The formulation is kept for real-data replay. The investment policy remains
discarded: across these two spent events, volatility rose `6.658 bp`, one
small order started a day earlier, and early factor imbalance rose `1.523`
percentage points. Realized net P&L improved by `$480k`, impact fell by `$55k`,
loss-CVaR fell `0.214 bp`, and the rising-volume shape strengthened, but spent
mechanics data cannot authorize promotion and the full 16-gate result failed.

Artifacts: `artifacts/numerical_scaling_mechanics*`.
