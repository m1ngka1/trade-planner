# Diagnostics walkthrough: a hidden portfolio-policy conflict

## The situation

This example deliberately creates an infeasible plan whose cause is not a
simple “one stock does not have enough ADV capacity” error. Every name can be
completed individually. The failure appears only when two portfolio policies
are considered together:

- every order must reach 60% completion on the first planner date; and
- first-day gross traded notional may not exceed $2,000.

The runnable example is
[`examples/diagnostics_walkthrough.py`](../examples/diagnostics_walkthrough.py).
Its inputs are ordinary CSV files under [`examples/data/`](../examples/data/).

## Sample data

The parent orders are:

| Symbol | Target shares | Price | ADV | Daily cap at 10% ADV |
|---|---:|---:|---:|---:|
| AAA | +100 | $10 | 1,000 | 100 shares |
| BBB | -80 | $20 | 800 | 80 shares |
| CCC | +60 | $25 | 600 | 60 shares |

There are three open planner dates with the same daily participation capacity.
Each name therefore has three times the capacity needed to complete its parent
order. `HardCompletionConstraint.validate()` passes, which is important: this
is not the obvious per-name capacity error caught before solving.

The daily gross limits are:

| Date | Maximum gross traded notional |
|---|---:|
| 2026-07-15 | $2,000 |
| 2026-07-16 | $5,000 |
| 2026-07-17 | $5,000 |

## Why the combined model is infeasible

The 60% first-day milestone requires at least:

```text
AAA: 60 shares × $10 =   $600
BBB: 48 shares × $20 =   $960
CCC: 36 shares × $25 =   $900
                           -----
minimum first-day gross = $2,460
```

But the first-day gross constraint allows only $2,000:

```text
minimum required gross − allowed gross = $2,460 − $2,000 = $460
```

Neither rule is unreasonable when viewed alone. The milestone is feasible under
each name's participation cap, and the gross limit permits plenty of trading.
They become impossible only in combination.

There are several valid business fixes:

- raise the first-day gross limit by at least $460;
- lower the first-day completion milestone to at most
  `$2,000 / $4,100 = 48.78%`; or
- move the milestone later, when another day's gross budget is available.

The diagnostic certificate identifies the conflicting rules. The exact $460
repair comes from the business arithmetic above; certificate weights alone do
not calculate a minimum safe parameter change.

## Step 1: catch the real planner failure

The example uses CLARABEL, which requires no commercial license:

```python
try:
    sample_planner(ctx, gross_limits).solve(ctx)
except InfeasiblePlanError as error:
    print(error.diagnostics["text"])
    failed_problem = error.problem
```

`TradePlanner` preserves both the structured report and the exact failed CVXPY
problem on the exception. The diagnostic code inspects that same solved object;
it does not remove constraints or run a relaxed model.

On the current environment, CLARABEL returns an infeasible status and a mapped
constraint dual that highlights the milestone. The report deliberately labels
this as a lower-confidence fallback:

```text
Outcome: NO FEASIBLE SCHEDULE
Solver-returned infeasibility duals highlight 1 policy constraint(s)...

1. min_completion_by_date[2026-07-15]
   Where: CCC, BBB, AAA
   Why: The milestone completion fraction is too aggressive for earlier
        capacity or notional limits.
   PM action: Lower the milestone fraction, move the milestone later, or
              increase capacity before the milestone.

Evidence limit: This fallback is not MOSEK's mapped IIS; treat its ranking as
directional.
```

This is useful but incomplete: it points to the timing requirement without
naming the gross rule that conflicts with it. The diagnostic layer does not
invent the missing rule just because it looks plausible.

## Step 2: demonstrate the full certificate decoder without MOSEK

To exercise the complete decoding path, the example attaches a teaching-only
certificate-shaped mapping to the already failed problem.

The fixture is not arbitrary. Multiplying each minimum-share requirement by its
stock price proves that first-day gross must be at least $2,460. The gross rule
proves that it must be no more than $2,000. Those two statements form the known
contradiction.

The script marks the fixture prominently:

```text
WARNING: the mapped certificate below is analytical demo data, not solver output.
```

Calling the normal API again:

```python
attach_analytical_certificate_fixture(failed_problem, ctx)
report = diagnose_problem(failed_problem)
print(report["text"])
```

now maps both original constraint IDs back to their plugin-owned business
metadata:

```text
Conflict members — choose one or more business levers from this set:

1. min_completion_by_date[2026-07-15] [completion]
   Where: AAA, BBB, CCC and their minimum completed shares
   PM action: Lower the milestone fraction, move it later, or increase capacity.

2. daily_gross_notional_limit[2026-07-15] [notional]
   Current setting: $2,000
   PM action: Increase the gross limit or move more volume to other dates.
```

The percentage weights should not be interpreted as “98.2% of the blame.” A
certificate may be rescaled, and the constraints use different units. The
important result is the conflict set and its concrete business levers.

## What would change with a licensed MOSEK run

The example's first solve and exception handling are fully real. Only the
certificate fixture is a stand-in.

With a working MOSEK license, `DiagnosticMOSEK` would preserve MOSEK's
certificate in `problem.solver_stats.extra_stats`. `diagnose_problem()` would
then perform the same mapping shown above automatically:

```text
solver certificate
    → original CVXPY constraint IDs
    → ConstraintDiagnostics metadata
    → dates, symbols, current settings, causes, and PM actions
```

The production path still performs zero additional diagnostic solves.

## Run the example

From the repository root:

```bash
python examples/diagnostics_walkthrough.py
```

The output intentionally shows both stages so a new contributor can see the
difference between an honest open-source fallback and a fully mapped conflict
certificate.
