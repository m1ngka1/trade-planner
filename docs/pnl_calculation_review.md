# P&L Calculation Review: Holding P&L vs. Trading P&L

## Overall assessment

**Needs revision for daily reporting; conditionally correct for full-horizon total P&L.**

The calculation shown in `IMG_8553.jpg` does **not** economically double count the same price interval. Its two components are:

- trade-date P&L from the execution VWAP to that day's close; and
- forward holding P&L from that day's close to the next day's close.

Those intervals do not overlap. However, the forward holding P&L is grouped under the earlier date. Consequently, `hld_pnl` and `trd_pnl` on the same output date do not describe the same accounting day. Daily P&L is wrong even though the sum across a complete horizon can still be right.

The conclusion assumes:

1. signed `trade_shares` are positive for buys and negative for sells;
2. all shares for date \(t\) execute at the full-day VWAP \(V_t\);
3. positions are marked at the close \(C_t\);
4. the starting position is zero;
5. there is exactly one price row and, after aggregation, one schedule row per date and symbol;
6. all trading dates are present for each held symbol; and
7. corporate actions, fees, commissions, financing, borrow costs, taxes, and FX effects are ignored.

The repository's native `TradePlanner` schedule is a dense planner-date-by-symbol grid, so condition 6 holds for an unfiltered planner result. A schedule from another source, or a filtered schedule containing only nonzero trades, would not satisfy it automatically.

## Correct accounting identity

For one symbol, define:

- \(q_t\): signed shares traded during day \(t\);
- \(x_{t-1}\): position at the start of day \(t\), after the prior day's trades;
- \(x_t=x_{t-1}+q_t\): position at the end of day \(t\);
- \(V_t\): execution VWAP on day \(t\); and
- \(C_t\): closing price on day \(t\).

Ignoring cash interest, the marked portfolio value is \(W_t=x_tC_t+K_t\), where trading changes cash by \(K_t=K_{t-1}-q_tV_t\). Therefore:

\[
\begin{aligned}
\mathrm{PnL}_t
&=W_t-W_{t-1} \\
&=x_tC_t-x_{t-1}C_{t-1}-q_tV_t \\
&=x_{t-1}(C_t-C_{t-1})+q_t(C_t-V_t).
\end{aligned}
\]

This gives the clean daily decomposition:

\[
\boxed{\mathrm{holding\_pnl}_t=x_{t-1}(C_t-C_{t-1})}
\]

\[
\boxed{\mathrm{trading\_pnl}_t=q_t(C_t-V_t)}
\]

\[
\boxed{\mathrm{total\_pnl}_t=\mathrm{holding\_pnl}_t+\mathrm{trading\_pnl}_t}
\]

There is no overlap in this decomposition. The opening position earns the close-to-close move. Today's trade earns only the move from its execution VWAP to today's close. The formulas work for buys and sells because \(q_t\) is signed.

## What the current code calculates

With corporate-action multipliers set to one, the screenshot reduces to:

```python
eod_position = cumulative_sum(trade_shares)               # x_t
holding_pnl = eod_position * close_t * fwd_ret_1d_t       # x_t(C_{t+1} - C_t)
trd_pnl = trade_shares * vwap_t * trd_ret_t               # q_t(C_t - V_t)
```

Thus the row labelled date \(t\) contains:

\[
\mathrm{reported\_pnl}_t=x_t(C_{t+1}-C_t)+q_t(C_t-V_t).
\]

The two terms cover disjoint intervals, so there is no double counting. But the first term belongs to day \(t+1\), while the second belongs to day \(t\). Grouping both by `date` mixes two accounting periods.

The forward form can still reconcile over a complete horizon ending at close \(C_T\):

\[
\sum_{t=0}^{T}q_t(C_t-V_t)
+\sum_{t=0}^{T-1}x_t(C_{t+1}-C_t)
=x_TC_T-\sum_{t=0}^{T}q_tV_t,
\]

assuming \(x_{-1}=0\). This is why the grand total may look correct while the daily values are shifted.

## Numerical spot check

Consider one symbol with no starting position:

| Date | Close | VWAP | Trade shares | End position |
|---|---:|---:|---:|---:|
| Day 0 | 100 | 98 | +10 | 10 |
| Day 1 | 110 | 108 | +5 | 15 |
| Day 2 | 105 | 106 | -3 | 12 |

The correct daily decomposition is:

| Date | Holding P&L | Trading P&L | Total P&L |
|---|---:|---:|---:|
| Day 0 | \(0\) | \(10(100-98)=20\) | 20 |
| Day 1 | \(10(110-100)=100\) | \(5(110-108)=10\) | 110 |
| Day 2 | \(15(105-110)=-75\) | \((-3)(105-106)=3\) | -72 |
| **Total** | **25** | **33** | **58** |

The screenshot's date grouping would instead report:

| Date | Forward holding P&L | Trading P&L | Reported total |
|---|---:|---:|---:|
| Day 0 | \(10(110-100)=100\) | 20 | 120 |
| Day 1 | \(15(105-110)=-75\) | 10 | -65 |
| Day 2 | unavailable | 3 | 3 |
| **Total** | **25** | **33** | **58** |

The horizon total is the same, but every daily total is misdated. The direct terminal-value check also gives 58:

\[
12(105)-[10(98)+5(108)-3(106)]=58.
\]

## Recommended implementation

The least ambiguous implementation uses the beginning-of-day position and the prior close:

```python
keys = ["symbol", "date"]
merged = merged.sort_values(keys).copy()

# Aggregate to one row per symbol/date before this block if the input can
# contain multiple fills or schedule rows for the same symbol/date.
merged["eod_position"] = merged.groupby("symbol")["trade_shares"].cumsum()
merged["bod_position"] = merged["eod_position"] - merged["trade_shares"]
merged["prev_close"] = merged.groupby("symbol")["usd_close"].shift(1)

merged["holding_pnl"] = (
    merged["bod_position"] * (merged["usd_close"] - merged["prev_close"])
)
merged["trd_pnl"] = (
    merged["trade_shares"] * (merged["usd_close"] - merged["usd_vwap"])
)
merged["total_pnl"] = merged["holding_pnl"] + merged["trd_pnl"]
```

For a zero initial position, set the first day's `holding_pnl` to zero. If there is an initial position, load the preceding close and that position instead of filling the first value. Do not silently replace a missing prior close with zero when `bod_position` is nonzero.

An equivalent, but easier to misread, fix is to retain \(x_t(C_{t+1}-C_t)\) and assign that value to date \(t+1\). The backward-looking formula above is preferable because every row then contains P&L for the date printed on that row.

## Additional checks before production use

1. **Join cardinality:** verify that price data have one row per `(date, symbol)` and aggregate multiple schedule fills before merging. A many-to-many merge can multiply P&L.
2. **Calendar completeness:** include every trading date while a position is open, including zero-trade dates. Otherwise holding P&L disappears on omitted dates.
3. **Horizon boundary:** define the mark at which simulation starts and ends. A zero initial position makes first-day holding P&L zero; an existing initial position requires the prior close.
4. **Missing values:** avoid relying on `groupby.sum()` silently skipping a missing final forward return. Handle boundary rows explicitly.
5. **Reconciliation test:** for each symbol and for the portfolio, verify

   \[
   \sum_t\mathrm{total\_pnl}_t
   =x_TC_T-x_{-1}C_{-1}-\sum_tq_tV_t.
   \]

   With zero starting position, the \(x_{-1}C_{-1}\) term is zero.
6. **Adjusted prices:** when corporate actions are reintroduced, use one internally consistent adjusted-price/share basis for prior close, current close, VWAP, trades, and positions. Applying a return adjustment to only part of the dollar-P&L identity can break reconciliation.

## Final conclusion

- **Overlap:** no, not in price intervals. `trd_pnl` covers VWAP-to-close on trade date; the current `holding_pnl` covers that close-to-next-close.
- **Daily correctness:** no. Holding P&L is attached to the prior date, so the two reported components are not for the same day.
- **Full-horizon total:** potentially correct under the stated assumptions and with complete date coverage, unique joins, zero initial position, and an explicitly handled final mark.
- **Correct daily formula:** use beginning-of-day position times prior-close-to-current-close, plus signed traded shares times current-close-minus-VWAP.
