# QUANT LOG -- D14 Bias-Free Re-Evaluation

## Engine Design Decisions

### 1. SL-First Conservative Bar Ordering

**Problem**: D10/D13 TRAIL_D engine processed each 1-min bar as:
  1. Update peak (find new highest R-multiple)
  2. Update trail level (tighten SL based on new peak)
  3. Check if SL is hit

This is optimistic because within a single 1-min bar, the peak and trough
could occur in any order. By updating the peak first, the trail tightens,
and then the SL check uses this tighter (higher) stop. In reality, if price
hits the old SL before making a new high, you are already stopped out.

**Solution**: D14 reverses the order:
  1. Check if SL is hit (using CURRENT sl_level, not yet updated)
  2. If SL not hit: update peak
  3. Update trail level

**Code**: `streaming_trail.py`, lines 135-237.

**Why this is correct**: In live trading, your broker's stop order triggers
the moment price touches your SL level. The bar has not completed yet, so
you cannot know if price will later make a new high within the same minute.
SL-first is the only ordering consistent with live execution.

**Measured impact**: For Cat-1 strategies, the ambiguity count was 0. This
means no 1-min bar simultaneously hit SL and made a new peak. The SL-first
vs peak-first ordering made ZERO difference for Cat-1. This is expected
because the SL is 0.25 ADR (tight) and most SL hits occur early before
the trail activates. Once the trail is active, the stock is trending strongly
enough that bars rarely reverse 0.5R within a single minute.

### 2. 1-Sec Sparse Resolution

**Problem**: When a 1-min bar IS ambiguous (SL and new peak both possible),
we need sub-minute resolution to determine the correct ordering.

**Design choice**: Instead of running the entire simulation on 1-sec data
(which would be 60x slower), we use a sparse approach:
  1. Run simulation on 1-min bars
  2. Only when a bar is flagged as ambiguous, load 1-sec data for that
     specific ticker/date/minute
  3. Replay that minute at 1-sec resolution
  4. If no 1-sec data available: default to conservative (SL first)

**Why 1-sec sparse is sufficient**:
  - Ambiguous bars are extremely rare (0 for all 9 Cat-1 strategies)
  - At 0.25 ADR SL, the distance is significant enough that intra-bar
    ambiguity almost never occurs
  - For Cat-2/Cat-3, the streaming detection changes matter far more than
    intra-bar resolution
  - The 1-sec data covers 2140 ticker-dates out of ~10252 total events

**Implementation**: `streaming_trail.py` function `resolve_intrabar_1sec()`.
  - Loads 1-sec parquet for the specific minute
  - Replays second-by-second with SL-first ordering
  - Returns either 'sl_hit' or 'peak_update' with the resolved value
  - Returns 'no_data' if 1-sec file unavailable

### 3. SPY Open Return Precomputation

**Problem**: Strategy 1.4 uses SPY market direction as a filter. D13 used
spy_return_day (SPY close-to-close), which is not known at 09:35.

**Solution**: Precompute spy_return_935 for all 1226 trading days:
  - Load SPY 1-min data
  - Calculate return from previous close to 09:35 close
  - Store as dict {date_str: return_value}

**Implementation**: `spy_realtime.py`.

**Impact**: N dropped from 85 to 74 (13% fewer trades). EV improved from
+0.730R to +0.883R. The 11 removed trades were ones where SPY was positive
EOD but negative at 09:35 -- these were systematically weaker setups
because the market had not yet confirmed the bullish bias at entry time.

### 4. Streaming Gap Fill Tracking

**Problem**: Strategy 2.2 (Failed Fade) used gap_filled as a binary flag
from metadata, which is an EOD determination. At detection time during the
day, you cannot know if the gap will ultimately fill.

**Solution**: `gap_fill_tracker.py` tracks gap fill status bar-by-bar:
  - Uses prev_close from metadata as the gap fill level
  - For gap-up: fill occurs when low <= prev_close
  - For gap-down: fill occurs when high >= prev_close
  - Once filled, stays filled (monotonic)

**Impact**: 2.2 Failed Fade collapsed from +0.662R to +0.022R. The entire
strategy was built on knowing the gap fills before it actually does.

### 5. Streaming LOD/HOD for Pattern Detection

**Problem**: Cat-2 strategies used .idxmin() and .idxmax() to find the
day's low/high, then detected patterns around those points. This is the
most severe form of look-ahead: knowing the exact location of the day's
extremes at any point during the day.

**Solution**: Maintain running LOD/HOD as streaming values:
  - At bar t, LOD = min(low[0:t]), HOD = max(high[0:t])
  - Pattern detection uses only information available at bar t
  - Entries occur when the pattern condition is first met in the stream

**Impact**:
  - 2.1 Sell-Off Recovery: -82% EV (running LOD != absolute LOD)
  - 2.2 Failed Fade: -97% EV (almost entirely look-ahead)
  - 2.3 Dip and Rip: +136% EV (streaming finds EARLIER entries)

---

## Code Changes vs D13

### New Files (d14_bias_free/engine/)
| File | Purpose | Lines |
|------|---------|-------|
| streaming_trail.py | Bias-free TRAIL_D (intraday + multi-day) | ~373 |
| spy_realtime.py | SPY open-return precomputation | ~50 |
| gap_fill_tracker.py | Bar-by-bar gap fill state | ~40 |
| intrabar_resolver.py | 1-sec zoom for ambiguous bars | integrated |

### New Files (d14_bias_free/strategies/)
| File | Purpose | Lines |
|------|---------|-------|
| cat1_close935.py | 9 Cat-1 strategies (exact D13 params) | ~225 |
| cat2_selloff_recovery.py | Streaming sell-off detection | ~100 |
| cat2_failed_fade.py | Streaming failed fade detection | ~100 |
| cat2_dip_and_rip.py | Streaming dip-and-rip detection | ~100 |
| cat3_lateday_swing.py | 7 patterns x 3 entry times | ~200 |

### Orchestration
| File | Purpose |
|------|---------|
| d14_run_all.py | Main runner: loads data, runs all phases, outputs results |
| d14_compare.py | D13 vs D14 comparison and bias impact analysis |

### Key Differences from D13 Code
1. `streaming_trail.py` line 137: SL check is now the FIRST operation in the
   bar loop (was LAST in D10/D13).
2. `cat1_close935.py` line 99-101: spy_returns parameter passed to filter_1_4,
   maps date to precomputed spy_return_935.
3. Cat-2 strategies: Complete rewrite. Instead of post-hoc pattern matching on
   full-day data, each strategy streams through bars sequentially, maintaining
   running state (LOD, HOD, gap_fill_status) and triggering entries on first
   detection.
4. `streaming_trail.py` lines 266-372: Multi-day TRAIL_D also uses SL-first.

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total runtime | 44.0 minutes |
| Total events processed | 10252 (IS: 4254, OOS: 5903) |
| Total trades simulated | 112990 |
| 1-sec files indexed | 2140 |
| Ambiguous bars (Cat-1) | 0 |
| SPY dates computed | 1226/1226 (100%) |
| Bootstrap iterations | 10000 per strategy per split |

Runtime breakdown (estimated):
- Phase 0 (data loading + SPY): ~2 min
- Phase 1 (Cat-1, 9 strategies): ~8 min
- Phase 2 (Cat-2, 3 strategies): ~12 min (streaming pattern detection is slower)
- Phase 3 (Cat-3, 21 combos): ~20 min
- Phase 4 (summary + comparison): ~2 min

---

## Statistical Methodology

### Bootstrap Confidence Intervals
- Method: Percentile bootstrap (non-parametric)
- Iterations: 10,000
- CI level: 95% (2.5th and 97.5th percentiles)
- Resampling unit: Individual trades (with replacement)
- Statistic: Mean PnL in R-multiples

### Robustness Criterion
A strategy is "ROBUST" if and only if:
  - CI_lo > 0 (the lower bound of the 95% bootstrap CI excludes zero)

This is equivalent to a one-sided test at alpha=2.5% that the true mean
EV is positive.

### P-Value Calculation
- Bootstrap p-value: fraction of bootstrap samples with mean <= 0
- Significance levels: * p<0.10, ** p<0.05, *** p<0.01

### Profit Factor
- PF = sum(winning trades) / |sum(losing trades)|
- PF > 1.5 considered good, PF > 2.0 considered strong

### Win Rate
- Simple: #winners / #total
- For Cat-3 (multi-day TRAIL_D): winner = pnl > 0
- Low WR (~27%) for Cat-3 15:59 entries is expected: overnight gaps create
  large winners but many small losers (trail stops out quickly on mean-reversion)

---

## Intra-Bar Analysis

### Cat-1 Ambiguity Results
```
Strategy                | Ambig Bars | 1-Sec Zooms | Resolved |
------------------------|------------|-------------|----------|
1.1 S4 Optimiert        |          0 |           0 |        0 |
1.2 Pre-10am Momentum   |          0 |           0 |        0 |
1.3 Earnings+OD+RVOL    |          0 |           0 |        0 |
1.4 Power Setup          |          0 |           0 |        0 |
1.5 Big First Candle     |          0 |           0 |        0 |
1.6 Low Wick             |          0 |           0 |        0 |
1.7 Big Body             |          0 |           0 |        0 |
1.8 Gap 3.0+ ADR         |          0 |           0 |        0 |
1.9 RVOL>=5+RSI>50       |          0 |           0 |        0 |
TOTAL                    |          0 |           0 |        0 |
```

### Interpretation
Zero ambiguous bars across all Cat-1 strategies confirms that:
1. At 0.25 ADR SL, the stop distance is wide enough relative to 1-min bar
   ranges that a bar cannot simultaneously trigger SL and make a new peak
2. The SL-first vs peak-first ordering is IRRELEVANT for Cat-1
3. The D14 engine change has no effect on Cat-1 results -- any EV changes
   are due to N differences (filter corrections), not engine changes
4. The 1-sec intra-bar resolution infrastructure, while correct to build,
   was not needed for these strategies

### Why Build It Anyway?
- Cat-2 and Cat-3 use different SL distances and entry points
- Future strategies may have tighter SLs where ambiguity matters
- The infrastructure cost is minimal (only loads 1-sec data on demand)
- Correctness is more important than optimization

---

## Data Integrity Notes

1. **metadata_v9.parquet**: 10252 events, 89+ columns. IS/OOS split at 2024-01-01.
2. **1-min data**: data/raw_1min/{ticker}/{date}.parquet, session='rth' filter applied.
3. **1-sec data**: data/raw_1sec/{ticker}/{date}.parquet, 2140 files available.
4. **SPY data**: data/raw_1min/SPY/{date}.parquet, all 1226 trading days covered.
5. **Known issues**: close_location is STRING, pm_rth5 can be NaN, rvol_5 can be NaN.
   All handled via .notna() checks in base filters.
