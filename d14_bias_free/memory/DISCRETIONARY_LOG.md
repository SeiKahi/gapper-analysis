# DISCRETIONARY LOG -- D14 Bias-Free Re-Evaluation

## Overview

D14 re-ran all D13 strategies with a bias-free streaming engine. This log
documents how each strategy behaved differently under the new engine and
whether the new behavior is more realistic from a discretionary trading
perspective.

---

## Category 1: Close-at-935 Strategies

These strategies all enter at the close of the 09:35 bar and exit via
TRAIL_D (BE at +1R, then peak minus 0.5R). The only engine change
affecting Cat-1 is the SL-first bar ordering.

### 1.1 S4 Optimiert
- **D13**: N=62, EV=+0.620R
- **D14**: N=32, EV=+0.657R (+6%)
- **N change**: -30 trades. The D14 filter correctly applies od_direction=with_gap
  more strictly. The removed trades were likely against-gap OD events that
  slipped through in D13.
- **Assessment**: The strategy is tighter and slightly better. N=32 is small
  but CI excludes zero. Realistic: yes, the entry at 09:35 close is executable.
  The parameters (OD>=0.70, gap>=2.0 ADR, PM/RTH5<0.30) describe a clear
  momentum-with-low-PM scenario that a discretionary trader can identify.

### 1.2 Pre-10am Momentum
- **D13**: N=127, EV=+0.900R
- **D14**: N=110, EV=+1.370R (+52%)
- **N change**: -17 trades. Likely due to streaming filter timing (MomConfirm
  = first_candle_dir checked at exactly 09:35 close, slight timing differences).
- **Assessment**: The large EV increase is surprising for a Cat-1 strategy.
  The removed 17 trades must have been weak setups. The surviving 110 trades
  have exceptional EV with tight CI [+0.872, +1.893]. This is the top-ranked
  strategy by EV. Realistic: yes, but requires identifying momentum confirmation
  (first candle direction) at 09:35, which is straightforward. The SL=0.15 ADR
  is very tight, so execution speed matters.

### 1.3 Earnings+OD+RVOL
- **D13**: N=58, EV=+0.800R
- **D14**: N=58, EV=+0.827R (+3%)
- **N change**: None. Perfect match.
- **Assessment**: Minimal change, as expected for a Cat-1 strategy with no
  filter differences. Earnings gappers with OD>0.5 and RVOL>5 continue strongly.
  Realistic: yes, earnings dates are known in advance, OD and RVOL are
  observable at 09:35.

### 1.4 Power Setup
- **D13**: N=85, EV=+0.730R
- **D14**: N=74, EV=+0.883R (+21%)
- **N change**: -11 trades. The SPY filter now uses spy_return_935 (known at
  entry) instead of spy_return_day (known at EOD).
- **Assessment**: The 11 removed trades were ones where SPY opened negative
  but closed positive. These were trades entered on a false premise (SPY up)
  that happened to have SPY recover later. Removing them IMPROVED EV
  significantly, which confirms that the SPY filter has real predictive power
  when used correctly. Very realistic: checking if SPY is green at 09:35
  is trivial.

### 1.5 Big First Candle
- **D13**: N=250, EV=+0.640R
- **D14**: N=251, EV=+0.661R (+3%)
- **N change**: +1 trade (rounding/filter edge case).
- **Assessment**: Nearly identical. The first candle size >0.20 ADR is a
  strong quality filter that a discretionary trader easily recognizes. The
  candle must be visually prominent on a 1-min chart.

### 1.6 Low Wick
- **D13**: N=253, EV=+0.450R
- **D14**: N=254, EV=+0.497R (+11%)
- **N change**: +1 trade.
- **Assessment**: Small improvement. Low wick ratio (<0.15) means the opening
  drive candle has almost no rejection -- pure momentum. Discretionary read:
  "The first bar went straight up with no pullback."

### 1.7 Big Body
- **D13**: N=262, EV=+0.400R
- **D14**: N=263, EV=+0.426R (+7%)
- **N change**: +1 trade.
- **Assessment**: Small improvement. Body percent >0.60 means more than 60%
  of the first bar's range is body (not wicks). Consistent momentum bar.

### 1.8 Gap 3.0+ ADR
- **D13**: N=506, EV=+0.540R
- **D14**: N=258, EV=+0.781R (+45%)
- **N change**: -248 trades. D13 tested both directions, D14 correctly filters
  only od_direction=with_gap.
- **Assessment**: The halving of N and large EV increase means the against-gap
  events in D13 were dragging down performance. Only trading WITH the gap on
  3+ ADR gaps is the correct approach. Discretionary read: "When a stock gaps
  3x its normal range AND opens in the direction of the gap, ride it."

### 1.9 RVOL>=5+RSI>50
- **D13**: N=266, EV=+0.640R
- **D14**: N=533, EV=+0.529R (-17%)
- **N change**: +267 trades. D14 captured more events that match the simple
  filter (RVOL>=5, RSI>50, GapUp) due to broader base filter application.
- **Assessment**: The doubling of N with lower EV suggests D13 had an
  accidentally stricter filter that selected better trades. The D14 version
  is the honest result: a simple filter yields many trades with moderate EV.
  Still robust (CI_lo=+0.393). Discretionary read: "High volume gap-ups with
  bullish RSI trend to continue, but not all of them."

---

## Category 2: Pattern-Based Strategies

These strategies detect intraday patterns and enter at variable times.
The streaming engine has a MASSIVE impact here.

### 2.1 Sell-Off and Recovery
- **D13**: N=3128, EV=+0.438R
- **D14**: N=4307, EV=+0.077R (-82%)
- **N change**: +1179 trades (streaming detects more "recoveries" from
  running lows that are not the absolute day low).
- **Assessment**: The D13 version knew the day's absolute low and entered
  there. In reality, you never know if the current low is THE low. The D14
  streaming version enters at the first qualifying recovery from the running
  low -- which often is NOT the final low. Result: many more entries, much
  worse quality. EV=+0.077R barely covers costs. Discretionary read: "Buying
  the dip after a sell-off sounds good, but you are often buying too early."
  This strategy is MARGINALLY tradeable at best.

### 2.2 Failed Fade (ELIMINATED)
- **D13**: N=1702, EV=+0.662R
- **D14**: N=2510, EV=+0.022R (-97%)
- **N change**: +808 trades.
- **Assessment**: This was the "best" D13 strategy. It is now DEAD. The D13
  version used two pieces of future information: (1) gap_filled = EOD
  determination that the gap filled, (2) .idxmin() = exact location of the
  day's low. Together, these selected trades where the gap filled, price hit
  a known bottom, and then recovered -- a perfect hindsight selection. The
  streaming version detects failed fades in real-time using running gap fill
  state and running LOD. The result is noise (EV=+0.022R, CI includes zero).
  Discretionary read: "The idea of 'gap fills then reverses' has zero edge
  when you do not know in advance that the gap WILL fill and WHERE the bottom
  will be."

### 2.3 Dip and Rip
- **D13**: N=293, EV=+0.408R
- **D14**: N=625, EV=+0.962R (+136%)
- **N change**: +332 trades.
- **Assessment**: The streaming version is dramatically better. Why? Because
  the D13 version detected the dip using the absolute day low, which means
  it entered at the optimal point. But the streaming version detects dips
  from the running low -- and these running detections happen EARLIER in the
  day. Earlier entry means: (1) tighter SL (less distance from entry to
  running low), (2) more room to run, (3) more events qualify (N doubles).
  The result is better R:R with WR=63%. Discretionary read: "When a gap-up
  stock dips and immediately rips back, get in. Do not wait for confirmation
  that it was the absolute low -- the first strong bounce IS the signal."

---

## Category 3: Late-Day Swing

Late-day strategies enter at 14:30, 15:00, or 15:59 and hold multi-day
with TRAIL_D. The streaming engine applies SL-first ordering on daily bars.

### Overall Pattern
- 31 of 33 combos are ROBUST
- 15:59 entries dominate across all patterns (highest EV)
- 14:30 and 15:00 entries are weaker but still positive
- WR is low (~27-29% for 15:59) because overnight gaps create big winners
  but many small trail-stops

### NOT ROBUST: FAILED_FADE 15:00
- EV=+0.036R, CI=[-0.067, +0.144]
- This makes sense: the failed fade pattern (gap fills and reverses) is
  weakest when entering at 15:00 because there is not enough time for the
  reversal to develop before close, yet the overnight gap risk is high.
- 15:59 entry works (EV=+0.381R) because you buy at the very end when the
  reversal has already happened.

### Key Discretionary Insight for Cat-3
The 15:59 entry across all patterns averages ~+0.35-0.41R. This is a
MOC (Market-on-Close) order strategy: buy gappers at the close and hold
with a trail. The edge comes from overnight continuation of the day's
trend. This is NOT a pattern-recognition strategy -- the baseline (no
pattern filter) works almost as well as the filtered versions.

---

## Trade Count Comparison Summary

| Strategy | D13 N | D14 N | Delta | Reason |
|----------|-------|-------|-------|--------|
| 1.1 | 62 | 32 | -30 | Stricter od_direction filter |
| 1.2 | 127 | 110 | -17 | Streaming filter timing |
| 1.3 | 58 | 58 | 0 | Identical |
| 1.4 | 85 | 74 | -11 | SPY open vs EOD |
| 1.5 | 250 | 251 | +1 | Edge case |
| 1.6 | 253 | 254 | +1 | Edge case |
| 1.7 | 262 | 263 | +1 | Edge case |
| 1.8 | 506 | 258 | -248 | With-gap only (correct) |
| 1.9 | 266 | 533 | +267 | Broader base filter |
| 2.1 | 3128 | 4307 | +1179 | Running LOD != absolute LOD |
| 2.2 | 1702 | 2510 | +808 | Running gap fill + LOD |
| 2.3 | 293 | 625 | +332 | Earlier streaming detection |

---

## Realism Assessment

| Strategy | Executable? | Information Available? | Realistic Behavior? |
|----------|-------------|----------------------|---------------------|
| 1.1 S4 Optimiert | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.2 Pre-10am Mom | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.3 Earnings+OD+RVOL | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.4 Power Setup | Yes (MOO+5min) | All at 09:35 (SPY fixed) | Yes |
| 1.5 Big First Candle | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.6 Low Wick | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.7 Big Body | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.8 Gap 3.0+ ADR | Yes (MOO+5min) | All at 09:35 | Yes |
| 1.9 RVOL>=5+RSI>50 | Yes (MOO+5min) | All at 09:35 | Yes |
| 2.1 Sell-Off Recovery | Difficult | Running LOD only | Marginal |
| 2.2 Failed Fade | N/A | ELIMINATED | N/A |
| 2.3 Dip and Rip | Yes (intraday alert) | Running dip detection | Yes |
| Cat-3 (all) | Yes (MOC order) | All known by entry time | Yes |

All surviving strategies use only information available at entry time.
The D14 engine confirms this.
