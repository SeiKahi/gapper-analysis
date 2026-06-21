"""Quick check of the mathematical mechanism behind RW artefact."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

sigma = 0.003894
n = 390
daily_std = sigma * np.sqrt(n)

print(f"sigma per min: {sigma:.6f}")
print(f"Daily std: {daily_std:.4f} = {daily_std*100:.2f}%")
print()

adr_ratio = 0.0412
sl_ratio = 0.35 * adr_ratio  # = 0.0144
tgt_ratio = 1.05 * adr_ratio  # = 0.0433

print(f"SL distance: {sl_ratio:.4f} = {sl_ratio/daily_std:.2f} daily sigma")
print(f"Target distance: {tgt_ratio:.4f} = {tgt_ratio/daily_std:.2f} daily sigma")
print()

# For Brownian Motion, P(hit +a before -b) = b/(a+b)
prob_tgt = sl_ratio / (sl_ratio + tgt_ratio)
prob_sl = tgt_ratio / (sl_ratio + tgt_ratio)
print(f"Infinite horizon (no timeout):")
print(f"  P(target hit first) = {prob_tgt:.4f} = {prob_tgt*100:.1f}%")
print(f"  P(SL hit first) = {prob_sl:.4f} = {prob_sl*100:.1f}%")
print(f"  E[PnL] = {prob_tgt:.4f}*{tgt_ratio:.4f} - {prob_sl:.4f}*{sl_ratio:.4f} = {prob_tgt*tgt_ratio - prob_sl*sl_ratio:.6f}")
print(f"  (Should be 0 by optional stopping theorem)")
print()

# But with timeout, the absorbed paths are removed,
# and surviving paths have asymmetric bounds: (-SL, +Target)
# The WR discrepancy (26.8% RW vs 6.1% real) is because:
# 1) Real gapper volatility on the gap day is MUCH higher than ADR_10
#    (ADR_10 is historical, gap day range is typically 2-4x larger)
# 2) So real SL in "today's sigma" units is MUCH smaller -> more SL hits
# 3) And real Target in "today's sigma" units is correspondingly smaller -> fewer target hits

# Let me compute what the "real" effective volatility is
# If real WR = 6.1% for T4_3R, and SL hit = 34.8%, timeout = 59.1%
# Total absorbed = 6.1 + 34.8 = 40.9%
# Timeout = 59.1%
real_wr = 0.061
real_sl = 0.348
real_to = 0.591

rw_wr = 0.268
rw_sl = 0.722
rw_to = 0.011

print(f"Real data: WR={real_wr:.1%}, SL={real_sl:.1%}, TO={real_to:.1%}")
print(f"Random Walk: WR={rw_wr:.1%}, SL={rw_sl:.1%}, TO={rw_to:.1%}")
print()
print("HUGE DISCREPANCY:")
print(f"  Real timeout rate: {real_to:.1%} vs RW: {rw_to:.1%}")
print(f"  Real WR: {real_wr:.1%} vs RW: {rw_wr:.1%}")
print()
print("INTERPRETATION:")
print("  The RW simulation uses ADR_10 as external parameter,")
print("  but 1-min volatility is from ALL gapper days (which include gap day).")
print("  On gap days, volatility is MUCH higher than normal days.")
print("  So the simulated paths are TOO VOLATILE relative to the ADR reference.")
print("  This means SL and Target are both hit MORE OFTEN than in reality.")
print()
print("  Specifically:")
print("  - In real data, SL (0.35 ADR_10) is hit 34.8% of the time")
print("  - In simulation, it's hit 72.2% of the time (2x more)")
print("  - This is because simulated daily range >> ADR_10")
print()
print("  The positive expectancy in the RW comes from the ASYMMETRY:")
print("  Target = 3x SL. With absorbing barriers at -1 and +3 units,")
print("  the probability of hitting +3 before -1 is 1/(1+3) = 25%.")
print("  25% * 3 - 75% * 1 = 0 (fair game with infinite horizon).")
print("  But with finite timeout and these levels being easily reachable,")
print("  almost no trades timeout (1.1%), so the timeout bias is minimal.")
print()
print("  The TRUE source of positive expectancy on the RW is:")
print("  The interplay between discrete-time sampling and continuous barriers.")
print("  At each bar, price can JUMP PAST the SL or target.")
print("  Jumps past target contribute more PnL (positive excess) than")
print("  jumps past SL (negative excess) because target = 3x SL distance.")
print("  This is the DISCRETIZATION BIAS in barrier options.")
print()

# Compute expected overshoot
# When price hits SL, average overshoot ~ sigma * 0.5 (half a bar's move)
# When price hits Target, average overshoot ~ sigma * 0.5
# But we cap at the barrier price, so overshoot for SL is limited
# Wait, in the simulation, exit_price = sl_price (exact), not the actual bar close
# So there's NO overshoot captured - it's exact barrier execution
# THEN the E[PnL] should be ~0 for BM...
# UNLESS the log-normal vs linear asymmetry matters

print("WAIT - checking simulation code logic...")
print("  SL exit: exit_price = sl_price (exact at barrier)")
print("  Target exit: exit_price = target_price (exact at barrier)")
print("  So overshoot is NOT captured. E[PnL] should be 0 for BM.")
print("  But BM is in LOG space, trades are in LINEAR space!")
print()
print("  The simulation uses log-normal prices but linear SL/Target.")
print("  This creates a POSITIVE BIAS because:")
print("  - Log-normal distribution is right-skewed")
print("  - Entry price + target_dist in dollars is EASIER to reach")
print("    than entry price - sl_dist in dollars (for the same log move)")
print("  - Because exp(x) > 1-exp(-x) for x > 0")
print()
lognorm_bias = np.exp(0.5 * sigma**2 * 240) - 1
print(f"  Jensen's inequality bias over 240 bars: {lognorm_bias:.6f} (tiny)")
print(f"  This is NOT enough to explain +0.0319 ADR")
print()
print("  REAL EXPLANATION: The per-minute sigma (0.003894) is from GAPPER days.")
print("  Gapper days have ~2-3x normal volatility.")
print("  If we compute daily range from this sigma:")
daily_range_expected = sigma * np.sqrt(n) * 1.6  # factor for high-low of BM
print(f"  Expected daily range: {daily_range_expected:.4f} = {daily_range_expected*100:.1f}%")
print(f"  This is {daily_range_expected / adr_ratio:.1f}x the ADR_10 ratio of {adr_ratio:.4f}")
print(f"  So the simulated intraday range is ~{daily_range_expected/adr_ratio:.1f}x the ADR reference")
print(f"  SL at 0.35 * ADR = {sl_ratio:.4f} is only {sl_ratio/daily_range_expected:.2f} of actual daily range")
print(f"  Target at 1.05 * ADR = {tgt_ratio:.4f} is only {tgt_ratio/daily_range_expected:.2f} of actual daily range")
print()
print("CONCLUSION:")
print("  The RW simulation volatility (from gapper 1-min data) is ~3x the ADR_10.")
print("  This makes SL and target both very easy to reach (99% absorbed).")
print("  The positive expectancy comes from the log-normal + finite horizon effects.")
print("  In REAL data, the effective volatility-to-ADR ratio is ALSO high on gap days,")
print("  which is why real data shows similar patterns (59% timeout shows barriers")
print("  are wider relative to the moves).")
print()
print("  For a FAIR comparison, the RW sigma should be calibrated so that")
print("  the timeout rate matches the real data (~59% for T4_3R).")
print("  This requires a LOWER sigma or WIDER SL/Target.")
