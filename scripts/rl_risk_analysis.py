"""
Durchlauf 5.0 — RISK MANAGER
Slippage stress, commission impact, drawdown, correlation, overtrading.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str((PROJECT_ROOT / 'scripts')))

from stable_baselines3 import PPO
from rl_environment import (make_env, GapperEnv, GapperEnvWrapper,
                            N_FEATURES, N_ACTIONS, DECISION_BARS, TIMEOUT_BAR,
                            SL_SIZES, TOTAL_COST_ADR, MAX_TRADES_PER_DAY)

BASE_DIR = PROJECT_ROOT
os.chdir(BASE_DIR)

EPISODES_PATH = BASE_DIR / 'data' / 'metadata' / 'rl_episodes.parquet'
MODEL_PATH = BASE_DIR / 'models' / 'rl_agent_v1.zip'


def evaluate_with_slippage(model, env, n_episodes, extra_slippage_adr):
    """
    Run model but add extra slippage per trade.
    Returns per-episode rewards adjusted for extra slippage.
    """
    rewards = []
    trade_counts = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_trades = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        n_trades = info.get('trade_count', 0)
        # Each trade incurs extra slippage
        adjusted = ep_reward - n_trades * extra_slippage_adr
        rewards.append(adjusted)
        trade_counts.append(n_trades)

    return np.array(rewards), np.array(trade_counts)


def drawdown_analysis(rewards):
    """Compute drawdown statistics from episode rewards."""
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum

    max_dd = drawdowns.max()
    max_dd_end = np.argmax(drawdowns)
    # Find start of max drawdown
    max_dd_start = np.argmax(cum[:max_dd_end + 1]) if max_dd_end > 0 else 0

    # Recovery
    if max_dd_end < len(cum) - 1:
        recovery_idx = np.where(cum[max_dd_end:] >= peak[max_dd_end])[0]
        recovery_days = recovery_idx[0] if len(recovery_idx) > 0 else len(cum) - max_dd_end
    else:
        recovery_days = -1  # Not recovered

    return {
        'max_drawdown_adr': max_dd,
        'max_dd_start_ep': max_dd_start,
        'max_dd_end_ep': max_dd_end,
        'max_dd_duration': max_dd_end - max_dd_start,
        'recovery_episodes': recovery_days,
        'cum_reward_final': cum[-1] if len(cum) > 0 else 0,
    }


def spy_correlation(rewards, episodes_df, split):
    """Check correlation between agent returns and SPY."""
    spy_returns = episodes_df[episodes_df['split'] == split]['spy_return_day'].values
    n = min(len(rewards), len(spy_returns))
    if n < 10:
        return {'correlation': 0, 'beta': 0, 'n': n}

    r = rewards[:n]
    s = spy_returns[:n]

    # Filter NaN
    mask = ~(np.isnan(r) | np.isnan(s))
    r = r[mask]
    s = s[mask]

    if len(r) < 10:
        return {'correlation': 0, 'beta': 0, 'n': len(r)}

    corr = np.corrcoef(r, s)[0, 1]
    # Beta = cov(r, s) / var(s)
    if s.std() > 0:
        beta = np.cov(r, s)[0, 1] / np.var(s)
    else:
        beta = 0

    return {'correlation': corr, 'beta': beta, 'n': len(r)}


def worst_case_scenarios(rewards, percentiles=[1, 5, 10]):
    """Compute worst-case scenario statistics."""
    results = {}
    for p in percentiles:
        val = np.percentile(rewards, p)
        results[f'worst_{p}pct'] = val

    # Worst streak
    negative = rewards < 0
    max_streak = 0
    current = 0
    for neg in negative:
        if neg:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    results['worst_losing_streak'] = max_streak

    # Average of worst 10 days
    results['avg_worst_10'] = np.sort(rewards)[:10].mean() if len(rewards) >= 10 else 0

    return results


def main():
    print("=" * 60, file=sys.stderr)
    print("DURCHLAUF 5.0 — RISK ANALYSIS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if not MODEL_PATH.exists():
        print(f"ERROR: No model at {MODEL_PATH}", file=sys.stderr)
        return

    # Load
    episodes = pd.read_parquet(EPISODES_PATH)
    model = PPO.load(MODEL_PATH)
    val_env = make_env(episodes, split='val', feature_dropout=0.0, noise_std=0.0)
    n_val = val_env.env.n_episodes

    output = []
    output.append("DURCHLAUF 5.0 — RISK ANALYSIS")
    output.append("=" * 70)
    output.append(f"Timestamp: {datetime.now().isoformat()}")
    output.append(f"Val episodes: {n_val}")

    # 1. Slippage Stress Test
    print("\n1. Slippage Stress Test...", file=sys.stderr)
    slippage_levels = [0.0, 0.01, 0.02, 0.05]

    output.append("\n\n1. SLIPPAGE STRESS TEST")
    output.append("-" * 60)
    output.append(f"Base costs already included: {TOTAL_COST_ADR:.3f} ADR/trade")
    output.append(f"These are ADDITIONAL slippage on top of base costs")
    output.append(f"\n{'Extra Slip':>10} {'MeanRew':>8} {'Sharpe':>8} {'CumRew':>8} "
                 f"{'Profitable%':>11}")

    for slip in slippage_levels:
        rews, trades = evaluate_with_slippage(model, val_env, n_val, slip)
        mean_r = rews.mean()
        sharpe = mean_r / (rews.std() + 1e-8)
        cum = rews.sum()
        prof_pct = (rews > 0).sum() / len(rews) * 100
        output.append(f"{slip:>10.3f} {mean_r:>+8.4f} {sharpe:>8.3f} "
                     f"{cum:>+8.2f} {prof_pct:>10.1f}%")

    # 2. Commission Impact
    print("\n2. Commission Impact...", file=sys.stderr)
    comm_levels = [0.0, 0.005, 0.010, 0.020]  # Additional commission per share ADR

    output.append("\n\n2. COMMISSION IMPACT")
    output.append("-" * 60)
    output.append(f"{'Extra Comm':>10} {'MeanRew':>8} {'Sharpe':>8}")

    for comm in comm_levels:
        rews, trades = evaluate_with_slippage(model, val_env, n_val, comm)
        mean_r = rews.mean()
        sharpe = mean_r / (rews.std() + 1e-8)
        output.append(f"{comm:>10.3f} {mean_r:>+8.4f} {sharpe:>8.3f}")

    # 3. Max Drawdown Analysis
    print("\n3. Drawdown Analysis...", file=sys.stderr)
    base_rews, base_trades = evaluate_with_slippage(model, val_env, n_val, 0.0)
    dd = drawdown_analysis(base_rews)

    output.append("\n\n3. MAX DRAWDOWN ANALYSIS")
    output.append("-" * 60)
    for k, v in dd.items():
        output.append(f"  {k}: {v}")

    # 4. SPY Correlation (Beta Check)
    print("\n4. SPY Correlation...", file=sys.stderr)
    spy = spy_correlation(base_rews, episodes, 'val')

    output.append("\n\n4. SPY CORRELATION")
    output.append("-" * 60)
    output.append(f"  Correlation: {spy['correlation']:.4f}")
    output.append(f"  Beta: {spy['beta']:.4f}")
    output.append(f"  N: {spy['n']}")
    if abs(spy['correlation']) > 0.3:
        output.append("  WARNING: High correlation with SPY — not market-neutral!")
    else:
        output.append("  OK: Low correlation with SPY")

    # 5. Trade Frequency Check
    print("\n5. Trade Frequency...", file=sys.stderr)
    output.append("\n\n5. TRADE FREQUENCY CHECK")
    output.append("-" * 60)
    trade_days = (base_trades > 0).sum()
    no_trade_days = (base_trades == 0).sum()
    output.append(f"  Total episodes: {len(base_trades)}")
    output.append(f"  Trade days: {trade_days} ({trade_days/len(base_trades)*100:.1f}%)")
    output.append(f"  No-trade days: {no_trade_days} ({no_trade_days/len(base_trades)*100:.1f}%)")
    output.append(f"  Avg trades per trade-day: {base_trades[base_trades>0].mean():.2f}")

    trade_dist = np.bincount(base_trades.astype(int), minlength=4)
    for i in range(4):
        output.append(f"  {i} trades: {trade_dist[i]} days")

    if trade_days / len(base_trades) < 0.20:
        output.append("  WARNING: Agent trades on <20% of days (undertrading)")
    if base_trades.mean() > 2.5:
        output.append("  WARNING: Agent averages >2.5 trades/day (overtrading risk)")

    # 6. Worst-Case Scenarios
    print("\n6. Worst-Case Scenarios...", file=sys.stderr)
    worst = worst_case_scenarios(base_rews)

    output.append("\n\n6. WORST-CASE SCENARIOS")
    output.append("-" * 60)
    for k, v in worst.items():
        output.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 7. Win/Loss Distribution
    print("\n7. Win/Loss Distribution...", file=sys.stderr)
    trading_rews = base_rews[base_trades > 0]

    output.append("\n\n7. WIN/LOSS DISTRIBUTION (trade days only)")
    output.append("-" * 60)
    if len(trading_rews) > 0:
        wins = trading_rews[trading_rews > 0]
        losses = trading_rews[trading_rews <= 0]
        output.append(f"  Win days: {len(wins)} ({len(wins)/len(trading_rews)*100:.1f}%)")
        output.append(f"  Loss days: {len(losses)} ({len(losses)/len(trading_rews)*100:.1f}%)")
        if len(wins) > 0:
            output.append(f"  Avg win: +{wins.mean():.4f} ADR")
        if len(losses) > 0:
            output.append(f"  Avg loss: {losses.mean():.4f} ADR")
        if len(wins) > 0 and len(losses) > 0 and losses.mean() != 0:
            output.append(f"  Win/Loss ratio: {abs(wins.mean()/losses.mean()):.2f}")

        # Percentiles
        output.append(f"\n  Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            output.append(f"    P{p:>2}: {np.percentile(trading_rews, p):>+.4f} ADR")

    # Save
    output_path = BASE_DIR / 'results' / 'rl_risk_analysis.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(output))

    print(f"\nResults saved to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
