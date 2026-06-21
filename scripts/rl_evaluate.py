"""
Durchlauf 5.0 — EVALUATOR
Baselines, RL agent evaluation, and comparison.
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
                            DECISION_BARS, TIMEOUT_BAR, SL_SIZES,
                            ACTION_NOTHING, ACTION_LONG_TIGHT, ACTION_LONG_MED,
                            ACTION_LONG_WIDE, ACTION_SHORT_TIGHT, ACTION_SHORT_MED,
                            ACTION_SHORT_WIDE, ACTION_CLOSE, N_ACTIONS,
                            MAX_TRADES_PER_DAY, TOTAL_COST_ADR)

BASE_DIR = PROJECT_ROOT
os.chdir(BASE_DIR)

EPISODES_PATH = BASE_DIR / 'data' / 'metadata' / 'rl_episodes.parquet'
MODEL_PATH = BASE_DIR / 'models' / 'rl_agent_v1.zip'


# ── BASELINES ────────────────────────────────────────────────────────

class RandomAgent:
    """Baseline 1: Random actions."""
    def predict(self, obs, **kwargs):
        return np.random.randint(N_ACTIONS), None


class AlwaysLongAgent:
    """Baseline 2: Always Long at 9:35 with SL 0.30, hold till timeout."""
    def __init__(self):
        self.entered = False

    def predict(self, obs, **kwargs):
        if not self.entered:
            self.entered = True
            return ACTION_LONG_MED, None  # Long with 0.30 SL
        return ACTION_NOTHING, None

    def reset(self):
        self.entered = False


class WithGapAgent:
    """Baseline 3: Long on GapUp, Short on GapDown at 9:35."""
    def __init__(self):
        self.entered = False

    def predict(self, obs, **kwargs):
        if not self.entered:
            self.entered = True
            gap_dir = obs[0]  # feature 0 = gap_direction_num
            if gap_dir > 0:
                return ACTION_LONG_MED, None
            else:
                return ACTION_SHORT_MED, None
        return ACTION_NOTHING, None

    def reset(self):
        self.entered = False


class ODFollowerAgent:
    """Baseline 4: Trade in OD direction when OD strength > 0.5 ADR."""
    def __init__(self):
        self.entered = False
        self.waited = False

    def predict(self, obs, **kwargs):
        od_strength = obs[10]  # feature 11 = od_strength_adr
        od_with_gap = obs[11]  # feature 12 = od_with_gap

        if not self.entered and od_strength > 0.5:
            self.entered = True
            # Trade in OD direction
            if od_with_gap > 0:
                # OD is with gap
                gap_dir = obs[0]
                if gap_dir > 0:
                    return ACTION_LONG_MED, None
                else:
                    return ACTION_SHORT_MED, None
            else:
                # OD against gap
                gap_dir = obs[0]
                if gap_dir > 0:
                    return ACTION_SHORT_MED, None
                else:
                    return ACTION_LONG_MED, None
        return ACTION_NOTHING, None

    def reset(self):
        self.entered = False
        self.waited = False


class PMRTHRatioAgent:
    """
    Baseline 5: Best signal from Durchlauf 4.5.
    Long if GapUp + PM/RTH<10%, Short if GapDown + PM/RTH<10%.
    Entry at 10:00 (when PM/RTH ratio becomes available, ~decision step 6).
    """
    def __init__(self):
        self.entered = False
        self.step_count = 0

    def predict(self, obs, **kwargs):
        self.step_count += 1

        # Wait until step 6 (~10:00) when PM/RTH ratio is meaningful
        if self.step_count < 6:
            return ACTION_NOTHING, None

        pm_rth = obs[16]  # feature 17 = pm_rth30_ratio

        if not self.entered and pm_rth < 0.10 and pm_rth > 0:
            self.entered = True
            gap_dir = obs[0]
            if gap_dir > 0:
                return ACTION_LONG_WIDE, None  # Wide SL for continuation
            else:
                return ACTION_SHORT_WIDE, None
        return ACTION_NOTHING, None

    def reset(self):
        self.entered = False
        self.step_count = 0


def evaluate_agent(agent, env, n_episodes, agent_name, deterministic=True):
    """Run agent through n episodes and collect metrics."""
    rewards = []
    trades_per_day = []
    trade_details = []

    for i in range(n_episodes):
        if i % 200 == 0 and i > 0:
            print(f"    ... {i}/{n_episodes}", file=sys.stderr)
        obs, _ = env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()

        episode_reward = 0
        done = False
        step = 0

        while not done:
            if isinstance(agent, PPO):
                action, _ = agent.predict(obs, deterministic=deterministic)
            else:
                action, _ = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        rewards.append(episode_reward)
        trades_per_day.append(info.get('trade_count', 0))

    rewards = np.array(rewards)
    trades = np.array(trades_per_day)

    # Metrics
    mean_rew = rewards.mean()
    std_rew = rewards.std()
    sharpe = mean_rew / (std_rew + 1e-8)
    trade_days = (trades > 0).sum() / len(trades) * 100
    mean_trades = trades[trades > 0].mean() if (trades > 0).any() else 0

    # Bootstrap CI
    n_boot = 1000
    boot_means = np.array([np.random.choice(rewards, len(rewards), replace=True).mean()
                           for _ in range(n_boot)])
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    # Drawdown
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = drawdown.max()

    result = {
        'agent': agent_name,
        'n_episodes': n_episodes,
        'mean_reward': mean_rew,
        'std_reward': std_rew,
        'sharpe': sharpe,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'trade_days_pct': trade_days,
        'mean_trades_per_day': mean_trades,
        'max_drawdown': max_dd,
        'total_cum_reward': rewards.sum(),
        'win_rate': (rewards > 0).sum() / max(1, (trades > 0).sum()) * 100,
    }
    return result, rewards


def quarterly_stability(rewards, episodes_df, split):
    """Check if agent is profitable in each quarter."""
    dates = episodes_df[episodes_df['split'] == split]['date'].values
    # Group by quarter
    quarters = {}
    for i, d in enumerate(dates[:len(rewards)]):
        q = d[:4] + '-Q' + str((int(d[5:7]) - 1) // 3 + 1)
        if q not in quarters:
            quarters[q] = []
        quarters[q].append(rewards[i])

    results = {}
    for q, r in sorted(quarters.items()):
        r = np.array(r)
        results[q] = {
            'n': len(r),
            'mean': r.mean(),
            'sharpe': r.mean() / (r.std() + 1e-8),
            'positive_pct': (r > 0).sum() / len(r) * 100,
        }
    return results


def main():
    print("=" * 60, file=sys.stderr)
    print("DURCHLAUF 5.0 — EVALUATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load data
    episodes = pd.read_parquet(EPISODES_PATH)
    print(f"Loaded {len(episodes)} episodes", file=sys.stderr)

    # Check if RL model exists
    has_rl = MODEL_PATH.exists()
    if has_rl:
        print(f"RL model found: {MODEL_PATH}", file=sys.stderr)
    else:
        print(f"WARNING: No RL model at {MODEL_PATH}", file=sys.stderr)
        print("Running baselines only.", file=sys.stderr)

    results = []
    all_rewards = {}

    # Evaluate on validation set (skip train to save time — 1248 vs 3006 episodes)
    for split_name in ['val']:
        print(f"\n--- Evaluating on {split_name} ({(episodes['split']==split_name).sum()} episodes) ---",
              file=sys.stderr)
        env = make_env(episodes, split=split_name, feature_dropout=0.0, noise_std=0.0)
        n_eps = env.env.n_episodes

        # Baselines
        baselines = [
            ('B1_Random', RandomAgent()),
            ('B2_AlwaysLong', AlwaysLongAgent()),
            ('B3_WithGap', WithGapAgent()),
            ('B4_ODFollower', ODFollowerAgent()),
            ('B5_PMRTHRatio', PMRTHRatioAgent()),
        ]

        for name, agent in baselines:
            print(f"  {name} ({n_eps} episodes)...", file=sys.stderr)
            r, rews = evaluate_agent(agent, env, n_eps, f'{name}_{split_name}')
            r['split'] = split_name
            results.append(r)
            all_rewards[f'{name}_{split_name}'] = rews

        # RL Agent
        if has_rl:
            print(f"  RL Agent ({n_eps} episodes)...", file=sys.stderr)
            rl_model = PPO.load(MODEL_PATH)
            r, rews = evaluate_agent(rl_model, env, n_eps, f'RL_Agent_{split_name}')
            r['split'] = split_name
            results.append(r)
            all_rewards[f'RL_Agent_{split_name}'] = rews

    # Write results
    output_path = BASE_DIR / 'results' / 'rl_evaluation.txt'
    with open(output_path, 'w') as f:
        f.write("DURCHLAUF 5.0 — RL EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Results table
        f.write(f"{'Agent':<25} {'Split':<6} {'N':>5} {'MeanRew':>8} {'Sharpe':>7} "
                f"{'CI_Low':>7} {'CI_Hi':>7} {'TrdDays%':>8} {'MaxDD':>7} {'WR%':>6}\n")
        f.write("-" * 95 + "\n")

        for r in results:
            f.write(f"{r['agent']:<25} {r.get('split',''):>6} {r['n_episodes']:>5} "
                    f"{r['mean_reward']:>8.4f} {r['sharpe']:>7.3f} "
                    f"{r['ci_low']:>7.4f} {r['ci_high']:>7.4f} "
                    f"{r['trade_days_pct']:>7.1f}% {r['max_drawdown']:>7.3f} "
                    f"{r['win_rate']:>5.1f}%\n")

        # RL vs Baselines comparison
        if has_rl:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("RL vs BASELINES (Validation)\n")
            f.write("-" * 80 + "\n")

            rl_val = [r for r in results if 'RL_Agent' in r['agent'] and r.get('split') == 'val']
            baseline_val = [r for r in results if 'RL_Agent' not in r['agent'] and r.get('split') == 'val']

            if rl_val:
                rl = rl_val[0]
                for b in baseline_val:
                    delta = rl['mean_reward'] - b['mean_reward']
                    beats = 'YES' if delta > 0 else 'NO'
                    f.write(f"  RL vs {b['agent']:<25}: Delta={delta:>+.4f} ADR  "
                            f"Beats: {beats}\n")

            # Quarterly stability
            f.write("\n\nQUARTERLY STABILITY (Validation)\n")
            f.write("-" * 60 + "\n")
            for key, rews in all_rewards.items():
                if 'val' in key:
                    qs = quarterly_stability(rews, episodes, 'val')
                    f.write(f"\n{key}:\n")
                    for q, s in qs.items():
                        f.write(f"  {q}: N={s['n']:>4}, Mean={s['mean']:>+.4f}, "
                                f"Sharpe={s['sharpe']:>.3f}, "
                                f"Positive={s['positive_pct']:>.1f}%\n")

        # Overfitting check
        f.write("\n\nOVERFITTING CHECK (Train Sharpe vs Val Sharpe)\n")
        f.write("-" * 60 + "\n")
        for name in set(r['agent'].rsplit('_', 1)[0] for r in results):
            train_r = [r for r in results if name in r['agent'] and r.get('split') == 'train']
            val_r = [r for r in results if name in r['agent'] and r.get('split') == 'val']
            if train_r and val_r:
                ratio = val_r[0]['sharpe'] / (train_r[0]['sharpe'] + 1e-8)
                f.write(f"  {name:<25}: Train Sharpe={train_r[0]['sharpe']:>.3f}, "
                        f"Val Sharpe={val_r[0]['sharpe']:>.3f}, "
                        f"Ratio={ratio:>.3f}\n")

    print(f"\nResults saved to {output_path}", file=sys.stderr)

    # Also run baselines without RL if model not found
    if not has_rl:
        print("\nTo evaluate RL agent, first train with:", file=sys.stderr)
        print("  .\\gapper_env\\Scripts\\python.exe scripts\\rl_train.py", file=sys.stderr)
        print("Then re-run this script.", file=sys.stderr)


if __name__ == '__main__':
    main()
