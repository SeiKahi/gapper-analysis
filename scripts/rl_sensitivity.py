"""
Durchlauf 5.0 — EVALUATOR: Parameter Sensitivity Analysis
Feature importance, partial dependence, ablation, parameter sweeps.
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
                            N_FEATURES, N_ACTIONS, DECISION_BARS)

BASE_DIR = PROJECT_ROOT
os.chdir(BASE_DIR)

EPISODES_PATH = BASE_DIR / 'data' / 'metadata' / 'rl_episodes.parquet'
MODEL_PATH = BASE_DIR / 'models' / 'rl_agent_v1.zip'

FEATURE_NAMES = [
    'gap_direction',      # 0
    'gap_size_adr',       # 1
    'adr_pct',            # 2
    'prior_10d_return',   # 3
    'prior_30d_return',   # 4
    'rsi_14',             # 5
    'pm_volume_ratio',    # 6
    'pm_range_adr',       # 7
    'has_earnings',       # 8
    'earnings_score',     # 9
    'od_strength_adr',    # 10
    'od_with_gap',        # 11
    'rvol_at_time',       # 12
    'vwap_position',      # 13
    'pct_adr_used',       # 14
    'vwap_broken',        # 15
    'pm_rth30_ratio',     # 16
    'vwap_test_count',    # 17
    'spy_return_current', # 18
]


def collect_episodes(model, env, n_episodes):
    """Run model on env and collect obs/actions/rewards per episode."""
    episodes_data = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_obs.append(obs.copy())
            next_obs, reward, done, truncated, info = env.step(action)
            ep_actions.append(int(action))
            ep_rewards.append(reward)
            obs = next_obs
        episodes_data.append({
            'obs': np.array(ep_obs),
            'actions': np.array(ep_actions),
            'rewards': np.array(ep_rewards),
            'total_reward': sum(ep_rewards),
        })
    return episodes_data


def permutation_importance(model, env, n_episodes, n_repeats=5):
    """
    For each feature: shuffle it, measure performance drop.
    """
    print("Computing permutation importance...", file=sys.stderr)

    # Baseline performance
    baseline_data = collect_episodes(model, env, n_episodes)
    baseline_reward = np.mean([e['total_reward'] for e in baseline_data])

    importance = {}
    for feat_idx in range(N_FEATURES):
        drops = []
        for _ in range(n_repeats):
            # Create modified env that shuffles this feature
            total_reward = 0
            count = 0
            for _ in range(n_episodes):
                obs, _ = env.reset()
                done = False
                ep_reward = 0
                while not done:
                    # Shuffle this feature
                    modified_obs = obs.copy()
                    modified_obs[feat_idx] = np.random.normal(0, 1)
                    action, _ = model.predict(modified_obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    ep_reward += reward
                total_reward += ep_reward
                count += 1

            perm_reward = total_reward / count
            drops.append(baseline_reward - perm_reward)

        importance[FEATURE_NAMES[feat_idx]] = {
            'mean_drop': np.mean(drops),
            'std_drop': np.std(drops),
            'idx': feat_idx,
        }

    return importance, baseline_reward


def partial_dependence(model, env, n_episodes, feature_idx, values):
    """
    For a feature: vary it while keeping other features from real episodes.
    Returns mean action probabilities and rewards at each value.
    """
    # Collect real observations
    data = collect_episodes(model, env, min(n_episodes, 200))
    all_obs = np.concatenate([e['obs'] for e in data], axis=0)

    results = {}
    for val in values:
        actions = []
        for obs in all_obs[:500]:  # Sample 500 obs
            modified = obs.copy()
            modified[feature_idx] = val
            action, _ = model.predict(modified, deterministic=True)
            actions.append(int(action))

        actions = np.array(actions)
        action_dist = np.bincount(actions, minlength=N_ACTIONS) / len(actions)
        results[val] = {
            'action_dist': action_dist,
            'pct_trade': 1 - action_dist[0],  # % not doing nothing
            'pct_long': action_dist[1] + action_dist[2] + action_dist[3],
            'pct_short': action_dist[4] + action_dist[5] + action_dist[6],
            'pct_close': action_dist[7],
        }

    return results


def regime_analysis(model, env, episodes_df, split='val'):
    """Quarterly performance breakdown."""
    dates = episodes_df[episodes_df['split'] == split]['date'].values
    env_inner = env.env if hasattr(env, 'env') else env

    rewards_by_quarter = {}
    obs, _ = env.reset()

    for i in range(env_inner.n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        if i < len(dates):
            d = dates[i]
            q = d[:4] + '-Q' + str((int(d[5:7]) - 1) // 3 + 1)
            if q not in rewards_by_quarter:
                rewards_by_quarter[q] = []
            rewards_by_quarter[q].append(ep_reward)

    return rewards_by_quarter


def main():
    print("=" * 60, file=sys.stderr)
    print("DURCHLAUF 5.0 — SENSITIVITY ANALYSIS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if not MODEL_PATH.exists():
        print(f"ERROR: No model at {MODEL_PATH}", file=sys.stderr)
        print("Train first with: .\\gapper_env\\Scripts\\python.exe scripts\\rl_train.py",
              file=sys.stderr)
        return

    # Load
    episodes = pd.read_parquet(EPISODES_PATH)
    model = PPO.load(MODEL_PATH)
    val_env = make_env(episodes, split='val', feature_dropout=0.0, noise_std=0.0)
    n_val = val_env.env.n_episodes

    output = []
    output.append("DURCHLAUF 5.0 — SENSITIVITY ANALYSIS")
    output.append("=" * 70)
    output.append(f"Timestamp: {datetime.now().isoformat()}")
    output.append(f"Model: {MODEL_PATH}")
    output.append(f"Val episodes: {n_val}")

    # 1. Permutation Importance (use small N to keep runtime reasonable)
    print("\n1. Permutation Importance...", file=sys.stderr)
    importance, baseline = permutation_importance(model, val_env, min(n_val, 100), n_repeats=2)

    output.append("\n\n1. PERMUTATION IMPORTANCE (Val)")
    output.append("-" * 50)
    output.append(f"Baseline reward: {baseline:.4f}")
    output.append(f"\n{'Feature':<25} {'Mean Drop':>10} {'Std':>8} {'Rank':>5}")
    output.append("-" * 50)

    sorted_imp = sorted(importance.items(), key=lambda x: -x[1]['mean_drop'])
    for rank, (name, v) in enumerate(sorted_imp, 1):
        output.append(f"{name:<25} {v['mean_drop']:>+10.4f} {v['std_drop']:>8.4f} #{rank}")

    # 2. Partial Dependence for top 5 features
    print("\n2. Partial Dependence Plots...", file=sys.stderr)
    top5 = [item[0] for item in sorted_imp[:5]]
    top5_idx = [importance[name]['idx'] for name in top5]

    output.append("\n\n2. PARTIAL DEPENDENCE (Top 5 Features)")
    output.append("-" * 70)

    feature_ranges = {
        'gap_direction': [-1, 1],
        'gap_size_adr': [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        'adr_pct': [0.02, 0.05, 0.08, 0.12, 0.18],
        'prior_10d_return': [-0.3, -0.1, 0.0, 0.1, 0.3],
        'prior_30d_return': [-0.5, -0.2, 0.0, 0.2, 0.5],
        'rsi_14': [0.2, 0.35, 0.5, 0.65, 0.8],
        'pm_volume_ratio': [0.2, 0.5, 1.0, 2.0, 5.0],
        'pm_range_adr': [0.1, 0.3, 0.5, 1.0, 2.0],
        'has_earnings': [0, 1],
        'earnings_score': [-1, 0, 1],
        'od_strength_adr': [0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
        'od_with_gap': [-1, 0, 1],
        'rvol_at_time': [0.5, 1.0, 2.0, 3.0, 5.0],
        'vwap_position': [-2, -1, 0, 1, 2],
        'pct_adr_used': [0.1, 0.3, 0.5, 0.75, 1.0, 1.5],
        'vwap_broken': [0, 1],
        'pm_rth30_ratio': [0.0, 0.05, 0.1, 0.3, 0.5, 1.0],
        'vwap_test_count': [0, 1, 2, 3, 5],
        'spy_return_current': [-0.03, -0.01, 0, 0.01, 0.03],
    }

    for feat_name, feat_idx in zip(top5, top5_idx):
        values = feature_ranges.get(feat_name, [-1, 0, 1])
        pd_result = partial_dependence(model, val_env, min(n_val, 80), feat_idx, values)

        output.append(f"\n  {feat_name} (idx={feat_idx}):")
        output.append(f"  {'Value':>8} {'%Trade':>8} {'%Long':>8} {'%Short':>8} {'%Close':>8}")
        for val, r in pd_result.items():
            output.append(f"  {val:>8.2f} {r['pct_trade']*100:>7.1f}% "
                         f"{r['pct_long']*100:>7.1f}% {r['pct_short']*100:>7.1f}% "
                         f"{r['pct_close']*100:>7.1f}%")

    # 3. Regime Analysis
    print("\n3. Regime Analysis...", file=sys.stderr)
    quarters = regime_analysis(model, val_env, episodes)

    output.append("\n\n3. REGIME ANALYSIS (Quarterly Val Performance)")
    output.append("-" * 60)
    output.append(f"{'Quarter':<12} {'N':>5} {'Mean':>8} {'Sharpe':>8} {'Pos%':>7}")
    for q, rews in sorted(quarters.items()):
        rews = np.array(rews)
        output.append(f"{q:<12} {len(rews):>5} {rews.mean():>+8.4f} "
                     f"{rews.mean()/(rews.std()+1e-8):>8.3f} "
                     f"{(rews>0).sum()/len(rews)*100:>6.1f}%")

    # 4. Action Distribution Analysis
    print("\n4. Action Distribution...", file=sys.stderr)
    data = collect_episodes(model, val_env, min(n_val, 150))
    all_actions = np.concatenate([e['actions'] for e in data])
    action_names = ['NOTHING', 'LONG_TIGHT', 'LONG_MED', 'LONG_WIDE',
                    'SHORT_TIGHT', 'SHORT_MED', 'SHORT_WIDE', 'CLOSE']

    output.append("\n\n4. ACTION DISTRIBUTION (Val)")
    output.append("-" * 40)
    action_counts = np.bincount(all_actions, minlength=N_ACTIONS)
    for i, name in enumerate(action_names):
        pct = action_counts[i] / len(all_actions) * 100
        output.append(f"  {name:<15}: {action_counts[i]:>6} ({pct:>5.1f}%)")

    # 5. Feature Statistics at Trade Decisions
    print("\n5. Feature Stats at Trade Points...", file=sys.stderr)
    trade_obs = []
    notrade_obs = []
    for ep in data:
        for obs, act in zip(ep['obs'], ep['actions']):
            if act > 0 and act < 7:  # Entry action
                trade_obs.append(obs)
            elif act == 0:
                notrade_obs.append(obs)

    if trade_obs and notrade_obs:
        trade_obs = np.array(trade_obs)
        notrade_obs = np.array(notrade_obs)

        output.append("\n\n5. FEATURE VALUES: TRADE vs NO-TRADE")
        output.append("-" * 70)
        output.append(f"  Trade decisions: {len(trade_obs)}, No-trade: {len(notrade_obs)}")
        output.append(f"\n  {'Feature':<25} {'Trade_Mean':>10} {'NoTrade_Mean':>12} {'Delta':>8}")
        for i, name in enumerate(FEATURE_NAMES):
            t_mean = trade_obs[:, i].mean()
            nt_mean = notrade_obs[:, i].mean()
            output.append(f"  {name:<25} {t_mean:>10.4f} {nt_mean:>12.4f} "
                         f"{t_mean - nt_mean:>+8.4f}")

    # Save
    output_path = BASE_DIR / 'results' / 'rl_sensitivity.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(output))

    print(f"\nResults saved to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
