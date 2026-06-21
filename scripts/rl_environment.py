"""
Durchlauf 5.0 — RL ARCHITECT
GapperEnv: Gym environment for learning when/how to trade gapper days.
"""
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT

# Decision points: every 5 min from 9:35 to 10:55 = 17 points
# In 1-min bar indices: bar 5 (9:35), bar 10 (9:40), ..., bar 85 (10:55)
DECISION_BARS = list(range(5, 90, 5))  # [5, 10, 15, ..., 85] = 17 points
TIMEOUT_BAR = 90  # 11:00 = bar 90 (0-indexed from 9:30)

# Actions
ACTION_NOTHING = 0
ACTION_LONG_TIGHT = 1   # SL = 0.15 ADR
ACTION_LONG_MED = 2     # SL = 0.30 ADR
ACTION_LONG_WIDE = 3    # SL = 0.50 ADR
ACTION_SHORT_TIGHT = 4  # SL = 0.15 ADR
ACTION_SHORT_MED = 5    # SL = 0.30 ADR
ACTION_SHORT_WIDE = 6   # SL = 0.50 ADR
ACTION_CLOSE = 7

SL_SIZES = {
    ACTION_LONG_TIGHT: 0.15, ACTION_LONG_MED: 0.30, ACTION_LONG_WIDE: 0.50,
    ACTION_SHORT_TIGHT: 0.15, ACTION_SHORT_MED: 0.30, ACTION_SHORT_WIDE: 0.50,
}

N_ACTIONS = 8
N_FEATURES = 19

# Costs
SLIPPAGE_ADR = 0.015  # Entry + Exit
COMMISSION_ADR = 0.005
TOTAL_COST_ADR = SLIPPAGE_ADR + COMMISSION_ADR

MAX_TRADES_PER_DAY = 3


class GapperEnv(gym.Env):
    """
    RL Environment for Gapper trading.
    Episode = 1 Gapper day.
    Observations: 19 features (pre-open + post-open).
    Actions: 8 discrete (nothing, long x3 SL, short x3 SL, close).
    """
    metadata = {'render_modes': []}

    def __init__(self, episodes_df, split='train', feature_dropout=0.0,
                 noise_std=0.0, shuffle=True):
        super().__init__()

        self.episodes_df = episodes_df[episodes_df['split'] == split].reset_index(drop=True)
        self.split = split
        self.feature_dropout = feature_dropout
        self.noise_std = noise_std
        self.shuffle = shuffle
        self.n_episodes = len(self.episodes_df)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(N_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state
        self.current_episode_idx = 0
        self.episode_order = np.arange(self.n_episodes)
        if self.shuffle:
            np.random.shuffle(self.episode_order)

        # Internal state (set in reset)
        self.rth_bars = None
        self.vwap_bars = None
        self.adr = 0.0
        self.gap_dir = 1.0
        self.pre_open_features = None
        self.decision_step = 0
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.trade_count = 0
        self.episode_reward = 0.0
        self.pm_rth30_ratio = 0.0
        self.spy_return = 0.0
        self.profitable_trades = 0
        self.total_trades_done = 0

        # Preload baseline volumes for RVOL (will be lazy-loaded per ticker)
        self._baseline_cache = {}

    def _load_episode_data(self, ep_row):
        """Load 1-min bars and VWAP for this episode."""
        ticker = ep_row['ticker']
        date = ep_row['date']
        self._current_ticker = ticker
        self.adr = ep_row['adr']
        self.gap_dir = ep_row['gap_direction_num']
        self.spy_return = ep_row.get('spy_return_day', 0.0)
        self.pm_rth30_ratio = ep_row.get('pm_rth30_ratio_val', 0.0)

        # Load raw 1-min
        raw_path = BASE_DIR / 'data' / 'raw_1min' / ticker / f'{date}.parquet'
        raw_df = pd.read_parquet(raw_path)
        rth = raw_df[raw_df['session'] == 'rth'].reset_index(drop=True)
        self.rth_bars = rth

        # Load VWAP
        vwap_path = BASE_DIR / 'data' / 'vwap' / ticker / f'{date}.parquet'
        if vwap_path.exists():
            self.vwap_bars = pd.read_parquet(vwap_path).reset_index(drop=True)
        else:
            self.vwap_bars = None

        # Pre-open features from episode data
        # prior returns are stored as percentages (e.g. 3.4 = 3.4%), convert to decimal
        prior_10d = ep_row.get('prior_10d_return', 0.0)
        prior_30d = ep_row.get('prior_30d_return', 0.0)
        if abs(prior_10d) > 5:  # Likely percentage, not decimal
            prior_10d /= 100.0
        if abs(prior_30d) > 5:
            prior_30d /= 100.0

        self.pre_open_features = np.array([
            ep_row['gap_direction_num'],
            np.clip(ep_row.get('gap_size_adr', 0.0), 0, 5.0),  # clip extreme gaps
            np.clip(ep_row.get('adr_pct', 0.05), 0, 0.25),  # clip ADR%
            np.clip(prior_10d, -0.5, 0.5),
            np.clip(prior_30d, -1.0, 1.0),
            ep_row.get('rsi_14', 0.5),
            np.clip(ep_row.get('pm_volume_ratio', 1.0), 0, 10.0),
            np.clip(ep_row.get('pm_range_adr', 0.5), 0, 3.0),
            ep_row.get('has_earnings', 0.0),
            ep_row.get('earnings_score', 0.0),
        ], dtype=np.float32)

        # Load baseline for RVOL
        self._load_baseline(ticker, date)

    def _load_baseline(self, ticker, date):
        """Load baseline volume data for RVOL computation."""
        if ticker in self._baseline_cache:
            return

        base_dir = BASE_DIR / 'data' / 'baseline_1min' / ticker
        if not base_dir.exists():
            self._baseline_cache[ticker] = None
            return

        try:
            files = sorted([f for f in os.listdir(base_dir) if f.endswith('.parquet')])
            # Take up to 10 recent files before current date
            files_before = [f for f in files if f.replace('.parquet', '') < date][-10:]
            if not files_before:
                self._baseline_cache[ticker] = None
                return

            cum_vols = []
            for f in files_before:
                df = pd.read_parquet(base_dir / f)
                if len(df) > 0:
                    cum_vols.append(df['volume'].cumsum().values)

            if cum_vols:
                # Pad to same length
                max_len = max(len(v) for v in cum_vols)
                padded = np.full((len(cum_vols), max_len), np.nan)
                for i, v in enumerate(cum_vols):
                    padded[i, :len(v)] = v
                self._baseline_cache[ticker] = np.nanmean(padded, axis=0)
            else:
                self._baseline_cache[ticker] = None
        except Exception:
            self._baseline_cache[ticker] = None

    def _compute_rvol(self, bar_idx):
        """Compute RVOL at current bar index."""
        ticker = getattr(self, '_current_ticker', '')
        baseline = self._baseline_cache.get(ticker)

        if baseline is None or bar_idx >= len(baseline) or baseline[bar_idx] <= 0:
            return 1.0

        cum_vol = self.rth_bars.iloc[:bar_idx + 1]['volume'].sum()
        return min(10.0, cum_vol / baseline[bar_idx])

    def _compute_state(self, bar_idx):
        """Compute full 19-feature state at given bar index."""
        rth = self.rth_bars
        adr = self.adr

        if bar_idx >= len(rth):
            bar_idx = len(rth) - 1

        bars = rth.iloc[:bar_idx + 1]
        open_price = rth.iloc[0]['open']
        current_close = bars.iloc[-1]['close']
        current_high = bars['high'].max()
        current_low = bars['low'].min()

        # Post-open features
        # 11. od_strength_adr
        od_strength = abs(current_close - open_price) / adr if adr > 0 else 0.0

        # 12. od_with_gap
        od_dir = 1.0 if current_close > open_price else -1.0
        od_with_gap = od_dir * self.gap_dir

        # 13. rvol_at_time
        rvol = self._compute_rvol(bar_idx)

        # 14. vwap_position (z-score)
        vwap_pos = 0.0
        if self.vwap_bars is not None and bar_idx < len(self.vwap_bars):
            vb = self.vwap_bars.iloc[bar_idx]
            vwap_val = vb.get('vwap', np.nan)
            std_val = vb.get('std_dev', np.nan)
            if pd.notna(vwap_val) and pd.notna(std_val) and std_val > 0:
                vwap_pos = np.clip((current_close - vwap_val) / std_val, -3, 3)

        # 15. pct_adr_used
        pct_adr = (current_high - current_low) / adr if adr > 0 else 0.0

        # 16. vwap_broken (3+ closes on same side)
        vwap_broken = 0.0
        if self.vwap_bars is not None and bar_idx >= 2:
            try:
                recent_closes = bars.iloc[-3:]['close'].values
                vwap_vals = []
                for j in range(max(0, bar_idx - 2), bar_idx + 1):
                    if j < len(self.vwap_bars):
                        vwap_vals.append(self.vwap_bars.iloc[j].get('vwap', np.nan))
                if len(vwap_vals) == 3 and all(pd.notna(v) for v in vwap_vals):
                    if all(c > v for c, v in zip(recent_closes, vwap_vals)):
                        vwap_broken = 1.0
                    elif all(c < v for c, v in zip(recent_closes, vwap_vals)):
                        vwap_broken = 1.0
            except Exception:
                pass

        # 17. pm_rth30_ratio (only valid after bar 30)
        pm_rth30 = self.pm_rth30_ratio if bar_idx >= 30 else 0.0

        # 18. vwap_test_count
        vwap_tests = 0.0
        if self.vwap_bars is not None:
            last_test = -6
            for j in range(min(bar_idx + 1, len(self.vwap_bars))):
                if j < len(bars) and j < len(self.vwap_bars):
                    bh = bars.iloc[j]['high']
                    bl = bars.iloc[j]['low']
                    vw = self.vwap_bars.iloc[j].get('vwap', np.nan)
                    if pd.notna(vw) and bl <= vw <= bh and j - last_test >= 5:
                        vwap_tests += 1
                        last_test = j

        # 19. spy_return_current (use daily as proxy)
        spy_ret = self.spy_return

        # Assemble state
        state = np.concatenate([
            self.pre_open_features,  # features 1-10
            np.array([
                min(od_strength, 3.0),           # 11
                od_with_gap,                      # 12
                min(rvol, 10.0),                  # 13
                vwap_pos,                         # 14
                min(pct_adr, 3.0),                # 15
                vwap_broken,                      # 16
                min(pm_rth30, 5.0),               # 17
                min(vwap_tests, 10.0),            # 18
                np.clip(spy_ret, -0.05, 0.05),    # 19
            ], dtype=np.float32)
        ])

        # Feature dropout (training only)
        if self.feature_dropout > 0 and self.split == 'train':
            mask = np.random.random(N_FEATURES) > self.feature_dropout
            state = state * mask

        # Noise injection (training only)
        if self.noise_std > 0 and self.split == 'train':
            state = state + np.random.normal(0, self.noise_std, N_FEATURES).astype(np.float32)

        return state

    def _check_sl_between_bars(self, from_bar, to_bar):
        """
        Check if SL was triggered between two decision points.
        Returns (triggered: bool, trigger_bar: int, trigger_price: float).
        """
        if self.position is None:
            return False, -1, 0.0

        rth = self.rth_bars
        for bar_i in range(from_bar, min(to_bar + 1, len(rth))):
            bar = rth.iloc[bar_i]
            if self.position == 'long':
                if bar['low'] <= self.sl_price:
                    return True, bar_i, self.sl_price
            elif self.position == 'short':
                if bar['high'] >= self.sl_price:
                    return True, bar_i, self.sl_price

        return False, -1, 0.0

    def _close_position(self, exit_price):
        """Close current position and compute reward."""
        if self.position is None:
            return 0.0

        direction = 1.0 if self.position == 'long' else -1.0
        pnl_adr = direction * (exit_price - self.entry_price) / self.adr
        reward = pnl_adr - TOTAL_COST_ADR

        if reward > 0:
            self.profitable_trades += 1
        self.total_trades_done += 1
        self.trade_count += 1

        self.position = None
        self.entry_price = 0.0
        self.sl_price = 0.0

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.current_episode_idx >= self.n_episodes:
            self.current_episode_idx = 0
            if self.shuffle:
                np.random.shuffle(self.episode_order)

        ep_idx = self.episode_order[self.current_episode_idx]
        ep_row = self.episodes_df.iloc[ep_idx]
        self._load_episode_data(ep_row)

        self.current_episode_idx += 1
        self.decision_step = 0
        self.position = None
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.trade_count = 0
        self.episode_reward = 0.0
        self.profitable_trades = 0
        self.total_trades_done = 0

        # Initial state at first decision point (bar 5 = 9:35)
        state = self._compute_state(DECISION_BARS[0])
        return state, {}

    def step(self, action):
        # Ensure action is a plain int (sb3 may pass numpy array)
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        if self.decision_step >= len(DECISION_BARS):
            return self._compute_state(TIMEOUT_BAR - 1), 0.0, True, False, {}

        current_bar = DECISION_BARS[self.decision_step]
        reward = 0.0
        done = False
        info = {}

        rth = self.rth_bars
        if current_bar >= len(rth):
            # Not enough data, end episode
            if self.position:
                exit_price = rth.iloc[-1]['close']
                reward += self._close_position(exit_price)
            done = True
            state = self._compute_state(len(rth) - 1)
            info['reason'] = 'no_data'
            self.episode_reward += reward
            return state, reward, done, False, info

        current_price = rth.iloc[current_bar]['close']

        # Execute action
        if action == ACTION_CLOSE and self.position is not None:
            # Close position
            reward += self._close_position(current_price)

        elif action in SL_SIZES and self.position is None and self.trade_count < MAX_TRADES_PER_DAY:
            # Open new position
            sl_dist = SL_SIZES[action] * self.adr
            is_long = action in (ACTION_LONG_TIGHT, ACTION_LONG_MED, ACTION_LONG_WIDE)

            self.position = 'long' if is_long else 'short'
            self.entry_price = current_price
            if is_long:
                self.sl_price = current_price - sl_dist
            else:
                self.sl_price = current_price + sl_dist

        # Move to next decision point
        self.decision_step += 1

        if self.decision_step < len(DECISION_BARS):
            next_bar = DECISION_BARS[self.decision_step]
        else:
            next_bar = TIMEOUT_BAR

        # Check SL between current and next decision point
        if self.position is not None:
            sl_hit, sl_bar, sl_price = self._check_sl_between_bars(current_bar + 1, next_bar)
            if sl_hit:
                reward += self._close_position(sl_price)

        # Check if done
        if self.decision_step >= len(DECISION_BARS) or self.trade_count >= MAX_TRADES_PER_DAY:
            # Timeout or max trades reached
            if self.position is not None:
                # Close at timeout
                timeout_bar = min(TIMEOUT_BAR, len(rth) - 1)
                exit_price = rth.iloc[timeout_bar]['close']
                reward += self._close_position(exit_price)
            done = True
            info['reason'] = 'timeout' if self.decision_step >= len(DECISION_BARS) else 'max_trades'

        # Efficiency bonus
        if done and self.total_trades_done > 0:
            eff = self.profitable_trades / self.total_trades_done
            reward += 0.005 * eff

        self.episode_reward += reward

        # Compute next state
        if not done:
            state = self._compute_state(DECISION_BARS[self.decision_step])
        else:
            state = self._compute_state(min(TIMEOUT_BAR, len(rth) - 1))

        info['episode_reward'] = self.episode_reward
        info['trade_count'] = self.trade_count
        info['position'] = self.position

        return state, reward, done, False, info


class GapperEnvWrapper(gym.Wrapper):
    """
    Wrapper that handles epoch tracking and min trade frequency enforcement.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_trades = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done:
            self.episode_rewards.append(info.get('episode_reward', 0))
            self.episode_trades.append(info.get('trade_count', 0))
        return obs, reward, done, truncated, info

    def get_stats(self):
        if not self.episode_rewards:
            return {}
        rewards = np.array(self.episode_rewards)
        trades = np.array(self.episode_trades)
        trade_days = (trades > 0).sum()
        return {
            'mean_reward': rewards.mean(),
            'std_reward': rewards.std(),
            'sharpe': rewards.mean() / (rewards.std() + 1e-8),
            'trade_days_pct': trade_days / len(trades) * 100,
            'mean_trades_per_day': trades.mean(),
            'total_episodes': len(rewards),
        }

    def reset_stats(self):
        self.episode_rewards = []
        self.episode_trades = []


def make_env(episodes_df, split='train', feature_dropout=0.2, noise_std=0.05):
    """Factory function for creating wrapped environment."""
    dropout = feature_dropout if split == 'train' else 0.0
    noise = noise_std if split == 'train' else 0.0
    env = GapperEnv(episodes_df, split=split, feature_dropout=dropout,
                    noise_std=noise, shuffle=(split == 'train'))
    return GapperEnvWrapper(env)


if __name__ == '__main__':
    # Quick smoke test
    import sys

    episodes = pd.read_parquet(BASE_DIR / 'data' / 'metadata' / 'rl_episodes.parquet')
    print(f"Loaded {len(episodes)} episodes", file=sys.stderr)

    env = make_env(episodes, split='train', feature_dropout=0.0, noise_std=0.0)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}", file=sys.stderr)
    print(f"Initial obs: {obs}", file=sys.stderr)

    # Run one episode
    total_reward = 0
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Episode done: steps={steps}, reward={total_reward:.4f}, "
          f"trades={info.get('trade_count', 0)}", file=sys.stderr)
    print("Smoke test PASSED!", file=sys.stderr)
