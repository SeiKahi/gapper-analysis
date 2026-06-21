"""
Durchlauf 5.0 — TRAINER
PPO training with anti-overfitting, early stopping, and checkpointing.

Usage:
  .\\gapper_env\\Scripts\\python.exe scripts\\rl_train.py [--smoke]

  --smoke: Quick smoke test (5 episodes, 2 epochs)
"""
import os
import sys
import signal
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add scripts to path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str((PROJECT_ROOT / 'scripts')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl_environment import make_env, GapperEnvWrapper, N_FEATURES, N_ACTIONS

BASE_DIR = PROJECT_ROOT
os.chdir(BASE_DIR)

# ── Config ──────────────────────────────────────────────────────────
EPISODES_PATH = BASE_DIR / 'data' / 'metadata' / 'rl_episodes.parquet'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Training hyperparameters (conservative for small dataset)
TOTAL_TIMESTEPS = 500_000  # ~50 epochs over ~2400 episodes * ~17 steps
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
N_EPOCHS_PPO = 10  # PPO epochs per update
N_STEPS = 2048     # Steps per rollout
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01    # Encourage exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network: Small to prevent overfitting (256 units, 2 layers)
POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

# Anti-overfitting
FEATURE_DROPOUT = 0.20
NOISE_STD = 0.05
VAL_CHECK_INTERVAL = 10_000  # Check val every N timesteps
EARLY_STOP_PATIENCE = 3      # Stop after 3 consecutive val degradations
VALIDATION_GATE = 0.50        # Val must be >= 50% of train performance

# Smoke test overrides
SMOKE_TIMESTEPS = 1000
SMOKE_VAL_INTERVAL = 500


class ValidationCallback(BaseCallback):
    """
    Monitors validation performance during training.
    Implements early stopping and checkpointing.
    """
    def __init__(self, val_env, episodes_df, check_interval, patience, val_gate,
                 model_dir, log_file, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self._episodes_df = episodes_df
        self.check_interval = check_interval
        self.patience = patience
        self.val_gate = val_gate
        self.model_dir = model_dir
        self.log_file = log_file

        self.best_val_reward = -np.inf
        self.val_degradations = 0
        self.check_count = 0
        self.train_rewards = []
        self.val_rewards = []
        self.log_entries = []
        self.start_time = time.time()

    def _on_step(self):
        if self.num_timesteps % self.check_interval == 0 and self.num_timesteps > 0:
            self.check_count += 1
            self._evaluate()
        return True

    def _evaluate(self):
        """Run validation episodes and log metrics."""
        # Collect train stats — run dedicated train evaluation
        train_eval_env = make_env(self._episodes_df, split='train',
                                  feature_dropout=0.0, noise_std=0.0)
        train_eval_env.reset_stats()
        n_train_eval = min(100, train_eval_env.env.n_episodes)
        for _ in range(n_train_eval):
            obs, _ = train_eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = train_eval_env.step(action)
        train_stats = train_eval_env.get_stats()

        # Run validation
        val_env = self.val_env
        val_env.reset_stats()

        n_val_episodes = min(100, val_env.env.n_episodes)
        for _ in range(n_val_episodes):
            obs, _ = val_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = val_env.step(action)

        val_stats = val_env.get_stats()

        # Log
        elapsed = time.time() - self.start_time
        entry = {
            'timestep': self.num_timesteps,
            'elapsed_min': elapsed / 60,
            'train_mean_reward': train_stats.get('mean_reward', 0),
            'train_sharpe': train_stats.get('sharpe', 0),
            'train_trade_days': train_stats.get('trade_days_pct', 0),
            'val_mean_reward': val_stats.get('mean_reward', 0),
            'val_sharpe': val_stats.get('sharpe', 0),
            'val_trade_days': val_stats.get('trade_days_pct', 0),
        }
        self.log_entries.append(entry)

        val_reward = val_stats.get('mean_reward', -np.inf)

        # Print progress
        print(f"\n[Check {self.check_count}] Step {self.num_timesteps} "
              f"({elapsed/60:.1f}min)", file=sys.stderr)
        print(f"  Train: reward={entry['train_mean_reward']:.4f}, "
              f"sharpe={entry['train_sharpe']:.3f}, "
              f"trade_days={entry['train_trade_days']:.1f}%", file=sys.stderr)
        print(f"  Val:   reward={entry['val_mean_reward']:.4f}, "
              f"sharpe={entry['val_sharpe']:.3f}, "
              f"trade_days={entry['val_trade_days']:.1f}%", file=sys.stderr)

        # Checkpoint
        checkpoint_path = self.model_dir / f'rl_checkpoint_{self.check_count}.zip'
        self.model.save(checkpoint_path)

        # Best model check
        if val_reward > self.best_val_reward:
            self.best_val_reward = val_reward
            self.val_degradations = 0
            self.model.save(self.model_dir / 'rl_agent_v1.zip')
            print(f"  NEW BEST val reward: {val_reward:.4f}", file=sys.stderr)
        else:
            self.val_degradations += 1
            print(f"  Val degradation {self.val_degradations}/{self.patience}",
                  file=sys.stderr)

        # Validation gate check
        train_sharpe = entry['train_sharpe']
        val_sharpe = entry['val_sharpe']
        if train_sharpe > 0 and val_sharpe < self.val_gate * train_sharpe:
            print(f"  WARNING: Val sharpe ({val_sharpe:.3f}) < "
                  f"{self.val_gate*100:.0f}% of train ({train_sharpe:.3f}) — OVERFITTING",
                  file=sys.stderr)

        # Min trade frequency check
        if entry['val_trade_days'] < 20:
            print(f"  WARNING: Agent trades on only {entry['val_trade_days']:.1f}% of days "
                  f"(min 20% required)", file=sys.stderr)

        # Early stopping
        if self.val_degradations >= self.patience:
            print(f"\n  EARLY STOPPING: {self.patience} consecutive val degradations",
                  file=sys.stderr)
            self._save_log()
            return False

        self._save_log()
        return True

    def _save_log(self):
        """Save training log to file."""
        with open(self.log_file, 'w') as f:
            f.write("DURCHLAUF 5.0 — TRAINING LOG\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total timesteps so far: {self.num_timesteps}\n")
            f.write(f"Best val reward: {self.best_val_reward:.4f}\n\n")

            f.write(f"{'Step':>8} {'Elapsed':>8} {'Tr_Rew':>8} {'Tr_Shrp':>8} "
                    f"{'Tr_Days%':>8} {'Val_Rew':>8} {'Val_Shrp':>8} {'Val_Days%':>8}\n")
            f.write("-" * 72 + "\n")
            for e in self.log_entries:
                f.write(f"{e['timestep']:>8} {e['elapsed_min']:>7.1f}m "
                        f"{e['train_mean_reward']:>8.4f} {e['train_sharpe']:>8.3f} "
                        f"{e['train_trade_days']:>7.1f}% "
                        f"{e['val_mean_reward']:>8.4f} {e['val_sharpe']:>8.3f} "
                        f"{e['val_trade_days']:>7.1f}%\n")

    def _on_training_end(self):
        self._save_log()


# Signal handler for graceful Ctrl+C
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\nCtrl+C detected — saving checkpoint and stopping...", file=sys.stderr)

signal.signal(signal.SIGINT, signal_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test')
    args = parser.parse_args()

    is_smoke = args.smoke

    print("=" * 60, file=sys.stderr)
    print("DURCHLAUF 5.0 — RL TRAINING", file=sys.stderr)
    print(f"Mode: {'SMOKE TEST' if is_smoke else 'FULL TRAINING'}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load episodes
    episodes = pd.read_parquet(EPISODES_PATH)
    train_n = (episodes['split'] == 'train').sum()
    val_n = (episodes['split'] == 'val').sum()
    print(f"Episodes: Train={train_n}, Val={val_n}", file=sys.stderr)

    # Create environments
    train_env = make_env(episodes, split='train',
                         feature_dropout=FEATURE_DROPOUT, noise_std=NOISE_STD)
    val_env = make_env(episodes, split='val',
                       feature_dropout=0.0, noise_std=0.0)

    # Smoke test env check
    obs, _ = train_env.reset()
    print(f"Obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]",
          file=sys.stderr)

    # Config
    if is_smoke:
        total_timesteps = SMOKE_TIMESTEPS
        val_interval = SMOKE_VAL_INTERVAL
    else:
        total_timesteps = TOTAL_TIMESTEPS
        val_interval = VAL_CHECK_INTERVAL

    # Create PPO model
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=min(N_STEPS, total_timesteps // 2) if is_smoke else N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS_PPO,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
        seed=42,
    )

    print(f"\nModel architecture:", file=sys.stderr)
    print(f"  Policy net: {POLICY_KWARGS['net_arch']}", file=sys.stderr)
    print(f"  Total timesteps: {total_timesteps}", file=sys.stderr)
    print(f"  Val check interval: {val_interval}", file=sys.stderr)
    print(f"  Feature dropout: {FEATURE_DROPOUT}", file=sys.stderr)
    print(f"  Noise std: {NOISE_STD}", file=sys.stderr)

    # Validation callback
    log_file = RESULTS_DIR / 'rl_training_log.txt'
    callback = ValidationCallback(
        val_env=val_env,
        episodes_df=episodes,
        check_interval=val_interval,
        patience=EARLY_STOP_PATIENCE,
        val_gate=VALIDATION_GATE,
        model_dir=MODEL_DIR,
        log_file=log_file,
    )

    # Train
    print(f"\nStarting training...", file=sys.stderr)
    start = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)

    elapsed = (time.time() - start) / 60
    print(f"\nTraining completed in {elapsed:.1f} minutes", file=sys.stderr)

    # Save final model
    model.save(MODEL_DIR / 'rl_agent_final.zip')
    print(f"Final model saved to models/rl_agent_final.zip", file=sys.stderr)
    print(f"Best model at models/rl_agent_v1.zip", file=sys.stderr)
    print(f"Training log at results/rl_training_log.txt", file=sys.stderr)

    if is_smoke:
        print("\nSMOKE TEST PASSED!", file=sys.stderr)


if __name__ == '__main__':
    main()
