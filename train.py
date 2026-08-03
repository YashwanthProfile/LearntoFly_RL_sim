"""
train.py — SB3 PPO training for the 5DOF drone gimbal environment.

Usage:
    python3 train.py
"""

import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from env import DroneEnv

# ------------------------------------------------------------------ #
#  Directories                                                         #
# ------------------------------------------------------------------ #
BASE_DIR      = Path(__file__).parent
RESULTS_DIR   = BASE_DIR / "results"
MODEL_DIR     = RESULTS_DIR / "models"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR       = RESULTS_DIR / "logs"

for d in [MODEL_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  Hyperparameters                                                     #
# ------------------------------------------------------------------ #
SEED             = 42
TOTAL_TIMESTEPS  = 500_000
CHECKPOINT_EVERY = 10_000
EVAL_EVERY       = 10_000

PPO_KWARGS = dict(
    learning_rate  = 3e-4,
    n_steps        = 1024,
    batch_size     = 256,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.0,
    verbose        = 1,
    tensorboard_log= str(LOG_DIR),
)

# ------------------------------------------------------------------ #
#  Build envs                                                          #
# ------------------------------------------------------------------ #
def make_env(seed_offset=0):
    def _factory():
        return Monitor(DroneEnv())
    return _factory

train_env = DummyVecEnv([make_env(seed_offset=0)])
eval_env  = DummyVecEnv([make_env(seed_offset=1000)])

# ------------------------------------------------------------------ #
#  Build model                                                         #
# ------------------------------------------------------------------ #
model = PPO("MlpPolicy", train_env, seed=SEED, **PPO_KWARGS)

print(f"\nPolicy network:\n{model.policy}\n")

# ------------------------------------------------------------------ #
#  Callbacks                                                           #
# ------------------------------------------------------------------ #
callbacks = [
    CheckpointCallback(
        save_freq   = CHECKPOINT_EVERY,
        save_path   = str(CHECKPOINT_DIR),
        name_prefix = "ppo_drone",
    ),
    EvalCallback(
        eval_env,
        best_model_save_path = str(MODEL_DIR / "best"),
        log_path             = str(LOG_DIR),
        eval_freq            = EVAL_EVERY,
        deterministic        = True,
        render               = False,
    ),
]

# ------------------------------------------------------------------ #
#  Train                                                               #
# ------------------------------------------------------------------ #
print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

# ------------------------------------------------------------------ #
#  Save final model                                                    #
# ------------------------------------------------------------------ #
final_path = MODEL_DIR / "ppo_drone_final"
model.save(str(final_path))
print(f"\nSaved final model to: {final_path}.zip")

# ------------------------------------------------------------------ #
#  Quick deterministic rollout to sanity-check                        #
# ------------------------------------------------------------------ #
print("\nRunning 1 deterministic evaluation episode...")
test_env  = DroneEnv()
obs, info = test_env.reset(seed=999)
total_reward = 0.0
steps = 0
done  = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    steps += 1
    done = terminated or truncated

print(f"  Steps        : {steps}")
print(f"  Total reward : {total_reward:.2f}")
print(f"  Terminated   : {terminated}  |  Truncated: {truncated}")
print("\nDone.")
