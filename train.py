"""
train.py — SB3 PPO training for the 5DOF drone gimbal environment.

Usage:
    python3 train.py
    python3 train.py --timesteps 200000      # short smoke test
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
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
N_ENVS           = 8
TOTAL_TIMESTEPS  = 3_000_000
CHECKPOINT_EVERY = 100_000          # total env steps (divided by N_ENVS below)
EVAL_EVERY       = 50_000           # total env steps (divided by N_ENVS below)

PPO_KWARGS = dict(
    learning_rate  = 3e-4,
    n_steps        = 1024,          # per env → 8192 samples per update
    batch_size     = 512,
    gamma          = 0.995,         # 25 Hz control → ~8 s horizon
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.005,
    verbose        = 1,
    tensorboard_log= str(LOG_DIR),
)

# ------------------------------------------------------------------ #
#  Build envs                                                          #
# ------------------------------------------------------------------ #
def make_env(seed_offset=0):
    def _factory():
        env = Monitor(DroneEnv())
        env.reset(seed=SEED + seed_offset)
        return env
    return _factory


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats next to best_model.zip whenever a new best is found."""
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    def _on_step(self):
        self.model.get_vec_normalize_env().save(self.save_path)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    train_env = SubprocVecEnv([make_env(seed_offset=i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, gamma=PPO_KWARGS["gamma"])

    # Eval env shares obs stats with train env (EvalCallback syncs them); raw rewards
    eval_env = DummyVecEnv([make_env(seed_offset=1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ------------------------------------------------------------------ #
    #  Build model                                                         #
    # ------------------------------------------------------------------ #
    model = PPO("MlpPolicy", train_env, seed=SEED, **PPO_KWARGS)

    print(f"\nPolicy network:\n{model.policy}\n")

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #
    best_dir = MODEL_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_freq         = max(CHECKPOINT_EVERY // N_ENVS, 1),
            save_path         = str(CHECKPOINT_DIR),
            name_prefix       = "ppo_drone",
            save_vecnormalize = True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path = str(best_dir),
            log_path             = str(LOG_DIR),
            eval_freq            = max(EVAL_EVERY // N_ENVS, 1),
            n_eval_episodes      = 10,
            deterministic        = True,
            render               = False,
            callback_on_new_best = SaveVecNormalizeCallback(str(best_dir / "vecnormalize.pkl")),
        ),
    ]

    # ------------------------------------------------------------------ #
    #  Train                                                               #
    # ------------------------------------------------------------------ #
    print(f"Training PPO for {args.timesteps:,} timesteps on {N_ENVS} envs...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    # ------------------------------------------------------------------ #
    #  Save final model                                                    #
    # ------------------------------------------------------------------ #
    final_path = MODEL_DIR / "ppo_drone_final"
    model.save(str(final_path))
    train_env.save(str(MODEL_DIR / "ppo_drone_final_vecnormalize.pkl"))
    print(f"\nSaved final model to: {final_path}.zip")

    # ------------------------------------------------------------------ #
    #  Quick deterministic rollout to sanity-check                        #
    # ------------------------------------------------------------------ #
    print("\nRunning 1 deterministic evaluation episode...")
    train_env.training = False
    test_env  = DroneEnv()
    obs, info = test_env.reset(seed=999)
    total_reward = 0.0
    steps = 0
    done  = False

    while not done:
        action, _ = model.predict(train_env.normalize_obs(obs), deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    err_yaw = DroneEnv._angle_wrap(info["yaw"] - info["target_yaw"])
    err_pitch = info["pitch"] - info["target_pitch"]
    print(f"  Steps        : {steps}")
    print(f"  Total reward : {total_reward:.2f}")
    print(f"  Final error  : yaw {np.rad2deg(err_yaw):+.1f}°  pitch {np.rad2deg(err_pitch):+.1f}°")
    print(f"  Terminated   : {terminated}  |  Truncated: {truncated}")
    print("\nDone.")


if __name__ == "__main__":   # required by SubprocVecEnv
    main()
