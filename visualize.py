"""
visualize.py — Load a saved PPO model and watch it run in the MuJoCo viewer.

Usage:
    python3 visualize.py                          # uses best model
    python3 visualize.py --model results/models/ppo_drone_final.zip
    python3 visualize.py --episodes 5
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from env import DroneEnv

# ------------------------------------------------------------------ #
#  Args                                                                #
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="results/models/best/best_model.zip",
    help="Path to the saved .zip model file",
)
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--seed",     type=int, default=0,  help="Random seed")
args = parser.parse_args()

model_path = Path(args.model)
if not model_path.exists():
    # Fall back to final model if best doesn't exist yet
    model_path = Path("results/models/ppo_drone_final.zip")
    if not model_path.exists():
        raise FileNotFoundError(
            "No saved model found. Run train.py first.\n"
            f"Looked for: {args.model} and results/models/ppo_drone_final.zip"
        )

print(f"Loading model: {model_path}")
model = PPO.load(str(model_path))

# ------------------------------------------------------------------ #
#  Separate env for rollout (we need direct access to mjData)         #
# ------------------------------------------------------------------ #
env = DroneEnv()

# Drone body ID — used to read 3D world position
_body_id = mujoco.mj_name2id(
    env.model, mujoco.mjtObj.mjOBJ_BODY, "Crazyflie_Combained-Body-v2"
)

# Temporary MjData used only to compute forward kinematics for the target
_tmp_data = mujoco.MjData(env.model)


def compute_target_pos(target_yaw: float, target_pitch: float) -> np.ndarray:
    """Return the 3-D world position of the drone body at the given joint angles."""
    mujoco.mj_resetDataKeyframe(env.model, _tmp_data, env.home_key_id)
    _tmp_data.qpos[env.yaw_qpos_adr]   = target_yaw
    _tmp_data.qpos[env.pitch_qpos_adr] = target_pitch
    mujoco.mj_forward(env.model, _tmp_data)
    return _tmp_data.xpos[_body_id].copy()

# ------------------------------------------------------------------ #
#  Viewer loop                                                         #
# ------------------------------------------------------------------ #
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.distance  = 1.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth   = 135

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        tgt_yaw   = info["target_yaw"]
        tgt_pitch = info["target_pitch"]
        target_pos = compute_target_pos(tgt_yaw, tgt_pitch)   # 3-D world pos

        print(f"\n── Episode {ep + 1}/{args.episodes} ──")
        print(f"   Start  : yaw={np.pi/2:.3f} rad  pitch=1.200 rad")
        print(f"   Target : yaw={tgt_yaw:.3f} rad  pitch={tgt_pitch:.3f} rad")
        print(f"   Target world pos: {target_pos}")

        total_reward = 0.0
        step = 0
        done = False

        while not done and viewer.is_running():
            step_start = time.time()

            # Policy inference (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # Sync viewer
            viewer.sync()

            # Draw blue sphere at target position every frame
            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                size  = np.array([0.02, 0.02, 0.02]),
                pos   = target_pos,
                mat   = np.eye(3).flatten(),
                rgba  = np.array([0.0, 0.4, 1.0, 0.85], dtype=np.float32),
            )
            viewer.user_scn.ngeom = 1

            # Print state at ~5 Hz
            if step % 2 == 0:
                yaw   = info["yaw"]
                pitch = info["pitch"]
                err_y = DroneEnv._angle_wrap(yaw - tgt_yaw)
                err_p = pitch - tgt_pitch
                print(
                    f"   t={step * env.control_dt:5.1f}s | "
                    f"yaw={yaw:.3f} ({np.rad2deg(yaw):6.1f}°)  "
                    f"pitch={pitch:.3f} ({np.rad2deg(pitch):5.1f}°)  "
                    f"err=[{np.rad2deg(err_y):+.1f}°, {np.rad2deg(err_p):+.1f}°]  "
                    f"r={reward:.3f}",
                    end="\r",
                )

            # Pace to real-time (0.1 s per control step)
            elapsed = time.time() - step_start
            sleep_t = env.control_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        status = "CRASHED" if terminated else "TIME LIMIT"
        print(f"\n   [{status}] steps={step}  total_reward={total_reward:.1f}")

    print("\nVisualization complete.")
