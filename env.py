import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class DroneEnv(gym.Env):
    """
    5DOF Drone-on-Gimbal RL Environment.
    Observation  : 8D  [yaw, pitch, yaw_vel, pitch_vel, tgt_yaw, tgt_pitch, err_yaw, err_pitch]
    Action       : 4D  [motor1, motor2, motor3, motor4]  each in [0, 1]
    Control rate : 10 Hz  (0.1 s per step)
    Episode      : 200 steps = 20 s
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------ #
    #  Component 1 — __init__                                             #
    # ------------------------------------------------------------------ #
    def __init__(self, xml_path=None, render_mode=None):
        super().__init__()

        # --- Load MuJoCo model ---
        if xml_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(current_dir, "5DOF.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # --- Home keyframe (qpos = [π/2, 1.2, 4.903]) ---
        self.home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if self.home_key_id < 0:
            raise RuntimeError("'home' keyframe not found in XML.")

        # --- Actuator IDs: motor1–4 ---
        motor_names = ["motor1", "motor2", "motor3", "motor4"]
        self.motor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in motor_names
        ]
        if any(mid < 0 for mid in self.motor_ids):
            raise RuntimeError("One or more motor actuators not found in XML.")

        # --- Joint qpos / qvel addresses ---
        yaw_jnt   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "BaseGrounded-v2_Yaw")
        pitch_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "3-DOF-Assm_3-DOF-mount-v1_Pitch")
        rev9_jnt  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Bearing_6x13x5-Rear-v1_Revolute-9")

        self.yaw_qpos_adr   = self.model.jnt_qposadr[yaw_jnt]
        self.pitch_qpos_adr = self.model.jnt_qposadr[pitch_jnt]
        self.rev9_qpos_adr  = self.model.jnt_qposadr[rev9_jnt]

        self.yaw_qvel_adr   = self.model.jnt_dofadr[yaw_jnt]
        self.pitch_qvel_adr = self.model.jnt_dofadr[pitch_jnt]

        # rev9 qvel not in observation, but needed for crash check
        self.rev9_qvel_adr  = self.model.jnt_dofadr[rev9_jnt]

        # --- Timing ---
        self.control_dt           = 0.1                    # 10 Hz
        self.sim_steps_per_control = int(
            round(self.control_dt / self.model.opt.timestep)
        )                                                  # 0.1 / 0.0002 = 500
        self.max_episode_steps    = 200                    # 20 s

        # --- Spaces ---
        # Action: 4 motors, each ctrl ∈ [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation: 8D, unbounded
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # --- Episode state (initialised in reset) ---
        self.target_yaw   = 0.0
        self.target_pitch = 0.0
        self.steps        = 0

        # --- Thrust ctrl range (from apply_forces.py) ---
        # min_thrust = 0.0500 N, max_thrust = 0.0545 N, gear_z = 0.27
        self.min_ctrl = 0.0500 / 0.27   # ≈ 0.1852
        self.max_ctrl = 0.0545 / 0.27   # ≈ 0.2019

        print(f"[DroneEnv] Loaded: {xml_path}")
        print(f"[DroneEnv] sim_steps_per_control = {self.sim_steps_per_control}")
        print(f"[DroneEnv] action_space  = {self.action_space}")
        print(f"[DroneEnv] obs_space     = {self.observation_space}")

    # ------------------------------------------------------------------ #
    #  Component 2 — reset()                                              #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset to home keyframe (qpos = [π/2, 1.2, 4.903], qvel = 0)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)   # sync derived quantities

        # Sample a random target
        self.target_yaw   = self.np_random.uniform(0.0, 2 * np.pi)
        self.target_pitch = self.np_random.uniform(-0.2, 1.2)

        self.steps = 0

        obs = self._get_obs()
        info = {
            "target_yaw":   self.target_yaw,
            "target_pitch": self.target_pitch,
        }
        return obs, info

    # ------------------------------------------------------------------ #
    #  Component 3 (stub) — _get_obs()                                   #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        yaw   = self.data.qpos[self.yaw_qpos_adr]
        pitch = self.data.qpos[self.pitch_qpos_adr]
        yaw_vel   = self.data.qvel[self.yaw_qvel_adr]
        pitch_vel = self.data.qvel[self.pitch_qvel_adr]

        err_yaw   = self._angle_wrap(yaw - self.target_yaw)
        err_pitch = pitch - self.target_pitch

        return np.array(
            [yaw, pitch, yaw_vel, pitch_vel,
             self.target_yaw, self.target_pitch,
             err_yaw, err_pitch],
            dtype=np.float32
        )

    @staticmethod
    def _angle_wrap(angle):
        """Wrap angle to [-π, π] (shortest angular distance)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------ #
    #  Component 4 — step()                                               #
    # ------------------------------------------------------------------ #
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply control to all 4 motors
        # Remap action [0,1] → [min_ctrl, max_ctrl]
        ctrl_values = self.min_ctrl + action * (self.max_ctrl - self.min_ctrl)
        for i, mid in enumerate(self.motor_ids):
            self.data.ctrl[mid] = ctrl_values[i]

        # Advance MuJoCo simulation (500 steps = 0.1 s of physics)
        for _ in range(self.sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated  = self.steps >= self.max_episode_steps

        info = {
            "yaw":          float(self.data.qpos[self.yaw_qpos_adr]),
            "pitch":        float(self.data.qpos[self.pitch_qpos_adr]),
            "yaw_vel":      float(self.data.qvel[self.yaw_qvel_adr]),
            "pitch_vel":    float(self.data.qvel[self.pitch_qvel_adr]),
            "target_yaw":   self.target_yaw,
            "target_pitch": self.target_pitch,
            "step":         self.steps,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Component 5 — reward + termination                                 #
    # ------------------------------------------------------------------ #
    def _compute_reward(self, action):
        yaw   = self.data.qpos[self.yaw_qpos_adr]
        pitch = self.data.qpos[self.pitch_qpos_adr]

        err_yaw   = self._angle_wrap(yaw - self.target_yaw)
        err_pitch = pitch - self.target_pitch

        reward = (
            - 1.0  * err_yaw   ** 2
            - 1.0  * err_pitch ** 2
            - 0.01 * float(np.sum(action ** 2))   # energy penalty
        )
        return float(reward)

    def _is_terminated(self):
        pitch   = float(self.data.qpos[self.pitch_qpos_adr])
        yaw_vel = float(self.data.qvel[self.yaw_qvel_adr])
        pit_vel = float(self.data.qvel[self.pitch_qvel_adr])
        rev9    = float(self.data.qpos[self.rev9_qpos_adr])

        # Joint boundary crash
        if pitch <= -0.2 or pitch >= 1.2:
            return True
        # Velocity runaway
        if abs(yaw_vel) > 20.0 or abs(pit_vel) > 20.0:
            return True
        # Gimbal arm twist (rev9 deviates > π/2 from home = 4.903)
        if abs(rev9 - 4.903) > np.pi / 2:
            return True
        # Physics diverged
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return True
        return False
