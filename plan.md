# `env.py` — 5DOF Drone RL Environment Plan

## System Summary (from XML)

| Element | Detail |
|---|---|
| **Joints** | `BaseGrounded-v2_Yaw` (free rotation), `3-DOF-Assm_3-DOF-mount-v1_Pitch` (range -0.2 → 1.2 rad), `Bearing_6x13x5-Rear-v1_Revolute-9` (gimbal arm roll/twist) |
| **Actuators** | `motor1–4`, `ctrlrange [0, 1]`, gear Z = -0.27 (thrust), gear 6 = ±0.002 (yaw torque) |
| **Home keyframe** | `qpos = [1.5707, 1.2, 4.903]` → [yaw=π/2 (90°), pitch=1.2 rad, revolute-9=4.903 rad] |
| **Timestep** | 0.0002s MuJoCo step → 500 sim steps per 10Hz control step |

---

## What is `Revolute-9`?

It is the **third passive DOF** — the roll/twist of the gimbal arm itself (`Gimbal-Arm-v1-v1` body). The drone sits at the end of this arm, so if this joint spins wildly it means the arm is twisting uncontrollably. The policy does **not** control it directly, but it appears in `qpos[2]`. We will **observe it** and use it as a crash indicator if it deviatevs too far from its home value (`4.903 rad`).

---

## Confirmed Design Decisions

| Decision | Choice |
|---|---|
| **Target range** | Yaw: full 0 → 2π, Pitch: full -0.2 → 1.2 rad |
| **Drone start position** | Yaw = π/2 (90°) via keyframe — **no override needed** |
| **Observation** | 8D — see Component 3 |
| **Revolute-9** | Observed + crash detection only |
| **Episode length** | 20s, 0.1s steps → **200 steps/episode** |
| **Sim steps per control** | 0.1 / 0.0002 = **500 MuJoCo steps** |
| **Control rate** | 10 Hz |
| **Reward** | Simple tracking + energy penalty |
| **Termination** | Joint limit hit OR velocity exceeded OR non-finite |

---

## Components to Build (in order)

### Component 1 — `__init__`: MuJoCo Setup
Load the model, resolve IDs for joints/actuators, set timing.

**What to define:**
- Load `5DOF.xml` → `self.model`, `self.data`
- Resolve home keyframe ID
- Resolve actuator IDs: `motor1–4`
- Resolve joint `qpos` + `qvel` addresses: yaw, pitch, revolute-9
- `control_dt = 0.1`, `sim_steps_per_control = 500`
- `max_episode_steps = 200`
- Define `action_space` (4D, [0,1]) and `observation_space` (8D)

---

### Component 2 — `reset()`: Episode Initialization
Reset the drone to home position and generate a new random target.

**What to define:**
- Call `mj_resetDataKeyframe` → home keyframe (sets yaw=π/2, pitch=1.2, rev9=4.903 automatically)
- Call `mj_forward` to sync physics state
- Sample random target: yaw ∈ [0, 2π], pitch ∈ [-0.2, 1.2]
- Return `_get_obs(), {}`

---

### Component 3 — `_get_obs()`: Observation Vector
What the policy sees every step.

**Observation (8D):**
```
[yaw, pitch,               # current joint positions (2)
 yaw_vel, pitch_vel,       # joint velocities (2)
 target_yaw, target_pitch, # goal (2)
 yaw_error, pitch_error]   # error (2)
```
→ **8D observation** (`observation_space = Box(-inf, inf, shape=(8,))`)

> [!NOTE]
> revolute-9 is NOT in the observation — it is only used internally for crash detection.

---

### Component 4 — `step()`: Apply Action + Advance Sim
The core loop.

**What to define:**
- Clip action to `[0, 1]` (ctrlrange)
- Write `data.ctrl[motor_id] = action[i]` for all 4 motors
- Loop 500 times → `mj_step`
- Read new joint positions + velocities
- Call `_compute_reward()`
- Check termination conditions
- Increment step counter
- Return `(obs, reward, terminated, truncated, info)`

---

### Component 5 — `_compute_reward()` + Termination

**Reward:**
```python
yaw_error   = angle_wrap(current_yaw - target_yaw)   # shortest path, handles 0/2π wrap
pitch_error = current_pitch - target_pitch

reward = - 1.0 * yaw_error²
         - 1.0 * pitch_error²
         - 0.01 * sum(action²)    # small energy penalty
```

**Termination conditions (all → `terminated=True`):**

| Condition | Reason | Value |
|---|---|---|
| `pitch <= -0.2 or pitch >= 1.2` | Hit joint limit | Boundary crash |
| `abs(yaw_vel) > 20 rad/s` | Spinning out of control | Velocity limit |
| `abs(pitch_vel) > 20 rad/s` | Pitching out of control | Velocity limit |
| `abs(rev9 - 4.903) > π/2` | Gimbal arm twisting wildly | Crash |
| State is non-finite | NaN/Inf | Physics diverged |

**Truncation:**
| Condition | Reason |
|---|---|
| `steps >= 200` | 20s time limit reached |

---

## Build Order

```
Step 1 → __init__  (MuJoCo load + ID resolution + spaces)
Step 2 → reset()   (keyframe → yaw=π/2 auto-set + random target)
Step 3 → _get_obs() (8D vector)
Step 4 → step()    (ctrl → 500x mj_step → obs/reward/done)
Step 5 → _compute_reward() + termination logic
Step 6 → check_env() sanity check
```

> [!TIP]
> We'll build and test each component one at a time. Start with Step 1.
