"""
Deterministic reconstruction of the force applied to the UAV (Roll body) from
the load-cell reading of the 3-DOF gimbal.

    F_applied = sum_i m_i * a_proper_i  -  F_loadcell        (world frame)

Input: ONE function, motor_command(t) -> 4 values (thrust [N] or RPM).
The input is a static gravity-hold thrust plus a DECAYING sinusoid per motor,
the gimbal joints have angle limits + damping (see the XML), and the joint
rates are checked against MAX_RATE so the rig cannot tumble.
"""

import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------
MODEL_PATH = '3DOF_drone_gimbal.xml'
INPUT_TYPE = 'thrust'          # 'thrust' (N per motor)  or  'rpm' (RPM per motor)
DURATION   = 7.0               # s
USE_VIEWER = True              # False = headless, much faster (macOS viewer needs `mjpython`)
LOG_EVERY  = 1                 # log every n-th physics step
MAX_RATE   = 3.0               # rad/s - attitude-rate limit used for the warning/plot

# Direction of a motor's force in its SITE frame (+z = thrust up for an upright drone).
THRUST_DIR_LOCAL = np.array([0.0, 0.0, 1.0])

# Crazyflie 2.1 constants
KF = 1.8e-8                    # thrust = KF * omega^2

# ----------------------------------------------------------------------
# 1. THE ONE INPUT FUNCTION
#    thrust_i(t) = T_HOLD_i + A_i * exp(-t/TAU) * sin(2 pi f_i t + phi_i)
#    - T_HOLD balances the arm's own weight about the pitch/roll axes (computed
#      from the model below), so the arm does not just fall onto the stop.
#    - the sinusoid is small and DECAYS to zero, so rates/angles stay bounded.
# ----------------------------------------------------------------------
HOLD_AGAINST_GRAVITY = True
TAU     = 1.5                                     # decay time constant [s]
AMP_N   = np.array([0.020, 0.015, 0.025, 0.010])  # initial sine amplitude per motor [N]
FREQ_HZ = np.array([0.50, 0.40, 0.60, 0.30])
PHASE   = np.array([0.00, 0.70, 1.30, 2.00])
T_HOLD  = np.zeros(4)                             # filled in after the model is loaded


def thrust_to_rpm(T):
    return np.sqrt(np.asarray(T) / KF) * 60.0 / (2 * np.pi)


def motor_command(t):
    """Per-motor command at time t: thrust [N] or RPM, depending on INPUT_TYPE."""
    T = T_HOLD + AMP_N * np.exp(-t / TAU) * np.sin(2*np.pi*FREQ_HZ*t + PHASE)
    if INPUT_TYPE == 'thrust':
        return T
    return thrust_to_rpm(np.clip(T, 0.0, None))      # a rotor cannot reverse


def command_to_thrust(cmd):
    if INPUT_TYPE == 'thrust':
        return np.asarray(cmd, dtype=float)
    return KF * (np.asarray(cmd) * 2*np.pi / 60.0)**2

# ----------------------------------------------------------------------
# 2. LOAD MODEL
# ----------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)


def _id(obj, name):
    i = mujoco.mj_name2id(model, obj, name)
    if i == -1:
        raise RuntimeError(f"{name!r} not found in model")
    return i


roll_id    = _id(mujoco.mjtObj.mjOBJ_BODY, "Roll")
pitch_id   = _id(mujoco.mjtObj.mjOBJ_BODY, "Pitch")
lc_site_id = _id(mujoco.mjtObj.mjOBJ_SITE, "loadcell_site")
sensor_adr = model.sensor_adr[_id(mujoco.mjtObj.mjOBJ_SENSOR, "loadcell_force")]
motor_site = [_id(mujoco.mjtObj.mjOBJ_SITE, f"motor{i}") for i in range(1, 5)]

JOINTS   = ["yaw_joint", "pitch_joint", "roll_joint"]
jnt_ids  = [_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
qpos_adr = [model.jnt_qposadr[j] for j in jnt_ids]
dof_adr  = [model.jnt_dofadr[j] for j in jnt_ids]

dt         = model.opt.timestep
body_mass  = model.body_mass.copy()
print(f"Total mass = {body_mass.sum():.5f} kg   (weight {body_mass.sum()*9.81:.4f} N)")
for n, j in zip(JOINTS, jnt_ids):
    lo, hi = np.degrees(model.jnt_range[j])
    lim = f"[{lo:.1f}, {hi:.1f}] deg" if model.jnt_limited[j] else "free"
    print(f"  {n:12s} range {lim}   damping {model.dof_damping[model.jnt_dofadr[j]]:.4f}")

# ----------------------------------------------------------------------
# 3. STATIC GRAVITY-HOLD THRUST (balances the arm's weight about pitch & roll axes)
# ----------------------------------------------------------------------
def compute_hold_thrusts():
    mujoco.mj_kinematics(model, data)
    P = data.xanchor[_id(mujoco.mjtObj.mjOBJ_JOINT, "roll_joint")].copy()   # pivot
    g = model.opt.gravity

    def grav_torque(bodies):
        tau = np.zeros(3)
        for b in bodies:
            tau += np.cross(data.xipos[b] - P, body_mass[b] * g)
        return tau

    tau_roll  = grav_torque([roll_id])               # acts about the roll axis (x)
    tau_pitch = grav_torque([pitch_id, roll_id])     # acts about the pitch axis (y)

    A = np.zeros((2, 4))
    for i, sid in enumerate(motor_site):
        r = data.site_xpos[sid] - P
        d = data.site_xmat[sid].reshape(3, 3) @ THRUST_DIR_LOCAL
        c = np.cross(r, d)                           # torque per newton of thrust
        A[0, i], A[1, i] = c[0], c[1]
    b = np.array([-tau_roll[0], -tau_pitch[1]])
    T, *_ = np.linalg.lstsq(A, b, rcond=None)        # min-norm solution
    T = np.clip(T, 0.0, None)
    print(f"Gravity-hold thrust per motor [N]: {np.round(T, 4)}  "
          f"(total {T.sum():.4f} N)   residual torque [Nm]: {np.round(A @ T - b, 8)}")
    return T


if HOLD_AGAINST_GRAVITY:
    T_HOLD = compute_hold_thrusts()

# ----------------------------------------------------------------------
# 4. APPLY MOTOR FORCES (exactly, at the motor sites)
# ----------------------------------------------------------------------
def apply_motor_forces(thrusts):
    """
    Put the four motor forces on the Roll body through xfrc_applied
    (force at the body COM + torque about the COM, world frame).
    Returns the net applied force in the world frame.
    Needs mj_kinematics() for the current state.
    """
    data.xfrc_applied[:] = 0.0
    com   = data.xipos[roll_id]
    F_tot = np.zeros(3)
    M_tot = np.zeros(3)
    for sid, T in zip(motor_site, thrusts):
        R  = data.site_xmat[sid].reshape(3, 3)
        Fw = T * (R @ THRUST_DIR_LOCAL)
        F_tot += Fw
        M_tot += np.cross(data.site_xpos[sid] - com, Fw)
    data.xfrc_applied[roll_id, 0:3] = F_tot
    data.xfrc_applied[roll_id, 3:6] = M_tot
    return F_tot

# ----------------------------------------------------------------------
# 5. RECONSTRUCTION  (call after mj_forward, same state as the sensor)
# ----------------------------------------------------------------------
_acc = np.zeros(6)

def loadcell_world():
    f_site = data.sensordata[sensor_adr:sensor_adr + 3]
    return data.site_xmat[lc_site_id].reshape(3, 3) @ f_site


def reconstruct_applied_force_world():
    """F_applied = sum_i m_i * a_proper(COM_i) - F_loadcell   [world frame]"""
    P_dot = np.zeros(3)
    for b in range(1, model.nbody):
        if model.body_weldid[b] == 0:
            # static body (LoadCell, Base): MuJoCo reports no acceleration for it,
            # but the load cell carries its weight -> proper acceleration = -g
            a_proper = -model.opt.gravity
        else:
            mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                         b, _acc, 0)
            a_proper = _acc[3:6]
        P_dot += body_mass[b] * a_proper
    return P_dot - loadcell_world()

# ----------------------------------------------------------------------
# 6. SANITY CHECK: zero motor input -> reconstructed force must be exactly 0
# ----------------------------------------------------------------------
mujoco.mj_kinematics(model, data)
apply_motor_forces(np.zeros(4))
mujoco.mj_forward(model, data)
print(f"Zero-input check: reconstructed force = "
      f"{np.round(reconstruct_applied_force_world(), 10)} N")

# ----------------------------------------------------------------------
# 7. SIMULATION
# ----------------------------------------------------------------------
log = []

def simulate(viewer=None):
    n_steps = int(round(DURATION / dt))
    wall0, sim0 = time.time(), data.time
    for k in range(n_steps):
        if viewer is not None and not viewer.is_running():
            break
        t = data.time - sim0

        mujoco.mj_kinematics(model, data)                 # refresh for current state
        cmd     = motor_command(t)
        thrusts = command_to_thrust(cmd)
        F_cmd_w = apply_motor_forces(thrusts)
        mujoco.mj_forward(model, data)                    # sensor + accelerations

        LC      = loadcell_world()
        F_rec_w = reconstruct_applied_force_world()
        R_roll  = data.xmat[roll_id].reshape(3, 3)
        F_cmd_r = R_roll.T @ F_cmd_w
        F_rec_r = R_roll.T @ F_rec_w
        q  = data.qpos[qpos_adr]
        qd = data.qvel[dof_adr]

        if k % LOG_EVERY == 0:
            log.append(np.concatenate(([t], cmd, thrusts, LC,
                                       F_cmd_w, F_rec_w, F_cmd_r, F_rec_r, q, qd)))

        mujoco.mj_step(model, data)

        if viewer is not None and k % 50 == 0:
            viewer.sync()
            lag = (data.time - sim0) - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)


if USE_VIEWER:
    with mujoco.viewer.launch_passive(model, data) as v:
        simulate(v)
else:
    simulate()

L = np.array(log)
t = L[:, 0]
cmd, thr, LC = L[:, 1:5], L[:, 5:9], L[:, 9:12]
Fc_w, Fr_w   = L[:, 12:15], L[:, 15:18]
Fc_r, Fr_r   = L[:, 18:21], L[:, 21:24]
q, qd        = L[:, 24:27], L[:, 27:30]

hdr = (["Time"] + [f"cmd{i}" for i in range(1, 5)] + [f"Thrust{i}" for i in range(1, 5)]
       + ["LC_Fx", "LC_Fy", "LC_Fz"]
       + [f"Cmd_{a}_world" for a in "xyz"] + [f"Recon_{a}_world" for a in "xyz"]
       + [f"Cmd_{a}_roll" for a in "xyz"] + [f"Recon_{a}_roll" for a in "xyz"]
       + [f"q_{n}" for n in ("yaw", "pitch", "roll")]
       + [f"qd_{n}" for n in ("yaw", "pitch", "roll")])
with open("loadcell_experiment.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(L)

# ----------------------------------------------------------------------
# 8. SUMMARY
# ----------------------------------------------------------------------
err_w = np.linalg.norm(Fc_w - Fr_w, axis=1)
err_r = np.linalg.norm(Fc_r - Fr_r, axis=1)
print(f"\nMax |F_cmd - F_recon| world: {err_w.max():.3e} N   roll: {err_r.max():.3e} N   "
      f"(peak commanded force {np.linalg.norm(Fc_w, axis=1).max():.4f} N)")
for i, n in enumerate(JOINTS):
    j = jnt_ids[i]
    msg = (f"  {n:12s} angle [{np.degrees(q[:, i].min()):7.2f}, {np.degrees(q[:, i].max()):7.2f}] deg"
           f"   max rate {np.abs(qd[:, i]).max():6.3f} rad/s")
    if np.abs(qd[:, i]).max() > MAX_RATE:
        msg += "   <-- exceeds MAX_RATE"
    if model.jnt_limited[j]:
        lo, hi = model.jnt_range[j]
        if q[:, i].min() <= lo + np.radians(0.5) or q[:, i].max() >= hi - np.radians(0.5):
            msg += "   <-- touched the joint limit"
    print(msg)

# ----------------------------------------------------------------------
# 9. PLOTS
# ----------------------------------------------------------------------
# Fig 1 - inputs
fig1, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i in range(4):
    a.plot(t, cmd[:, i], label=f"Motor {i+1}")
    b.plot(t, thr[:, i], label=f"Motor {i+1}")
b.plot(t, thr.sum(axis=1), 'k--', label="Total")
a.set_ylabel("RPM" if INPUT_TYPE == 'rpm' else "Commanded thrust (N)")
b.set_ylabel("Thrust (N)"); b.set_xlabel("Time (s)")
a.legend(ncol=4); b.legend(ncol=5); a.grid(alpha=.4); b.grid(alpha=.4)
fig1.tight_layout(); fig1.savefig("plot_1_input.png", dpi=200)

# Fig 2 - load cell
fig2, ax = plt.subplots(figsize=(10, 4.5))
for j, n in enumerate("xyz"):
    ax.plot(t, LC[:, j], label=f"$F_{n}$")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Load-cell force (N)")
ax.set_title("Load cell (world frame)"); ax.legend(ncol=3); ax.grid(alpha=.4)
fig2.tight_layout(); fig2.savefig("plot_2_loadcell.png", dpi=200)

# Fig 3 - validation: world frame (left), Roll/UAV body frame (right).
# In the body frame the thrust is purely along z, so commanded Fx = Fy = 0:
# the body-frame x/y panels use the same y-scale as the force so the zero line is visible.
lim_r = 1.15 * np.abs(Fc_r).max()
fig3, axs = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
for j, n in enumerate("xyz"):
    for col, (C, R_, title) in enumerate([(Fc_w, Fr_w, "World frame"),
                                          (Fc_r, Fr_r, "Roll (UAV body) frame")]):
        ax = axs[j, col]
        ax.plot(t, C[:, j], 'b-',  lw=1.8, label="Commanded")
        ax.plot(t, R_[:, j], 'r--', lw=1.2, label="Reconstructed")
        ax.set_ylabel(f"$F_{n}$ (N)"); ax.grid(alpha=.4)
        if col == 1 and j < 2:
            ax.set_ylim(-lim_r, lim_r)
        ax.text(0.02, 0.06, f"max error {np.abs(C[:, j] - R_[:, j]).max():.1e} N",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
        if j == 0:
            ax.set_title(title); ax.legend(loc="upper right")
axs[-1, 0].set_xlabel("Time (s)"); axs[-1, 1].set_xlabel("Time (s)")
fig3.suptitle("Applied force: commanded vs reconstructed from load cell")
fig3.tight_layout(rect=[0, 0, 1, .97]); fig3.savefig("plot_3_validation.png", dpi=200)

# Fig 4 - error
fig4, ax = plt.subplots(figsize=(10, 3.5))
ax.semilogy(t, np.maximum(err_w, 1e-18), label="world")
ax.semilogy(t, np.maximum(err_r, 1e-18), label="roll", alpha=.7)
ax.set_xlabel("Time (s)"); ax.set_ylabel("|error| (N)"); ax.legend()
ax.set_title("Reconstruction error (should sit at numerical precision)")
ax.grid(alpha=.4, which="both"); fig4.tight_layout(); fig4.savefig("plot_4_error.png", dpi=200)

# Fig 5 - joint angles (with limits) and rates (with MAX_RATE)
fig5, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, n in enumerate(JOINTS):
    line, = a.plot(t, np.degrees(q[:, i]), label=n)
    j = jnt_ids[i]
    if model.jnt_limited[j]:
        for lim in np.degrees(model.jnt_range[j]):
            a.axhline(lim, color=line.get_color(), ls=":", lw=1)
    b.plot(t, qd[:, i], label=n)
b.axhline(MAX_RATE, color="k", ls="--", lw=1, label="MAX_RATE")
b.axhline(-MAX_RATE, color="k", ls="--", lw=1)
a.set_ylabel("Joint angle (deg)  [dotted = limits]"); b.set_ylabel("Joint rate (rad/s)")
b.set_xlabel("Time (s)"); a.legend(ncol=3); b.legend(ncol=4)
a.grid(alpha=.4); b.grid(alpha=.4)
fig5.tight_layout(); fig5.savefig("plot_5_joints.png", dpi=200)

plt.show()
