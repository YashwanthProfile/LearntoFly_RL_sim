"""
Deterministic reconstruction of the force applied to the UAV (Roll body) from
the load-cell reading of the 3-DOF gimbal.

Physics used (Newton for the whole sprung system, everything above the load cell):

    sum_i m_i * (a_i - g)  =  F_applied + F_loadcell
    =>  F_applied = sum_i m_i * a_proper_i  -  F_loadcell

where a_proper_i = a_i - g is what MuJoCo stores in cacc (gravity is folded in),
and F_loadcell is the force the support exerts on the structure (MuJoCo force
sensor, expressed in the site frame).

Only ONE input function is needed: motor_command(t) -> 4 values (thrust [N] or RPM).
No noise, no turbulence, no joint locking.
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
USE_VIEWER = True              # False = headless, runs much faster
LOG_EVERY  = 1                 # log every n-th physics step

# Direction of a motor's force in its SITE frame.
# The XML actuators use gear="0 0 -1" and the original script used -z, so we keep -z.
# If your drone is mounted so that thrust should be +z, change it here (one place).
THRUST_DIR_LOCAL = np.array([0.0, 0.0, -1.0])

# Crazyflie 2.1 constants
KF          = 1.8e-8           # thrust = KF * omega^2   [N / (rad/s)^2]
DRONE_MASS  = 0.034
HOVER_N     = DRONE_MASS * 9.81 / 4.0
RPM_HOVER   = np.sqrt(HOVER_N / KF) * 60.0 / (2 * np.pi)

# ----------------------------------------------------------------------
# 1. THE ONE INPUT FUNCTION
#    Smooth, deterministic, different on every motor so that force AND
#    torque excite the gimbal in all directions.
# ----------------------------------------------------------------------
REL_AMP = np.array([0.40, 0.25, 0.50, 0.15])     # fraction of hover value
FREQ_HZ = np.array([0.50, 0.40, 0.60, 0.30])
PHASE   = np.array([0.00, 0.70, 1.30, 2.00])
TAU     = 2.0                                    # decay time constant [s]


def motor_command(t):
    """Per-motor command at time t: thrust [N] or RPM, depending on INPUT_TYPE."""
    base = HOVER_N if INPUT_TYPE == 'thrust' else RPM_HOVER
    env  = np.exp(-t / TAU)
    return np.clip(base * (1.0 + REL_AMP * np.sin(2*np.pi*FREQ_HZ*t + PHASE) * env),
                   0.0, None)


def command_to_thrust(cmd):
    if INPUT_TYPE == 'thrust':
        return np.asarray(cmd, dtype=float)
    omega = np.asarray(cmd) * 2*np.pi / 60.0
    return KF * omega**2

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


roll_id    = _id(mujoco.mjtObj.mjOBJ_BODY,   "Roll")
lc_site_id = _id(mujoco.mjtObj.mjOBJ_SITE,   "loadcell_site")
sensor_adr = model.sensor_adr[_id(mujoco.mjtObj.mjOBJ_SENSOR, "loadcell_force")]
motor_site = [_id(mujoco.mjtObj.mjOBJ_SITE, f"motor{i}") for i in range(1, 5)]

dt          = model.opt.timestep
body_mass   = model.body_mass.copy()
total_mass  = body_mass.sum()
print(f"Total mass = {total_mass:.5f} kg   (weight {total_mass*9.81:.4f} N)")

# ----------------------------------------------------------------------
# 3. APPLY MOTOR FORCES (exactly, at the motor sites)
# ----------------------------------------------------------------------
def apply_motor_forces(thrusts):
    """
    Put the four motor forces on the Roll body through xfrc_applied
    (force at the body COM + torque about the COM, both in the WORLD frame).
    Returns the net applied force in the world frame.
    Requires mj_kinematics() to have been called for the current state.
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
# 4. RECONSTRUCTION  (call after mj_forward, same state as the sensor)
# ----------------------------------------------------------------------
_acc = np.zeros(6)

def loadcell_world():
    """Load-cell force, rotated from the site frame into the world frame."""
    f_site = data.sensordata[sensor_adr:sensor_adr + 3]
    return data.site_xmat[lc_site_id].reshape(3, 3) @ f_site


def reconstruct_applied_force_world():
    """
    F_applied = sum_i m_i * a_proper(COM_i)  -  F_loadcell     [world frame]

    mj_objectAcceleration returns the acceleration of the body's own COM
    (rot; lin, world orientation), corrected for the rotating reference frame
    and including the -g offset.  Do NOT use data.cacc directly: it is
    referenced to the subtree COM of the whole tree, not to each body's COM.
    """
    P_dot = np.zeros(3)
    for b in range(1, model.nbody):
        if model.body_weldid[b] == 0:
            # Static body (welded to the world: LoadCell, Base).  MuJoCo does not
            # fill in its acceleration (mj_objectAcceleration returns 0), but the
            # load cell still carries its weight, i.e. proper acceleration = -g.
            a_proper = -model.opt.gravity
        else:
            mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                         b, _acc, 0)
            a_proper = _acc[3:6]
        P_dot += body_mass[b] * a_proper
    return P_dot - loadcell_world()

# ----------------------------------------------------------------------
# 5. SANITY CHECK: zero motor input -> reconstructed force must be exactly 0
# ----------------------------------------------------------------------
mujoco.mj_kinematics(model, data)
apply_motor_forces(np.zeros(4))
mujoco.mj_forward(model, data)
print(f"Rest check: load cell = {np.round(loadcell_world(), 5)} N, "
      f"reconstructed force = {np.round(reconstruct_applied_force_world(), 10)} N")

# ----------------------------------------------------------------------
# 6. SIMULATION
# ----------------------------------------------------------------------
log = []   # rows: t, cmd(4), thrust(4), LC(3), Fcmd_w(3), Frec_w(3), Fcmd_r(3), Frec_r(3)

def simulate(viewer=None):
    n_steps = int(round(DURATION / dt))
    wall0, sim0 = time.time(), data.time
    for k in range(n_steps):
        if viewer is not None and not viewer.is_running():
            break
        t = data.time - sim0

        # 1) refresh kinematics for the CURRENT state, 2) apply forces,
        # 3) forward dynamics -> sensor + accelerations for this exact state/force
        mujoco.mj_kinematics(model, data)
        cmd     = motor_command(t)
        thrusts = command_to_thrust(cmd)
        F_cmd_w = apply_motor_forces(thrusts)
        mujoco.mj_forward(model, data)

        LC       = loadcell_world()
        F_rec_w  = reconstruct_applied_force_world()
        R_roll   = data.xmat[roll_id].reshape(3, 3)
        F_cmd_r  = R_roll.T @ F_cmd_w
        F_rec_r  = R_roll.T @ F_rec_w

        if k % LOG_EVERY == 0:
            log.append(np.concatenate(([t], cmd, thrusts, LC,
                                       F_cmd_w, F_rec_w, F_cmd_r, F_rec_r)))

        mujoco.mj_step(model, data)      # same state & same applied force

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

hdr = (["Time"] + [f"cmd{i}" for i in range(1, 5)] + [f"Thrust{i}" for i in range(1, 5)]
       + ["LC_Fx", "LC_Fy", "LC_Fz"]
       + [f"Cmd_{a}_world" for a in "xyz"] + [f"Recon_{a}_world" for a in "xyz"]
       + [f"Cmd_{a}_roll" for a in "xyz"] + [f"Recon_{a}_roll" for a in "xyz"])
with open("loadcell_experiment.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(L)

err_w = np.linalg.norm(Fc_w - Fr_w, axis=1)
print(f"\nMax |F_cmd - F_recon| (world): {err_w.max():.3e} N   "
      f"(peak commanded force {np.linalg.norm(Fc_w, axis=1).max():.4f} N)")
print(f"Max |F_cmd - F_recon| (roll) : {np.linalg.norm(Fc_r - Fr_r, axis=1).max():.3e} N")

# ----------------------------------------------------------------------
# 7. PLOTS
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

# Fig 3 - validation, world frame (left) and Roll/UAV body frame (right)
fig3, axs = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
for j, n in enumerate("xyz"):
    for col, (C, R_, title) in enumerate([(Fc_w, Fr_w, "World frame"),
                                          (Fc_r, Fr_r, "Roll (UAV body) frame")]):
        ax = axs[j, col]
        ax.plot(t, C[:, j], 'b-',  lw=1.8, label="Commanded")
        ax.plot(t, R_[:, j], 'r--', lw=1.2, label="Reconstructed")
        ax.set_ylabel(f"$F_{n}$ (N)"); ax.grid(alpha=.4)
        if j == 0:
            ax.set_title(title); ax.legend(loc="upper right")
axs[-1, 0].set_xlabel("Time (s)"); axs[-1, 1].set_xlabel("Time (s)")
fig3.suptitle("Applied force: commanded vs reconstructed from load cell")
fig3.tight_layout(rect=[0, 0, 1, .97]); fig3.savefig("plot_3_validation.png", dpi=200)

# Fig 4 - error
fig4, ax = plt.subplots(figsize=(10, 3.5))
ax.semilogy(t, err_w + 1e-18)
ax.set_xlabel("Time (s)"); ax.set_ylabel("|error| (N)")
ax.set_title("Reconstruction error (should sit at numerical precision)")
ax.grid(alpha=.4, which="both"); fig4.tight_layout(); fig4.savefig("plot_4_error.png", dpi=200)

plt.show()
