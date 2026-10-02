"""
Apply a user‑selectable time‑varying FORCE and TORQUE at the drone mount point.
All inputs (Fx, Fy, Fz, Tx, Ty, Tz) are defined in the ARM (local) frame.
The code rotates them to the world frame before applying them via mj_applyFT.
Logs and plots: applied force/torque, load‑cell reaction, joint angles, and
a cross‑validation plot comparing the commanded applied force to a force
reconstructed from the load‑cell sensor (to verify consistency).

The reconstruction uses the ANALYTIC center‑of‑mass acceleration obtained
from MuJoCo's per‑body linear COM accelerations (data.cacc), rather than a
finite difference of the COM velocity. This avoids step‑lag and numerical
noise.
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

# ----------------------------------------------------------------------
# 0. USER SELECTION: FORCE PROFILE AND PARAMETERS
# ----------------------------------------------------------------------
FORCE_PROFILE = 'realistic'
#   'constant' | 'ramp' | 'step' | 'sine_decay' | 'chirp'
#   | 'impulse' | 'drone_mimic' | 'realistic'

# ----------------------------------------------------------------------
# Profile parameters
# ----------------------------------------------------------------------
Fx_const = 0.0
Fy_const = 0.2
Fz_const = 0.0
Tx_const, Ty_const, Tz_const = 0.0, 0.0, 0.0

amp_x   = 0.1
amp_y   = 0.4
amp_z   = 0.5
freq    = 1.0
tau     = 0.001

tor_amp_x = 0.0
tor_amp_y = 0.0
tor_amp_z = 0.0

ramp_duration = 2.0
ramp_amp_x = 0.0
ramp_amp_y = 0.0
ramp_amp_z = 0.5
ramp_tor_x = 0.0
ramp_tor_y = 0.0
ramp_tor_z = 0.0

step_time = 1.0
step_amp_x = 0.0
step_amp_y = 0.0
step_amp_z = 0.5
step_tor_x = 0.0
step_tor_y = 0.0
step_tor_z = 0.0

impulse_times = [1.0, 5.0]
impulse_width = 0.1
impulse_peak_x = 0.0
impulse_peak_y = 1.0
impulse_peak_z = 0.0
impulse_tor_x = 0.0
impulse_tor_y = 0.0
impulse_tor_z = 0.0

freq_start = 0.2
freq_end   = 1.0
chirp_duration = 5.0

thrust_hover = 0.5
thrust_noise = 0.05
noise_freq   = 0.3

# ----------------------------------------------------------------------
# REALISTIC DRONE FORCING  (applied in ARM frame)
# ----------------------------------------------------------------------
# Drone physical parameters
rnd_drone_mass      = 0.034                       # kg (Crazyflie-class)
rnd_hover_thrust_N  = rnd_drone_mass * 9.81       # N — supports drone weight (~0.334 N)

# Thrust-axis turbulence (fraction of hover).  Broadband gust is built from
# a weighted sum of incommensurate sinusoids, giving a quasi-periodic
# signal with energy concentrated at low frequency (vK-like spectrum).
rnd_thrust_turb_amp   = 0.20                      # ±20 % of hover at 1σ
rnd_thrust_turb_spec  = [                         # (freq Hz, weight)
    (0.11, 1.00),
    (0.27, 0.75),
    (0.58, 0.55),
    (1.21, 0.35),
    (2.47, 0.20),
    (5.03, 0.10),
]
rnd_thrust_turb_phase = [0.0, 1.7, 3.1, 4.6, 0.9, 2.3]

# Lateral forces — from wind gusts and small attitude deviations.  Much
# smaller than thrust.
rnd_lat_amp           = 0.02                      # ±20 mN on each axis
rnd_lat_freq_x        = 0.53                      # Hz
rnd_lat_freq_y        = 0.31

# Slow pilot-like maneuvers — attitude offsets modulate lateral force and
# tilt the thrust vector slightly.
rnd_pilot_pitch_amp   = 0.06                      # rad, ±3.4°
rnd_pilot_roll_amp    = 0.06
rnd_pilot_pitch_freq  = 0.22
rnd_pilot_roll_freq   = 0.18

# Residual attitude torques — from differential thrust between motors,
# CG misalignment, prop imbalance.
rnd_tor_x_amp         = 0.003                     # Nm
rnd_tor_y_amp         = 0.003
rnd_tor_z_amp         = 0.0015
rnd_tor_x_freq        = 0.41
rnd_tor_y_freq        = 0.29
rnd_tor_z_freq        = 0.15

# Prop-wash / motor ripple — small high-frequency noise on all channels
rnd_jitter_amp_F      = 0.005                     # N
rnd_jitter_amp_T      = 0.0003                    # Nm
rnd_jitter_freq_F     = 43.0                      # Hz
rnd_jitter_freq_T     = 61.0

# ----------------------------------------------------------------------
# 1. LOAD MODEL AND CHECK KEYFRAME / SENSOR
# ----------------------------------------------------------------------
model_path = '3DOF_gimbal.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print("Reset to 'home' keyframe.")
else:
    print("WARNING: 'home' keyframe not found. Using default initial state.")

sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "loadcell_force")
if sensor_id == -1:
    print("ERROR: Sensor 'loadcell_force' not found in the model!")
    sys.exit(1)
else:
    sensor_adr = model.sensor_adr[sensor_id]
    print(f"Load‑cell sensor found: ID={sensor_id}, address={sensor_adr}")

app_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "force_app_point")
if app_site_id == -1:
    print("ERROR: Site 'force_app_point' not found in the model.")
    sys.exit(1)
else:
    force_local_pos = model.site_pos[app_site_id].copy()
    print(f"Force application point (local to Gimbal-Arm): {force_local_pos}")

arm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Gimbal-Arm-v1-v1")
if arm_body_id == -1:
    print("ERROR: Body 'Gimbal-Arm-v1-v1' not found.")
    sys.exit(1)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BaseGrounded-v2")
if base_id == -1:
    print("ERROR: Body 'BaseGrounded-v2' not found.")
    sys.exit(1)

g = 9.81
total_mass = np.sum(model.body_mass)
body_masses = model.body_mass.copy()   # used by the analytic COM acceleration
print(f"Total mass: {total_mass:.6f} kg")

# ----------------------------------------------------------------------
# 2. WRENCH GENERATOR (Force + Torque in ARM frame)
# ----------------------------------------------------------------------
def _gust(t, spec, phases):
    """Broadband gust: weighted sum of incommensurate sines, normalised to ≈[-1, 1]."""
    s = 0.0
    w = 0.0
    for (f, a), ph in zip(spec, phases):
        s += a * np.sin(2*np.pi*f*t + ph)
        w += a
    return s / w


def get_applied_wrench(t):
    if FORCE_PROFILE == 'constant':
        if t < 3.0:
            F = 0*np.array([Fx_const, Fy_const, Fz_const])
            T = 0*np.array([Tx_const, Ty_const, Tz_const])
        else:
            F = np.array([Fx_const, Fy_const, Fz_const])
            T = np.array([Tx_const, Ty_const, Tz_const])

    elif FORCE_PROFILE == 'sine_decay':
        if t > 3.0:
            envelope = np.exp(-t * tau)
            Fx = 0*amp_x * np.sin(2 * np.pi * freq * t) * envelope
            Fy = amp_y * np.cos(2 * np.pi * freq * t) * envelope
            Fz = 0*Fz_const + 0*amp_z * np.sin(2 * np.pi * freq * t * 1.2) * envelope
            F = np.array([Fx, Fy, Fz])

            Tx = 0*tor_amp_x * np.sin(2 * np.pi * freq * t * 0.8) * envelope
            Ty = 0*tor_amp_y * np.cos(2 * np.pi * freq * t * 0.9) * envelope
            Tz = 0*tor_amp_z * np.sin(2 * np.pi * freq * t * 1.1) * envelope
            T = np.array([Tx, Ty, Tz])
        else:
            F = 0*np.array([Fx_const, Fy_const, Fz_const])
            T = 0*np.array([Tx_const, Ty_const, Tz_const])

    elif FORCE_PROFILE == 'chirp':
        if t < chirp_duration:
            f = freq_start + (freq_end - freq_start) * (t / chirp_duration)
        else:
            f = freq_end
        if t < chirp_duration:
            phase = 2*np.pi*(freq_start*t + 0.5*(freq_end - freq_start)*t**2/chirp_duration)
        else:
            phase = 2*np.pi*(freq_start*chirp_duration + 0.5*(freq_end - freq_start)*chirp_duration) \
                    + 2*np.pi*freq_end*(t - chirp_duration)

        Fx = amp_x * np.sin(phase) * 0.5
        Fy = amp_y * np.cos(phase * 0.8) * 0.5
        Fz = Fz_const + amp_z * np.sin(phase * 1.2) * 0.5
        F = np.array([Fx, Fy, Fz])

        Tx = tor_amp_x * np.sin(phase * 0.7) * 0.5
        Ty = tor_amp_y * np.cos(phase * 0.6) * 0.5
        Tz = tor_amp_z * np.sin(phase * 0.9) * 0.5
        T = np.array([Tx, Ty, Tz])

    elif FORCE_PROFILE == 'ramp':
        ramp_factor = t / ramp_duration if t < ramp_duration else 1.0
        F = np.array([ramp_amp_x, ramp_amp_y, Fz_const + ramp_amp_z]) * ramp_factor
        T = np.array([ramp_tor_x, ramp_tor_y, ramp_tor_z]) * ramp_factor

    elif FORCE_PROFILE == 'step':
        step_val = 1.0 if t >= step_time else 0.0
        F = np.array([step_amp_x, step_amp_y, Fz_const + step_amp_z]) * step_val
        T = np.array([step_tor_x, step_tor_y, step_tor_z]) * step_val

    elif FORCE_PROFILE == 'impulse':
        widths = [impulse_width]*len(impulse_times) if not isinstance(impulse_width, (list, tuple)) else impulse_width
        Fx, Fy, Fz = 0.0, 0.0, 0.0
        Tx, Ty, Tz = 0.0, 0.0, 0.0
        for t0, w in zip(impulse_times, widths):
            if t0 <= t <= t0 + w:
                s = np.sin(np.pi * (t - t0) / w)
                Fx += impulse_peak_x * s
                Fy += impulse_peak_y * s
                Fz += impulse_peak_z * s
                Tx += impulse_tor_x * s
                Ty += impulse_tor_y * s
                Tz += impulse_tor_z * s
        F = np.array([Fx, Fy, Fz_const + Fz])
        T = np.array([Tx, Ty, Tz])

    elif FORCE_PROFILE == 'drone_mimic':
        Fx = 0.02 * np.sin(2*np.pi*0.2*t)
        Fy = 0.02 * np.cos(2*np.pi*0.15*t)
        Fz = thrust_hover + thrust_noise * np.sin(2*np.pi*noise_freq*t)
        F = np.array([Fx, Fy, Fz])
        Tx = 0.001 * np.sin(2*np.pi*0.1*t)
        Ty = 0.001 * np.cos(2*np.pi*0.12*t)
        Tz = 0.002 * np.sin(2*np.pi*0.2*t)
        T = np.array([Tx, Ty, Tz])

    elif FORCE_PROFILE == 'realistic':
        # ---- 1) Thrust with broadband turbulence ----
        turb = _gust(t, rnd_thrust_turb_spec, rnd_thrust_turb_phase)
        Fz = rnd_hover_thrust_N * (1.0 + rnd_thrust_turb_amp * turb)

        # ---- 2) Slow pilot-like attitude offsets ----
        pitch = rnd_pilot_pitch_amp * np.sin(2*np.pi*rnd_pilot_pitch_freq*t + 0.4)
        roll  = rnd_pilot_roll_amp  * np.cos(2*np.pi*rnd_pilot_roll_freq*t  + 1.2)

        # Small lateral force from tilt (Fz * sin(tilt)) plus a slow gust
        Fx = Fz * np.sin(pitch) + rnd_lat_amp * np.sin(2*np.pi*rnd_lat_freq_x*t + 0.9)
        Fy = Fz * np.sin(roll)  + rnd_lat_amp * np.cos(2*np.pi*rnd_lat_freq_y*t + 2.1)

        # ---- 3) Prop-wash jitter on forces ----
        jit = rnd_jitter_amp_F * np.sin(2*np.pi*rnd_jitter_freq_F*t)
        Fx += jit
        Fy += jit * 0.6
        Fz += jit

        F = np.array([Fx, Fy, Fz])

        # ---- 4) Residual torques + jitter ----
        Tx = rnd_tor_x_amp * np.sin(2*np.pi*rnd_tor_x_freq*t + 0.7) \
             + rnd_jitter_amp_T * np.sin(2*np.pi*rnd_jitter_freq_T*t)
        Ty = rnd_tor_y_amp * np.cos(2*np.pi*rnd_tor_y_freq*t + 1.3) \
             + rnd_jitter_amp_T * np.cos(2*np.pi*rnd_jitter_freq_T*t*0.8)
        Tz = rnd_tor_z_amp * np.sin(2*np.pi*rnd_tor_z_freq*t + 2.1)
        T = np.array([Tx, Ty, Tz])

    else:
        print(f"WARNING: Unknown profile '{FORCE_PROFILE}'. Applying zero wrench.")
        F = np.zeros(3)
        T = np.zeros(3)

    return F, T

print(f"\nSelected force profile: '{FORCE_PROFILE}'")
if FORCE_PROFILE == 'sine_decay':
    print(f"  Force Amps: Fx={amp_x}, Fy={amp_y}, Fz={amp_z}, freq={freq}, tau={tau}, Fz_const={Fz_const}")
    print(f"  Torque Amps: Tx={tor_amp_x}, Ty={tor_amp_y}, Tz={tor_amp_z}")
elif FORCE_PROFILE == 'ramp':
    print(f"  Force Amps: Fx={ramp_amp_x}, Fy={ramp_amp_y}, Fz={ramp_amp_z}, duration={ramp_duration}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'step':
    print(f"  Force Amps: Fx={step_amp_x}, Fy={step_amp_y}, Fz={step_amp_z}, time={step_time}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'impulse':
    print(f"  Force Peaks: Fx={impulse_peak_x}, Fy={impulse_peak_y}, Fz={impulse_peak_z}, times={impulse_times}, width={impulse_width}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'chirp':
    print(f"  Force Amps: Fx={amp_x}, Fy={amp_y}, Fz={amp_z}, freq_start={freq_start}, freq_end={freq_end}, duration={chirp_duration}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'drone_mimic':
    print(f"  thrust_hover={thrust_hover}, thrust_noise={thrust_noise}, noise_freq={noise_freq}")
elif FORCE_PROFILE == 'realistic':
    print(f"  hover thrust = {rnd_hover_thrust_N:.4f} N (drone mass {rnd_drone_mass} kg)")
    print(f"  thrust turbulence = ±{rnd_thrust_turb_amp*100:.0f} %  "
          f"({len(rnd_thrust_turb_spec)} gust bands)")
    print(f"  pilot pitch/roll = ±{np.degrees(rnd_pilot_pitch_amp):.1f}° / "
          f"±{np.degrees(rnd_pilot_roll_amp):.1f}°")
    print(f"  lateral gust = ±{rnd_lat_amp*1000:.0f} mN")
    print(f"  residual torques = ±({rnd_tor_x_amp*1000:.1f}, "
          f"{rnd_tor_y_amp*1000:.1f}, {rnd_tor_z_amp*1000:.1f}) mNm")
else:
    print("  (No specific parameters)")

# ----------------------------------------------------------------------
# 3. SIMULATION SETUP
# ----------------------------------------------------------------------
duration = 9.0
dt = model.opt.timestep
csv_filename = "gimbal_force_comparison.csv"

headers = [
    "Time",
    "Fx_app_arm", "Fy_app_arm", "Fz_app_arm",
    "Tx_app_arm", "Ty_app_arm", "Tz_app_arm",
    "LC_sensor_x", "LC_sensor_y", "LC_sensor_z",
    "Yaw", "Pitch", "Roll",
    "Fx_recon_arm", "Fy_recon_arm", "Fz_recon_arm"
]
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

# ----------------------------------------------------------------------
# 4. RUN SIMULATION
# ----------------------------------------------------------------------
print("\nStarting simulation. Viewer will open shortly...\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    start_time = data.time
    step_counter = 0

    while viewer.is_running() and data.time - start_time < duration:
        step_start = time.time()
        t = data.time - start_time

        # ---- 4a. Compute wrench in ARM frame ----
        F_arm, T_arm = get_applied_wrench(t)

        # ---- 4b. Rotate wrench to WORLD frame ----
        R_arm = data.xmat[arm_body_id].reshape(3, 3)
        F_world = R_arm @ F_arm
        T_world = R_arm @ T_arm

        # ---- 4c. Apply wrench to the arm at the site location ----
        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data,
            F_world, T_world,
            force_local_pos,
            arm_body_id,
            data.qfrc_applied
        )

        # ------------------------------------------------------------------
        # Lock pitch and roll joints (sanity check)
        # ------------------------------------------------------------------
        pitch_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "3-DOF-Assm_3-DOF-mount-v1_Pitch")
        roll_joint_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Bearing_6x13x5-Rear-v1_Revolute-9")
        model.jnt_range[pitch_joint_id] = [0.0, 0.0]
        model.jnt_limited[pitch_joint_id] = True
        model.jnt_range[roll_joint_id] = [0.0, 0.0]
        model.jnt_limited[roll_joint_id] = True
        # ------------------------------------------------------------------

        # ---- 4d. Step physics ----
        mujoco.mj_step(model, data)

        # ---- 4e. Read MuJoCo sensor (load cell in world frame) ----
        LC_sensor = data.sensordata[sensor_adr:sensor_adr+3].copy()

        # ---- 4f. RECONSTRUCT APPLIED FORCE FROM SENSOR (ANALYTIC) ----
        #   F_applied_world = M * a_com - F_sensor
        # where a_com is obtained ANALYTICALLY from MuJoCo's data.cacc.
        # cacc already excludes gravity, so gravity cancels on both sides
        # of Newton's 2nd law for the whole system.
        a_com = (body_masses[:, None] * data.cacc[:, 3:6]).sum(axis=0) / total_mass

        F_applied_recon_world = total_mass * a_com - LC_sensor
        F_applied_recon_arm   = R_arm.T @ F_applied_recon_world

        # ---- 4g. Log data ----
        if step_counter % 10 == 0:
            qpos = data.qpos[:3].copy()
            row = [data.time] + list(F_arm) + list(T_arm) + list(LC_sensor) + list(qpos) + list(F_applied_recon_arm)
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            if int(t * 2) != int((t - dt*10) * 2):
                print(f"t={t:.2f}s | F_arm=({F_arm[0]:.3f}, {F_arm[1]:.3f}, {F_arm[2]:.3f}) N, "
                      f"T_arm=({T_arm[0]:.4f}, {T_arm[1]:.4f}, {T_arm[2]:.4f}) Nm | "
                      f"LC_sensor=({LC_sensor[0]:.3f}, {LC_sensor[1]:.3f}, {LC_sensor[2]:.3f}) N | "
                      f"Recon Fz={F_applied_recon_arm[2]:.3f} N | "
                      f"Angles (rad): Yaw={qpos[0]:.2f}, Pitch={qpos[1]:.2f}, Roll={qpos[2]:.2f}")

        step_counter += 1

        # ---- 4h. Visualize force arrow (in WORLD frame) ----
        world_pos = data.site_xpos[app_site_id]
        viewer.user_scn.ngeom = 0
        arrow_len = np.linalg.norm(F_world) * 0.2
        if arrow_len > 0.001:
            dir_vec = F_world / np.linalg.norm(F_world)
            z_axis = dir_vec
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
            if np.linalg.norm(x_axis) < 0.1:
                x_axis = np.cross(np.array([1, 0, 0]), z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            rot_mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.005, 0.005, arrow_len]),
                pos=world_pos,
                mat=rot_mat,
                rgba=np.array([1.0, 0.0, 0.0, 0.9])
            )
            viewer.user_scn.ngeom += 1

        viewer.sync()
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print(f"\nSimulation complete. Data saved to '{csv_filename}'.")

# ----------------------------------------------------------------------
# 5. PLOTTING
# ----------------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

col_names = ['Time', 'Fx_app', 'Fy_app', 'Fz_app', 'Tx_app', 'Ty_app', 'Tz_app',
             'LC_sx', 'LC_sy', 'LC_sz', 'Yaw', 'Pitch', 'Roll',
             'Fx_recon', 'Fy_recon', 'Fz_recon']
data_csv = np.genfromtxt(csv_filename, delimiter=',', skip_header=1, names=col_names)

t = data_csv['Time']
Fx_app, Fy_app, Fz_app = data_csv['Fx_app'], data_csv['Fy_app'], data_csv['Fz_app']
Tx_app, Ty_app, Tz_app = data_csv['Tx_app'], data_csv['Ty_app'], data_csv['Tz_app']
LC_sx, LC_sy, LC_sz = data_csv['LC_sx'], data_csv['LC_sy'], data_csv['LC_sz']
Yaw, Pitch, Roll = data_csv['Yaw'], data_csv['Pitch'], data_csv['Roll']
Fx_recon, Fy_recon, Fz_recon = data_csv['Fx_recon'], data_csv['Fy_recon'], data_csv['Fz_recon']

# ---- Figure 1: Applied Force (ARM frame) ----
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(t, Fx_app, label=r'$F_x$ (arm)', linewidth=1.5)
ax1.plot(t, Fy_app, label=r'$F_y$ (arm)', linewidth=1.5)
ax1.plot(t, Fz_app, label=r'$F_z$ (arm)', linewidth=1.5)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Force (N)')
ax1.set_title('Applied Force (Profile: ' + FORCE_PROFILE + ')')
ax1.grid(True, alpha=0.4); ax1.legend(loc='best')
fig1.tight_layout(); fig1.savefig('plot_1_applied_force.png', dpi=200)
plt.show()

# ---- Figure 2: Applied Torque (ARM frame) ----
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(t, Tx_app, label=r'$\tau_x$ (arm)', linewidth=1.5)
ax2.plot(t, Ty_app, label=r'$\tau_y$ (arm)', linewidth=1.5)
ax2.plot(t, Tz_app, label=r'$\tau_z$ (arm)', linewidth=1.5)
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Torque (Nm)')
ax2.set_title('Applied Torque (Profile: ' + FORCE_PROFILE + ')')
ax2.grid(True, alpha=0.4); ax2.legend(loc='best')
fig2.tight_layout(); fig2.savefig('plot_2_applied_torque.png', dpi=200)
plt.show()

# ---- Figure 3: Load cell sensor ----
fig3, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, LC_sx, 'b-', linewidth=1.5, label='MuJoCo Sensor')
axes[0].set_ylabel(r'$F_x$ (N)'); axes[0].grid(True, alpha=0.4); axes[0].legend(loc='best')
axes[1].plot(t, LC_sy, 'b-', linewidth=1.5, label='MuJoCo Sensor')
axes[1].set_ylabel(r'$F_y$ (N)'); axes[1].grid(True, alpha=0.4); axes[1].legend(loc='best')
axes[2].plot(t, LC_sz, 'b-', linewidth=1.5, label='MuJoCo Sensor')
axes[2].set_ylabel(r'$F_z$ (N)'); axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.4); axes[2].legend(loc='best')
fig3.suptitle('Load Cell Reaction Force (MuJoCo Sensor)', fontsize=16)
fig3.tight_layout(rect=[0, 0, 1, 0.97])
fig3.savefig('plot_3_loadcell_sensor.png', dpi=200)
plt.show()

# ---- Figure 4: Gimbal joint angles ----
fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(t, Yaw, label='Yaw', linewidth=1.5)
ax4.plot(t, Pitch, label='Pitch', linewidth=1.5)
ax4.plot(t, Roll, label='Roll', linewidth=1.5)
ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Joint Angle (rad)')
ax4.set_title('Gimbal Joint Motion (3 DOF)')
ax4.grid(True, alpha=0.4); ax4.legend(loc='best')
fig4.tight_layout(); fig4.savefig('plot_4_joint_angles.png', dpi=200)
plt.show()

# ---- Figure 5: VALIDATION – Commanded vs Reconstructed ----
fig5, axes5 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes5[0].plot(t, Fx_app, 'b-', linewidth=1.5, label='Commanded Fx')
axes5[0].plot(t, Fx_recon, 'r--', linewidth=1.2, label='Reconstructed Fx (cacc)')
axes5[0].set_ylabel(r'$F_x$ (N)'); axes5[0].grid(True, alpha=0.4); axes5[0].legend(loc='best')
axes5[1].plot(t, Fy_app, 'b-', linewidth=1.5, label='Commanded Fy')
axes5[1].plot(t, Fy_recon, 'r--', linewidth=1.2, label='Reconstructed Fy (cacc)')
axes5[1].set_ylabel(r'$F_y$ (N)'); axes5[1].grid(True, alpha=0.4); axes5[1].legend(loc='best')
axes5[2].plot(t, Fz_app, 'b-', linewidth=1.5, label='Commanded Fz')
axes5[2].plot(t, Fz_recon, 'r--', linewidth=1.2, label='Reconstructed Fz (cacc)')
axes5[2].set_xlabel('Time (s)'); axes5[2].set_ylabel(r'$F_z$ (N)')
axes5[2].grid(True, alpha=0.4); axes5[2].legend(loc='best')
fig5.suptitle('Validation: Commanded vs Reconstructed Applied Force (arm frame, analytic cacc)',
              fontsize=15)
fig5.tight_layout(rect=[0, 0, 1, 0.97])
fig5.savefig('plot_5_validation.png', dpi=200)
plt.show()

print("\nPlots saved:")
print("  - plot_1_applied_force.png")
print("  - plot_2_applied_torque.png")
print("  - plot_3_loadcell_sensor.png")
print("  - plot_4_joint_angles.png")
print("  - plot_5_validation.png")
print("\n(All forces and torques are defined in the ARM frame.)")
print("Validation plot compares commanded vs reconstructed applied force from sensor.")