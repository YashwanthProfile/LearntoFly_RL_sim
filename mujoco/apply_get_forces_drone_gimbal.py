"""
Drive motors m1–m4 with a prescribed RPM profile and plot:
  1. motor RPMs,
  2. drone body thrust Fz and moments τx, τy, τz,
  3. 3-axis load-cell reaction forces.

Actuation is done by applying per-motor thrust as an external force:
    F_i = k_f * omega_i^2 ,   omega_i = RPM_i * 2*pi/60 ,   RPM_i = u_i * MAX_RPM
so no <actuator> entries are needed in the XML.
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. USER SELECTION
# ----------------------------------------------------------------------
MOTOR_PROFILE = 'drone_mimic'      # 'constant' | 'ramp' | 'step' | 'sine_decay'
                                   # 'chirp' | 'impulse' | 'drone_mimic'

# ---- Crazyflie 2.1 datasheet-based constants ----
MAX_RPM = 25000.0    # rpm at full throttle (u = 1)
KF      = 1.8e-8     # thrust coefficient [N/(rad/s)^2]

# Hover thrust per motor -> derived hover RPM
hover_N     = 0.052
omega_hover = np.sqrt(hover_N / KF)
rpm_hover   = omega_hover * 60.0 / (2*np.pi)
u_hover     = rpm_hover / MAX_RPM

# ---- Profile amplitudes, all in "fraction of MAX_RPM" ----
# Ramp
ramp_duration = 2.0
ramp_u_target = 0.20
# Step
step_time     = 1.0
step_u_target = 0.20
# Sine-decay
sine_base_u   = 0.10
sine_base_amp = 0.05
sine_dpitch   = 0.03
sine_droll    = 0.03
sine_dyaw     = 0.01
sine_freq     = 1.0
sine_tau      = 1.0
# Chirp
chirp_base_u  = 0.10
chirp_amp     = 0.05
chirp_dpitch  = 0.03
chirp_droll   = 0.03
chirp_dyaw    = 0.01
freq_start    = 0.2
freq_end      = 1.0
chirp_duration = 5.0
# Impulse
impulse_times  = [1.0, 5.0]
impulse_width  = 0.10
impulse_peak   = 0.10
impulse_dpitch = 0.03
impulse_droll  = 0.03
impulse_dyaw   = 0.02
# Drone mimic
thrust_noise  = 0.1
noise_freq    = 0.30
mimic_dpitch  = 0.010
mimic_droll   = 0.010
mimic_dyaw    = 0.005

# ----------------------------------------------------------------------
# 1. LOAD MODEL & RESET
# ----------------------------------------------------------------------
model_path = '3DOF_drone_gimbal.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print("Reset to 'home' keyframe.")
else:
    print("Warning: 'home' not found, using default state.")

base_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BaseGrounded-v2")
drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Crazyflie_Combined-Body-v2")
if drone_id == -1:
    raise RuntimeError("Body 'Crazyflie_Combined-Body-v2' not found.")

motor_names = ["motor1", "motor2", "motor3", "motor4"]
site_ids    = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in motor_names}
for n, sid in site_ids.items():
    if sid == -1:
        raise RuntimeError(f"Site '{n}' not found in model.")

# Optional load-cell sensor
sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "loadcell_force")
if sensor_id != -1:
    sensor_adr = model.sensor_adr[sensor_id]
    use_sensor = True
    print(f"Load-cell sensor found: addr={sensor_adr}")
else:
    use_sensor = False
    print("No 'loadcell_force' sensor found — reporting analytic load-cell forces.")

g          = 9.81
total_mass = np.sum(model.body_mass)
print(f"Total mass: {total_mass:.4f} kg  (weight = {total_mass*g:.3f} N)")
print(f"Hover per motor: {hover_N} N -> {rpm_hover:.0f} rpm  (u_hover = {u_hover:.3f})")

# ----------------------------------------------------------------------
# 2. RPM PROFILE GENERATOR
# ----------------------------------------------------------------------
def get_motor_rpms(t):
    """Return [rpm1, rpm2, rpm3, rpm4] for time t (X-mixer applied)."""
    base_u, dpitch, droll, dyaw = 0.0, 0.0, 0.0, 0.0

    if MOTOR_PROFILE == 'constant':
        base_u = u_hover

    elif MOTOR_PROFILE == 'ramp':
        base_u = ramp_u_target * min(t / ramp_duration, 1.0)

    elif MOTOR_PROFILE == 'step':
        base_u = step_u_target * (1.0 if t >= step_time else 0.0)

    elif MOTOR_PROFILE == 'sine_decay':
        env = np.exp(-t / sine_tau)
        base_u = sine_base_u + sine_base_amp * np.sin(2*np.pi*sine_freq*t) * env
        dpitch = sine_dpitch * np.sin(2*np.pi*sine_freq*t*1.2) * env
        droll  = sine_droll  * np.cos(2*np.pi*sine_freq*t*0.9) * env
        dyaw   = sine_dyaw   * np.sin(2*np.pi*sine_freq*t*0.7) * env

    elif MOTOR_PROFILE == 'chirp':
        if t < chirp_duration:
            phase = 2*np.pi*(freq_start*t
                             + 0.5*(freq_end - freq_start)*t**2/chirp_duration)
        else:
            phase = 2*np.pi*(freq_start*chirp_duration
                             + 0.5*(freq_end - freq_start)*chirp_duration)
            phase += 2*np.pi*freq_end*(t - chirp_duration)
        base_u = chirp_base_u + chirp_amp * np.sin(phase)
        dpitch = chirp_dpitch * np.sin(phase*1.1)
        droll  = chirp_droll  * np.cos(phase*0.9)
        dyaw   = chirp_dyaw   * np.sin(phase*0.7)

    elif MOTOR_PROFILE == 'impulse':
        base_u = u_hover
        for t0 in impulse_times:
            if t0 <= t <= t0 + impulse_width:
                s = np.sin(np.pi * (t - t0) / impulse_width)
                base_u += impulse_peak   * s
                dpitch += impulse_dpitch * s
                droll  += impulse_droll  * s
                dyaw   += impulse_dyaw   * s

    elif MOTOR_PROFILE == 'drone_mimic':
        base_u = u_hover + thrust_noise * np.sin(2*np.pi*noise_freq*t)
        dpitch = mimic_dpitch * np.sin(2*np.pi*0.20*t)
        droll  = mimic_droll  * np.cos(2*np.pi*0.15*t)
        dyaw   = mimic_dyaw   * np.sin(2*np.pi*0.10*t)

    else:
        print(f"WARNING: unknown profile '{MOTOR_PROFILE}'. Using hover.")
        base_u = u_hover

    u1 = np.clip(base_u + dpitch + droll - dyaw, 0.0, 1.0)
    u2 = np.clip(base_u - dpitch - droll - dyaw, 0.0, 1.0)
    u3 = np.clip(base_u + dpitch - droll + dyaw, 0.0, 1.0)
    u4 = np.clip(base_u - dpitch + droll + dyaw, 0.0, 1.0)
    return [u1*MAX_RPM, u2*MAX_RPM, u3*MAX_RPM, u4*MAX_RPM]

# ----------------------------------------------------------------------
# 3. SIMULATION SETTINGS & CSV
# ----------------------------------------------------------------------
duration = 10.0
dt       = model.opt.timestep
csv_filename = "loadcell_experiment.csv"

headers = ["Time",
           "rpm1", "rpm2", "rpm3", "rpm4",
           "Drone_Fz_body",
           "Drone_Taux_body", "Drone_Tauy_body", "Drone_Tauz_body",
           "Loadcell_Fx", "Loadcell_Fy", "Loadcell_Fz"]
with open(csv_filename, 'w', newline='') as f:
    csv.writer(f).writerow(headers)

print(f"\nMotor profile: '{MOTOR_PROFILE}'")
print(f"Simulation: {duration} s, dt = {dt} s\n")

# ----------------------------------------------------------------------
# 4. RUN SIMULATION
# ----------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    start_time = data.time
    step_counter = 0

    while viewer.is_running() and data.time - start_time < duration:
        step_start = time.time()
        t = data.time - start_time

        # ---- A) RPM commands and per-motor thrust ----
        rpm_list = get_motor_rpms(t)
        omegas   = [rpm * 2*np.pi / 60.0 for rpm in rpm_list]
        thrusts  = [KF * w*w for w in omegas]

        # ---- B) Aggregate thrust wrench onto drone body ----
        data.xfrc_applied[:] = 0.0
        drone_com_world = data.xipos[drone_id]
        F_body_total = np.zeros(3)
        M_body_total = np.zeros(3)
        thrusts_world = []
        for name, F_mag in zip(motor_names, thrusts):
            sid = site_ids[name]
            site_mat = data.site_xmat[sid].reshape(3, 3)
            # thrust along local -Z of the site
            F_world = F_mag * (-site_mat[:, 2])
            r_world = data.site_xpos[sid] - drone_com_world
            M_world = np.cross(r_world, F_world)
            F_body_total += F_world
            M_body_total += M_world
            thrusts_world.append(F_world)
        data.xfrc_applied[drone_id, 0:3] = F_body_total
        data.xfrc_applied[drone_id, 3:6] = M_body_total

        # ---- C) Step physics ----
        mujoco.mj_step(model, data)

        # ---- D) Log every 10 steps ----
        if step_counter % 10 == 0:
            # Drone body Fz and moments (analytic, in drone body frame)
            total_thrust  = sum(thrusts)
            Drone_Fz_body = -total_thrust         # upward = positive

            drone_mat = data.xmat[drone_id].reshape(3, 3)
            Taux = Tauy = Tauz = 0.0
            for name, F in zip(motor_names, thrusts):
                local_pos = drone_mat.T @ (data.site_xpos[site_ids[name]]
                                           - data.xipos[drone_id])
                Taux += local_pos[1] * (-F)
                Tauy += -local_pos[0] * (-F)
            Drone_Taux_body, Drone_Tauy_body, Drone_Tauz_body = Taux, Tauy, Tauz

            # Load-cell reaction
            if use_sensor:
                LC = data.sensordata[sensor_adr:sensor_adr+3].copy()
                Loadcell_Fx, Loadcell_Fy, Loadcell_Fz = LC
            else:
                gravity_vec = np.array([0.0, 0.0, -total_mass*g])
                thrust_vec  = np.sum(thrusts_world, axis=0)
                Loadcell_Fx, Loadcell_Fy, Loadcell_Fz = -(gravity_vec + thrust_vec)

            row = [data.time] + rpm_list + [
                Drone_Fz_body,
                Drone_Taux_body, Drone_Tauy_body, Drone_Tauz_body,
                Loadcell_Fx, Loadcell_Fy, Loadcell_Fz]
            with open(csv_filename, 'a', newline='') as f:
                csv.writer(f).writerow(row)

            if int(t*2) != int((t - dt*10)*2):
                print(f"t={t:5.2f}s | rpm=[{rpm_list[0]:5.0f},{rpm_list[1]:5.0f},"
                      f"{rpm_list[2]:5.0f},{rpm_list[3]:5.0f}] "
                      f"| Fz_drone={Drone_Fz_body:+.4f} N "
                      f"| LC_Fz={Loadcell_Fz:+.4f} N")

        step_counter += 1

        # ---- E) Thrust arrows ----
        viewer.user_scn.ngeom = 0
        for name, F_mag in zip(motor_names, thrusts):
            sid      = site_ids[name]
            pos      = data.site_xpos[sid]
            site_mat = data.site_xmat[sid].reshape(3, 3)
            arrow_len = max(F_mag * 0.5, 0.001)
            mat_rot   = site_mat @ np.diag([1, -1, -1])
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.003, 0.003, arrow_len]),
                pos=pos, mat=mat_rot.flatten(),
                rgba=np.array([1.0, 0.1, 0.0, 0.9]))
            viewer.user_scn.ngeom += 1

        viewer.sync()
        time_until = dt - (time.time() - step_start)
        if time_until > 0:
            time.sleep(time_until)

print(f"\nSimulation complete. Data saved to '{csv_filename}'.")

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
col_names = ['Time', 'rpm1', 'rpm2', 'rpm3', 'rpm4',
             'Drone_Fz_body', 'Drone_Taux_body', 'Drone_Tauy_body', 'Drone_Tauz_body',
             'Loadcell_Fx', 'Loadcell_Fy', 'Loadcell_Fz']
d = np.genfromtxt(csv_filename, delimiter=',', skip_header=1, names=col_names)
t = d['Time']

# Plot 1: Motor RPMs
plt.figure(figsize=(10, 4))
for i in range(4):
    plt.plot(t, d[f'rpm{i+1}'], label=f'Motor {i+1}', lw=1.5)
plt.xlabel('Time (s)'); plt.ylabel('RPM')
plt.title(f'1. Motor RPM (profile: {MOTOR_PROFILE})')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig('plot_1_rpm.png', dpi=150); plt.show()

# Plot 2: Drone body thrust and moments
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, d['Drone_Fz_body'], 'r-', lw=1.5)
ax[0].set_ylabel('Thrust Fz (N)'); ax[0].set_title('2a. Drone Body Thrust')
ax[0].grid(True, alpha=0.3)
ax[1].plot(t, d['Drone_Taux_body'], label='τx (roll)',  lw=1.2)
ax[1].plot(t, d['Drone_Tauy_body'], label='τy (pitch)', lw=1.2)
ax[1].plot(t, d['Drone_Tauz_body'], label='τz (yaw)',   lw=1.2)
ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('Moment (Nm)')
ax[1].set_title('2b. Drone Body Moments')
ax[1].grid(True, alpha=0.3); ax[1].legend()
plt.tight_layout(); plt.savefig('plot_2_drone_Fz_tau.png', dpi=150); plt.show()

# Plot 3: Load-cell reaction forces
plt.figure(figsize=(10, 4))
plt.plot(t, d['Loadcell_Fx'], label='Fx', lw=1.5)
plt.plot(t, d['Loadcell_Fy'], label='Fy', lw=1.5)
plt.plot(t, d['Loadcell_Fz'], label='Fz', lw=1.5)
plt.xlabel('Time (s)'); plt.ylabel('Force (N)')
plt.title('3. Load Cell Readings (3-axis, World Frame)')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig('plot_3_loadcell_Fxyz.png', dpi=150); plt.show()

print("\nPlots saved: plot_1_rpm.png, plot_2_drone_Fz_tau.png, plot_3_loadcell_Fxyz.png")