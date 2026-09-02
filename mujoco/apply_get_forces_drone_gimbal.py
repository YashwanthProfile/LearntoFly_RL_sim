# Apply prescribed rpm profile to motors m1–4 and plotting the body moments and forces generated against time.
# Also plots the time series of Fx,Fy,Fz at the load cell placement location below the mounting bracket, for sanity check.

import mujoco
import mujoco.viewer
import time
import numpy as np
import csv
import matplotlib.pyplot as plt


# Load model & reset to keyframe

model_path = '3DOF.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print("Reset to 'home' keyframe.")
else:
    print("Warning: 'home' not found, using default state.")

# ids for base and drone bodies
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BaseGrounded-v2")
drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Crazyflie_Combained-Body-v2")

motor_names = ["motor1", "motor2", "motor3", "motor4"]
actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in motor_names}
site_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in motor_names}

gear_z = 0.27          # Thrust gear (N per unit control)
g = 9.81
total_mass = np.sum(model.body_mass)   # Total mass of the entire gimbal + drone

# -------------------------------
# 3. Simulation settings & force profile
# -------------------------------
duration = 10.0
dt = model.opt.timestep

hover_thrust_N = 0.052
base_ctrl = hover_thrust_N / gear_z
perturb_amp_N = 0.015

print(f"Simulation: {duration}s, base thrust = {hover_thrust_N:.3f} N, perturbation = ±{perturb_amp_N:.3f} N")
print(f"Total mass of system: {total_mass:.4f} kg  (gravity force = {total_mass*g:.3f} N)")

# -------------------------------
# 4. CSV logging (Only what you asked for)
# -------------------------------
csv_filename = "loadcell_experiment.csv"
headers = [
    "Time",
    "u1", "u2", "u3", "u4",                     # Motor controls (0-1)
    "Drone_Fz_body",                            # Thrust along drone body Z
    "Drone_Taux_body", "Drone_Tauy_body", "Drone_Tauz_body",  # Body moments
    "Loadcell_Fx", "Loadcell_Fy", "Loadcell_Fz" # 3-axis load cell forces
]

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# -------------------------------
# 5. Launch viewer & run simulation
# -------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    start_time = data.time
    step_counter = 0

    while viewer.is_running() and data.time - start_time < duration:
        step_start = time.time()
        t = data.time - start_time

        # ---- A) Generate control signals ----
        ramp = min(t / 2.0, 1.0)
        perturbation = perturb_amp_N / gear_z * np.sin(2 * np.pi * 0.5 * t)
        ctrl_val = np.clip(ramp * base_ctrl + perturbation, 0, 1)

        # Slight imbalance to excite pitch/roll moments
        u1 = ctrl_val * 1.0
        u2 = ctrl_val * 1.0
        u3 = ctrl_val * 0.95
        u4 = ctrl_val * 1.05
        u_list = [u1, u2, u3, u4]

        for name, u in zip(motor_names, u_list):
            data.ctrl[actuator_ids[name]] = u

        # ---- B) Step physics ----
        mujoco.mj_step(model, data)

        # ---- C) Log data every 10 steps ----
        if step_counter % 10 == 0:
            # --- 1. Drone body-axis thrust (Fz) and moments (Taux, Tauy, Tauz) ---
            thrusts = [u * gear_z for u in u_list]
            total_thrust = sum(thrusts)
            Drone_Fz_body = -total_thrust   # Positive = upward thrust

            # Compute moments about drone COM using motor positions
            drone_pos = data.xpos[drone_id]
            drone_mat = data.xmat[drone_id].reshape(3, 3)
            Taux, Tauy, Tauz = 0.0, 0.0, 0.0
            for name, F in zip(motor_names, thrusts):
                site_id = site_ids[name]
                world_pos = data.site_xpos[site_id]
                local_pos = drone_mat.T @ (world_pos - drone_pos)  # position in body frame
                # Force vector in body frame is [0, 0, -F]
                Taux += local_pos[1] * (-F)
                Tauy += -local_pos[0] * (-F)   # r_x * F_z
                # Tauz from the yaw gear (0.002) is neglected here for clarity
            Drone_Taux_body, Drone_Tauy_body, Drone_Tauz_body = Taux, Tauy, Tauz

            # --- 2. 3-Axis Load Cell Readings (Fx, Fy, Fz) at the base ---
            # Total gravity force on the entire system (world frame)
            gravity_vec = np.array([0, 0, -total_mass * g])

            # Total thrust vector from all motors (world frame)
            total_thrust_vec = np.zeros(3)
            for name, u in zip(motor_names, u_list):
                site_id = site_ids[name]
                F_mag = u * gear_z
                site_mat = data.site_xmat[site_id].reshape(3, 3)
                # Thrust acts along local -Z of the site
                force_world = -F_mag * site_mat[:, 2]
                total_thrust_vec += force_world

            # Net external force on the system = Gravity + Thrust
            net_force = gravity_vec + total_thrust_vec

            # The load cell measures the REACTION force (opposite of net external force)
            Loadcell_Fx, Loadcell_Fy, Loadcell_Fz = -net_force

            # ---- Write to CSV ----
            row = [data.time] + u_list + [
                Drone_Fz_body, Drone_Taux_body, Drone_Tauy_body, Drone_Tauz_body,
                Loadcell_Fx, Loadcell_Fy, Loadcell_Fz
            ]
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Print status every 0.5s
            if int(t * 2) != int((t - dt*10) * 2):
                print(f"t={t:.2f}s | u1={u1:.3f} | Fz_drone={Drone_Fz_body:.4f}N | Loadcell_Fz={Loadcell_Fz:.4f}N")

        step_counter += 1

        # ---- D) Visualize thrust arrows ----
        viewer.user_scn.ngeom = 0
        scale_factor = 0.2
        for name in motor_names:
            site_id = site_ids[name]
            pos = data.site_xpos[site_id]
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            thrust = data.ctrl[actuator_ids[name]] * gear_z
            arrow_len = max(thrust * scale_factor, 0.001)
            # Rotate matrix so arrow points along local -Z (thrust direction)
            mat_rot = site_mat @ np.diag([1, -1, -1])
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.003, 0.003, arrow_len]),
                pos=pos,
                mat=mat_rot.flatten(),
                rgba=np.array([1.0, 0.1, 0.0, 0.9])
            )
            viewer.user_scn.ngeom += 1

        viewer.sync()
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print(f"\nSimulation complete. Data saved to '{csv_filename}'.")

# -------------------------------
# 6. Generate the 3 required plots
# -------------------------------
col_names = [
    'Time', 'u1', 'u2', 'u3', 'u4',
    'Drone_Fz_body', 'Drone_Taux_body', 'Drone_Tauy_body', 'Drone_Tauz_body',
    'Loadcell_Fx', 'Loadcell_Fy', 'Loadcell_Fz'
]
data_csv = np.genfromtxt(csv_filename, delimiter=',', skip_header=1, names=col_names)

t = data_csv['Time']
u1, u2, u3, u4 = data_csv['u1'], data_csv['u2'], data_csv['u3'], data_csv['u4']
Fz_drone = data_csv['Drone_Fz_body']
Taux_drone, Tauy_drone, Tauz_drone = data_csv['Drone_Taux_body'], data_csv['Drone_Tauy_body'], data_csv['Drone_Tauz_body']
Fx_lc, Fy_lc, Fz_lc = data_csv['Loadcell_Fx'], data_csv['Loadcell_Fy'], data_csv['Loadcell_Fz']

# --- Plot 1: Control inputs (0-100%) ---
plt.figure(figsize=(10, 4))
plt.plot(t, u1*100, label='Motor 1', lw=1.5)
plt.plot(t, u2*100, label='Motor 2', lw=1.5)
plt.plot(t, u3*100, label='Motor 3', lw=1.5)
plt.plot(t, u4*100, label='Motor 4', lw=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Control input (%)')
plt.title('1. Motor Control Signals (u1 - u4)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('plot_1_controls.png', dpi=150)
plt.show()

# --- Plot 2: Drone Body Fz and Moments (τx, τy, τz) ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, Fz_drone, 'r-', lw=1.5)
axes[0].set_ylabel('Thrust Fz (N)')
axes[0].set_title('2a. Drone Body Thrust')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, Taux_drone, label='τx (roll)', lw=1.2)
axes[1].plot(t, Tauy_drone, label='τy (pitch)', lw=1.2)
axes[1].plot(t, Tauz_drone, label='τz (yaw)', lw=1.2)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Moment (Nm)')
axes[1].set_title('2b. Drone Body Moments')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
plt.tight_layout()
plt.savefig('plot_2_drone_Fz_tau.png', dpi=150)
plt.show()

# --- Plot 3: 3-Axis Load Cell Forces (Fx, Fy, Fz) ---
plt.figure(figsize=(10, 4))
plt.plot(t, Fx_lc, label='Fx', lw=1.5)
plt.plot(t, Fy_lc, label='Fy', lw=1.5)
plt.plot(t, Fz_lc, label='Fz', lw=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('3. Load Cell Readings (3-axis, World Frame)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('plot_3_loadcell_Fxyz.png', dpi=150)
plt.show()

print("\nAll plots saved:")
print("  - plot_1_controls.png")
print("  - plot_2_drone_Fz_tau.png")
print("  - plot_3_loadcell_Fxyz.png")