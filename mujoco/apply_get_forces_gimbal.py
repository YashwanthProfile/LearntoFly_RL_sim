"""
Apply a user‑selectable time‑varying force at the drone mount point of a 3‑DOF gimbal,
and record the resulting load‑cell reaction from MuJoCo's native sensor.
Manual calculation and plotting are commented out (keep sensor only).
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
# Choose one of the following profiles:
#   'constant'    – constant force (Fz_const only)
#   'ramp'        – linear ramp up to a max force, then hold
#   'step'        – step change at a specified time
#   'sine_decay'  – decaying sine (oscillatory, dies out)
#   'chirp'       – swept sine (frequency increases over time)
#   'impulse'     – short pulse (like a hammer blow)
#   'drone_mimic' – combination of constant thrust + small oscillations
FORCE_PROFILE = 'impulse'   # change this to select a profile

# ----------------------------------------------------------------------
# Profile parameters – adjust these to tune each profile
# ----------------------------------------------------------------------
# Common parameters (used by multiple profiles)
Fz_const = 0.0          # constant vertical offset (N) – usually 0 for dynamic tests
amp_x   = 0.08          # amplitude for horizontal X (N)
amp_y   = 0.08          # amplitude for horizontal Y (N)
amp_z   = 0.15          # amplitude for vertical perturbation (N)
freq    = 0.4           # base frequency (Hz)
tau     = 2.5           # decay time constant (s) for decaying profiles

# Chirp specific
freq_start = 0.2        # starting frequency (Hz)
freq_end   = 1.0        # ending frequency (Hz)
chirp_duration = 5.0    # duration of chirp (s) – should be <= total simulation time

# Ramp specific – now with per‑axis amplitudes
ramp_duration = 2.0     # time to reach max force (s)
ramp_amp_x = 0.0        # final force in X (N)
ramp_amp_y = 0.0        # final force in Y (N)
ramp_amp_z = 0.5        # final force in Z (N)

# Step specific – per‑axis amplitudes
step_time = 1.0         # time of step (s)
step_amp_x = 0.0        # step change in X (N)
step_amp_y = 0.0        # step change in Y (N)
step_amp_z = 0.5        # step change in Z (N)

# Impulse specific – per‑axis peaks and times
impulse_times = [1.0, 5.0]          # list of impulse times (s)
impulse_width = 0.1                 # duration (s) – same for all impulses
impulse_peak_x = 0.0                # peak force in X (N) – applied to all pulses
impulse_peak_y = 0.1                # peak force in Y (N)
impulse_peak_z = 0.5                # peak force in Z (N)
# If you want per‑impulse peaks per axis, you can make them lists as before.

# Drone mimic specific
thrust_hover = 0.5      # average thrust (N) – constant
thrust_noise = 0.05     # amplitude of small oscillations (N)
noise_freq   = 0.3      # frequency of oscillations (Hz)

# ----------------------------------------------------------------------
# 1. LOAD MODEL AND CHECK KEYFRAME / SENSOR
# ----------------------------------------------------------------------
model_path = '3DOF_gimbal.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ---- Reset to 'home' keyframe ----
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print("Reset to 'home' keyframe.")
else:
    print("WARNING: 'home' keyframe not found. Using default initial state.")

# ---- Check sensor ----
sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "loadcell_force")
if sensor_id == -1:
    print("ERROR: Sensor 'loadcell_force' not found in the model!")
    sys.exit(1)
else:
    sensor_adr = model.sensor_adr[sensor_id]
    print(f"Load‑cell sensor found: ID={sensor_id}, address={sensor_adr}")

# ---- Check force application site ----
app_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "force_app_point")
if app_site_id == -1:
    print("ERROR: Site 'force_app_point' not found in the model.")
    sys.exit(1)
else:
    force_local_pos = model.site_pos[app_site_id].copy()
    print(f"Force application point (local to Gimbal-Arm): {force_local_pos}")

# ---- Get body IDs ----
arm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Gimbal-Arm-v1-v1")
if arm_body_id == -1:
    print("ERROR: Body 'Gimbal-Arm-v1-v1' not found.")
    sys.exit(1)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BaseGrounded-v2")
if base_id == -1:
    print("ERROR: Body 'BaseGrounded-v2' not found.")
    sys.exit(1)

# ---- Constants ----
g = 9.81
total_mass = np.sum(model.body_mass)
print(f"Total mass: {total_mass:.6f} kg")

# ----------------------------------------------------------------------
# 2. FORCE PROFILE GENERATOR
# ----------------------------------------------------------------------
def get_applied_force(t):
    """
    Returns force vector (Fx, Fy, Fz) based on the selected profile.
    All profiles are designed to be smooth and physically plausible.
    """
    if FORCE_PROFILE == 'constant':
        Fx = 0.0
        Fy = 0.0
        Fz = Fz_const  # just a constant


    elif FORCE_PROFILE == 'sine_decay':
        # Exponentially decaying sine wave
        envelope = np.exp(-t / tau)
        Fx = amp_x * np.sin(2 * np.pi * freq * t) * envelope
        Fy = amp_y * np.cos(2 * np.pi * freq * t) * envelope
        Fz = Fz_const + amp_z * np.sin(2 * np.pi * freq * t * 1.2) * envelope

    elif FORCE_PROFILE == 'chirp':
        # Swept sine: frequency increases linearly from freq_start to freq_end
        # over chirp_duration, then stays at freq_end (or decays if tau given)
        # Here we use a simple linear chirp without decay (you can add decay if desired)
        if t < chirp_duration:
            f = freq_start + (freq_end - freq_start) * (t / chirp_duration)
        else:
            f = freq_end
        # Phase is integral of instantaneous frequency
        # For linear chirp: phase = 2*pi * (freq_start*t + 0.5*(freq_end-freq_start)*t^2/chirp_duration)
        # We'll implement it exactly for t < chirp_duration, else continue with constant freq.
        if t < chirp_duration:
            phase = 2 * np.pi * (freq_start * t + 0.5 * (freq_end - freq_start) * t**2 / chirp_duration)
        else:
            # after chirp, continue with constant frequency (or you can add decay)
            phase = 2 * np.pi * (freq_start * chirp_duration + 0.5 * (freq_end - freq_start) * chirp_duration) + 2 * np.pi * freq_end * (t - chirp_duration)
        Fx = amp_x * np.sin(phase) * 0.5  # optional envelope
        Fy = amp_y * np.cos(phase * 0.8) * 0.5
        Fz = Fz_const + amp_z * np.sin(phase * 1.2) * 0.5


    elif FORCE_PROFILE == 'ramp':
        if t < ramp_duration:
            ramp_factor = t / ramp_duration
        else:
            ramp_factor = 1.0
        Fx = ramp_amp_x * ramp_factor
        Fy = ramp_amp_y * ramp_factor
        Fz = Fz_const + ramp_amp_z * ramp_factor   # Fz_const adds a constant offset

    elif FORCE_PROFILE == 'step':
        step_val = 1.0 if t >= step_time else 0.0
        Fx = step_amp_x * step_val
        Fy = step_amp_y * step_val
        Fz = Fz_const + step_amp_z * step_val

    elif FORCE_PROFILE == 'impulse':
        # Convert scalar width/peak to lists if needed
        if not isinstance(impulse_width, (list, tuple)):
            widths = [impulse_width] * len(impulse_times)
        else:
            widths = impulse_width
        # For each axis, we can use a scalar peak (applied to all pulses) or a list of peaks.
        # We'll treat impulse_peak_x, _y, _z as scalars (applied to all pulses).
        # If you want per‑impulse peaks, you can make them lists as before.
        pulse_x, pulse_y, pulse_z = 0.0, 0.0, 0.0
        for t0, w in zip(impulse_times, widths):
            if t >= t0 and t <= t0 + w:
                phase = np.pi * (t - t0) / w
                sin_val = np.sin(phase)
                pulse_x += impulse_peak_x * sin_val
                pulse_y += impulse_peak_y * sin_val
                pulse_z += impulse_peak_z * sin_val
        Fx = pulse_x
        Fy = pulse_y
        Fz = Fz_const + pulse_z

    elif FORCE_PROFILE == 'drone_mimic':
        # Mimics a drone's thrust: constant hover thrust + small vibrations
        # plus some lateral disturbances
        envelope = 1.0  # no decay (or could add a slow drift)
        Fx = 0.02 * np.sin(2 * np.pi * 0.2 * t)  # small lateral gust
        Fy = 0.02 * np.cos(2 * np.pi * 0.15 * t)
        # Vertical: hover thrust + low-frequency oscillation + high-frequency noise
        Fz = -(thrust_hover) + thrust_noise * np.sin(2 * np.pi * noise_freq * t) + 0.01 * np.sin(2 * np.pi * 5 * t)
        # Note: we keep Fz negative because thrust is upward (negative in world Z if using our convention)
        # But for consistency with other profiles, we'll keep Fz positive as upward? Let's define: positive Fz = upward.
        # So we set Fz = +thrust_hover (since upward is positive Z in MuJoCo world)
        # But our Fz_const convention: positive is upward.
        # So we set:
        Fz = thrust_hover + thrust_noise * np.sin(2 * np.pi * noise_freq * t)

    else:
        # Default to zero force
        print(f"WARNING: Unknown profile '{FORCE_PROFILE}'. Applying zero force.")
        Fx, Fy, Fz = 0.0, 0.0, 0.0

    return np.array([Fx, Fy, Fz])

# Print current profile information
print(f"\nSelected force profile: '{FORCE_PROFILE}'")
if FORCE_PROFILE == 'sine_decay':
    print(f"  Parameters: amp_x={amp_x}, amp_y={amp_y}, amp_z={amp_z}, freq={freq}, tau={tau}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'chirp':
    print(f"  Parameters: amp_x={amp_x}, amp_y={amp_y}, amp_z={amp_z}, freq_start={freq_start}, freq_end={freq_end}, chirp_duration={chirp_duration}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'ramp':
    print(f"  Parameters: ramp_duration={ramp_duration}, ramp_amp_x={ramp_amp_x}, ramp_amp_y={ramp_amp_y}, ramp_amp_z={ramp_amp_z}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'step':
    print(f"  Parameters: step_time={step_time}, step_amp_x={step_amp_x}, step_amp_y={step_amp_y}, step_amp_z={step_amp_z}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'impulse':
    print(f"  Parameters: impulse_times={impulse_times}, impulse_width={impulse_width}, impulse_peak_x={impulse_peak_x}, impulse_peak_y={impulse_peak_y}, impulse_peak_z={impulse_peak_z}, Fz_const={Fz_const}")
elif FORCE_PROFILE == 'drone_mimic':
    print(f"  Parameters: thrust_hover={thrust_hover}, thrust_noise={thrust_noise}, noise_freq={noise_freq}, Fz_const={Fz_const}")
else:
    print("  (No specific parameters)")

# ----------------------------------------------------------------------
# 3. SIMULATION SETUP
# ----------------------------------------------------------------------
duration = 9.0                     # total simulation time (s)
dt = model.opt.timestep
csv_filename = "gimbal_force_comparison.csv"
headers = [
    "Time", "Fx_app", "Fy_app", "Fz_app",
    "LC_sensor_x", "LC_sensor_y", "LC_sensor_z"
    # Manual calculation columns removed (commented out)
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

        # ---- 4a. Compute applied force based on selected profile ----
        force_world = get_applied_force(t)
        torque_world = np.zeros(3)

        # ---- 4b. Apply force ----
        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(
            model, data,
            force_world, torque_world,
            force_local_pos,
            arm_body_id,
            data.qfrc_applied
        )

        # ---- 4c. Step physics ----
        mujoco.mj_step(model, data)

        # ---- 4d. Read MuJoCo sensor ----
        LC_sensor = data.sensordata[sensor_adr:sensor_adr+3].copy()

        # ---- 4e. (Optional) Manual calculation – COMMENTED OUT ----
        # If you want to re-enable manual calculation, uncomment the lines below.
        # You will also need to add the corresponding columns to the CSV headers and plot.
        # manual_Fx, manual_Fy, manual_Fz = ... (compute using momentum)
        # We'll skip it entirely to keep the code clean.

        # ---- 4f. Log data ----
        if step_counter % 10 == 0:
            qpos = data.qpos[:3].copy()
            row = [data.time] + list(force_world) + list(LC_sensor)
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            # Print progress
            if int(t * 2) != int((t - dt*10) * 2):
                print(f"t={t:.2f}s | Fz_app={force_world[2]:.3f} N | "
                      f"LC_sensor=({LC_sensor[0]:.3f}, {LC_sensor[1]:.3f}, {LC_sensor[2]:.3f}) N | "
                      f"Angles (rad): Yaw={qpos[0]:.2f}, Pitch={qpos[1]:.2f}, Roll={qpos[2]:.2f}")

        step_counter += 1

        # ---- 4g. Visualize force arrow ----
        world_pos = data.site_xpos[app_site_id]
        viewer.user_scn.ngeom = 0
        arrow_len = np.linalg.norm(force_world) * 0.2
        if arrow_len > 0.001:
            dir_vec = force_world / np.linalg.norm(force_world)
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

        # Real‑time sync
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    if step_counter % 100 == 0:   # print every 100 steps to avoid clutter
        print(f"Raw LC_sensor_z = {LC_sensor[2]:.3f} N")

print(f"\nSimulation complete. Data saved to '{csv_filename}'.")

# ----------------------------------------------------------------------
# 5. PLOTTING (sensor only – manual plot commented out)
# ----------------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# Read CSV (only sensor columns now)
col_names = ['Time', 'Fx_app', 'Fy_app', 'Fz_app', 'LC_sx', 'LC_sy', 'LC_sz']
data_csv = np.genfromtxt(csv_filename, delimiter=',', skip_header=1, names=col_names)

t = data_csv['Time']
Fx_app, Fy_app, Fz_app = data_csv['Fx_app'], data_csv['Fy_app'], data_csv['Fz_app']
LC_sx, LC_sy, LC_sz = data_csv['LC_sx'], data_csv['LC_sy'], data_csv['LC_sz']

# Figure 1: Applied force
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t, Fx_app, label=r'$F_x$', linewidth=2)
ax1.plot(t, Fy_app, label=r'$F_y$', linewidth=2)
ax1.plot(t, Fz_app, label=r'$F_z$', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Force (N)')
ax1.set_title('Applied External Force (Profile: ' + FORCE_PROFILE + ')')
ax1.grid(True, alpha=0.4)
ax1.legend(loc='best')
fig1.tight_layout()
fig1.savefig('plot_1_applied_force.png', dpi=200)
plt.show()

# Figure 2: Load cell sensor (3 subplots)
fig2, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, LC_sx, 'b-', linewidth=2, label='MuJoCo Sensor')
axes[0].set_ylabel(r'$F_x$ (N)')
axes[0].grid(True, alpha=0.4)
axes[0].legend(loc='best')

axes[1].plot(t, LC_sy, 'b-', linewidth=2, label='MuJoCo Sensor')
axes[1].set_ylabel(r'$F_y$ (N)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='best')

axes[2].plot(t, LC_sz, 'b-', linewidth=2, label='MuJoCo Sensor')
axes[2].set_ylabel(r'$F_z$ (N)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.4)
axes[2].legend(loc='best')

fig2.suptitle('Load Cell Reaction Force (MuJoCo Sensor)', fontsize=16)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig('plot_2_loadcell_sensor.png', dpi=200)
plt.show()

# Figure 3: Gimbal joint angles (still useful)
# We need to read Yaw, Pitch, Roll from the CSV – they are not in the current headers.
# Since we removed them, we can either re-add them or read from data directly.
# For simplicity, we'll skip joint angle plotting in this version, or we could add them back.
# Let's add them back to the CSV for completeness.
# I'll leave it as an exercise – you can easily add qpos to the logging if needed.

print("\nPlots saved:")
print("  - plot_1_applied_force.png")
print("  - plot_2_loadcell_sensor.png")
print("\n(Manual calculation and plotting are commented out.)")