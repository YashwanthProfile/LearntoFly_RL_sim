"""
Drive motors m1–m4 with a prescribed RPM profile and plot, in real time:
  1. motor RPMs (all four)
  2. per-motor thrust (N) + total thrust (N)
  3. 3-axis load-cell reaction force (N)

The simulation runs in a MuJoCo passive viewer while a matplotlib window
updates continuously with rolling 3-second traces.  Final publication-quality
plots are saved to disk at the end of the run.

Realistic drone-mimic profile: hover thrust + small pitch/roll/yaw wobble.
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from collections import deque

# ----------------------------------------------------------------------
# 0. USER SELECTION
# ----------------------------------------------------------------------
MOTOR_PROFILE = 'constant'   # 'constant' | 'step' | 'sine_decay' | 'drone_mimic'

# ---- Crazyflie 2.1 datasheet constants ----
MAX_RPM = 25000.0
KF      = 1.8e-8                # N / (rad/s)^2

drone_mass  = 0.034
hover_N     = drone_mass * 9.81 / 4.0
omega_hover = np.sqrt(hover_N / KF)
rpm_hover   = omega_hover * 60.0 / (2*np.pi)
u_hover     = (rpm_hover / MAX_RPM)

# ---- Step ----
step_time     = 0.01
step_u_target = u_hover * 1.05

# ---- Sine-decay ----
sine_base_u   = u_hover
sine_base_amp = 0.05
sine_dpitch   = 0.03
sine_droll    = 0.03
sine_dyaw     = 0.01
sine_freq     = 1.0
sine_tau      = 1.0

# ---- Drone-mimic (realistic) ----
dm_base_amp   = 0.02
dm_pitch_amp  = 0.002
dm_roll_amp   = 0.002
dm_yaw_amp    = 0.001
dm_base_freq  = 0.30
dm_pitch_freq = 0.60
dm_roll_freq  = 0.50
dm_yaw_freq   = 0.20

# ----------------------------------------------------------------------
# 1. LOAD MODEL
# ----------------------------------------------------------------------
model_path = '3DOF_drone_gimbal.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print("Reset to 'home' keyframe.")

roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Roll")
if roll_id == -1:
    raise RuntimeError("Body 'Roll' not found.")

motor_names = ["motor1", "motor2", "motor3", "motor4"]
site_ids    = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
               for n in motor_names}
for n, sid in site_ids.items():
    if sid == -1:
        raise RuntimeError(f"Site '{n}' not found.")

sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "loadcell_force")
if sensor_id == -1:
    raise RuntimeError("Sensor 'loadcell_force' not found.")
sensor_adr = model.sensor_adr[sensor_id]

g          = 9.81
total_mass = np.sum(model.body_mass)
print(f"Total mass = {total_mass:.4f} kg    Drone = {drone_mass:.4f} kg")
print(f"Hover per motor = {hover_N*1000:.1f} mN = {rpm_hover:.0f} RPM  (u_hover = {u_hover:.3f})")
print(f"Total gimbal weight = {total_mass*g:.3f} N  (load-cell should read ≈ this)")

# ----------------------------------------------------------------------
# 2. MOTOR PROFILE
# ----------------------------------------------------------------------
def get_motor_rpms(t):
    base_u, dpitch, droll, dyaw = 0.0, 0.0, 0.0, 0.0

    if MOTOR_PROFILE == 'constant':
        base_u = u_hover * np.sin(2*np.pi*t) #* np.exp(-0.5*t)

    elif MOTOR_PROFILE == 'step':
        base_u = step_u_target if t >= step_time else 0.0

    elif MOTOR_PROFILE == 'sine_decay':
        env = np.exp(-t / sine_tau)
        base_u = sine_base_u + sine_base_amp * np.sin(2*np.pi*sine_freq*t) * env
        dpitch = sine_dpitch * np.sin(2*np.pi*sine_freq*t*1.2) * env
        droll  = sine_droll  * np.cos(2*np.pi*sine_freq*t*0.9) * env
        dyaw   = sine_dyaw   * np.sin(2*np.pi*sine_freq*t*0.7) * env

    elif MOTOR_PROFILE == 'drone_mimic':
        env = np.exp(-10*t)
        base_u = u_hover + dm_base_amp * np.sin(2*np.pi*dm_base_freq*t)*env
        dpitch = dm_pitch_amp * np.sin(0.5*np.pi*dm_pitch_freq*t)
        droll  = dm_roll_amp  * np.cos(0.5*np.pi*dm_roll_freq*t)*0.0
        dyaw   = dm_yaw_amp   * np.sin(0.5*np.pi*dm_yaw_freq*t)*0.0

    else:
        base_u = u_hover

    u1 = np.clip((base_u + (dpitch - droll) - dyaw), 0.0, 1.0)
    u2 = np.clip((base_u - (dpitch + droll) - dyaw), 0.0, 1.0)
    u3 = np.clip((base_u + (dpitch - droll) + dyaw), 0.0, 1.0)
    u4 = np.clip((base_u - (dpitch + droll) + dyaw), 0.0, 1.0)
    return [u1*MAX_RPM, u2*MAX_RPM, u3*MAX_RPM, u4*MAX_RPM]

# ----------------------------------------------------------------------
# 3. LOG + LIVE BUFFERS
# ----------------------------------------------------------------------
duration = 7.0
dt       = model.opt.timestep

csv_filename = "loadcell_experiment.csv"
headers = ["Time", "rpm1", "rpm2", "rpm3", "rpm4",
           "Thrust1", "Thrust2", "Thrust3", "Thrust4", "Thrust_total",
           "Loadcell_Fx", "Loadcell_Fy", "Loadcell_Fz"]
with open(csv_filename, 'w', newline='') as f:
    csv.writer(f).writerow(headers)

# Live buffer: sample every SAMPLE_EVERY sim steps, keep 3 s window
SAMPLE_EVERY  = 50                 # 50 * 0.0002 = 0.01 s sim
LIVE_WINDOW_S = 3.0
N_LIVE        = int(LIVE_WINDOW_S / (dt * SAMPLE_EVERY))
PLOT_EVERY_WALL = 0.10             # seconds wall time between plot redraws

buf_t      = deque(maxlen=N_LIVE)
buf_rpm    = [deque(maxlen=N_LIVE) for _ in range(4)]
buf_thrust = [deque(maxlen=N_LIVE) for _ in range(4)]
buf_lc     = [deque(maxlen=N_LIVE) for _ in range(3)]
buf_total  = deque(maxlen=N_LIVE)

# ----------------------------------------------------------------------
# 4. LIVE PLOT SETUP
# ----------------------------------------------------------------------
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle(f"Live telemetry — profile: {MOTOR_PROFILE}", fontsize=13)

colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']

rpm_lines = []
for i in range(4):
    ln, = axes[0].plot([], [], color=colors[i], lw=1.4, label=f'Motor {i+1}')
    rpm_lines.append(ln)
axes[0].set_ylabel('RPM')
# axes[0].set_ylim(0, MAX_RPM*1.05)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right', fontsize=9)

thrust_lines = []
for i in range(4):
    ln, = axes[1].plot([], [], color=colors[i], lw=1.4, label=f'Motor {i+1}')
    thrust_lines.append(ln)
total_ln, = axes[1].plot([], [], 'k--', lw=1.6, label='Total')
axes[1].set_ylabel('Thrust (N)')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right', fontsize=9)

lc_colors = ['tab:red', 'tab:green', 'tab:blue']
lc_lines = []
for i, lbl in enumerate(['Fx', 'Fy', 'Fz']):
    ln, = axes[2].plot([], [], color=lc_colors[i], lw=1.4, label=lbl)
    lc_lines.append(ln)
axes[2].set_ylabel('Load cell (N)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show(block=False)
fig.canvas.draw()
fig.canvas.flush_events()

# ----------------------------------------------------------------------
# 5. SIMULATION
# ----------------------------------------------------------------------
print(f"\nRunning '{MOTOR_PROFILE}' for {duration:.1f} s ...\n")

last_plot_wall = 0.0
t0 = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time   = data.time
    step_counter = 0

    while viewer.is_running() and data.time - start_time < duration:
        step_start = time.time()
        t = data.time - start_time

        # --- motor commands ---
        rpm_list = get_motor_rpms(t)
        omegas   = [r*2*np.pi/60.0 for r in rpm_list]
        thrusts  = [KF*w*w for w in omegas]

        # --- apply to roll body ---
        data.xfrc_applied[:] = 0.0
        roll_com = data.xipos[roll_id]
        F_tot = np.zeros(3)
        M_tot = np.zeros(3)
        for name, F_mag in zip(motor_names, thrusts):
            sid = site_ids[name]
            sm  = data.site_xmat[sid].reshape(3, 3)
            Fw  = F_mag * (-sm[:, 2])
            rw  = data.site_xpos[sid] - roll_com
            Mw  = np.cross(rw, Fw)
            F_tot += Fw
            M_tot += Mw
        data.xfrc_applied[roll_id, 0:3] = F_tot
        data.xfrc_applied[roll_id, 3:6] = M_tot

        # --- step ---
        mujoco.mj_step(model, data)

        # --- CSV every 10 steps ---
        if step_counter % 10 == 0:
            LC = data.sensordata[sensor_adr:sensor_adr+3].copy()
            row = [data.time] + rpm_list + thrusts + [sum(thrusts)] + list(LC)
            with open(csv_filename, 'a', newline='') as f:
                csv.writer(f).writerow(row)

        # --- live buffer ---
        if step_counter % SAMPLE_EVERY == 0:
            LC = data.sensordata[sensor_adr:sensor_adr+3]
            buf_t.append(t)
            for i in range(4):
                buf_rpm[i].append(rpm_list[i])
                buf_thrust[i].append(thrusts[i])
            for i in range(3):
                buf_lc[i].append(LC[i])
            buf_total.append(sum(thrusts))

        # --- live plot redraw (rate-limited on wall-clock) ---
        now = time.time()
        if now - last_plot_wall > PLOT_EVERY_WALL and len(buf_t) > 1:
            xs = list(buf_t)
            for i in range(4):
                rpm_lines[i].set_data(xs, list(buf_rpm[i]))
                thrust_lines[i].set_data(xs, list(buf_thrust[i]))
            total_ln.set_data(xs, list(buf_total))
            for i in range(3):
                lc_lines[i].set_data(xs, list(buf_lc[i]))

            for ax in axes:
                ax.set_xlim(xs[0], max(xs[-1], xs[0] + LIVE_WINDOW_S*0.1))
            for ax in axes:
                ax.relim()
                ax.autoscale_view(scalex=False)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_plot_wall = now

        step_counter += 1

        # --- thrust arrows ---
        viewer.user_scn.ngeom = 0
        for name, F_mag in zip(motor_names, thrusts):
            sid = site_ids[name]
            pos = data.site_xpos[sid]
            sm  = data.site_xmat[sid].reshape(3, 3)
            arrow_len = max(F_mag * 2.0, 0.001)
            mat_rot   = sm @ np.diag([1, -1, -1])
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.003, 0.003, arrow_len]),
                pos=pos, mat=mat_rot.flatten(),
                rgba=np.array([1.0, 0.1, 0.0, 0.9]))
            viewer.user_scn.ngeom += 1

        viewer.sync()
        until = dt - (time.time() - step_start)
        if until > 0:
            time.sleep(until)

wall = time.time() - t0
print(f"\nSimulation complete in {wall:.1f} s wall time. "
      f"Data saved to '{csv_filename}'.")

# ----------------------------------------------------------------------
# 6. STATIC PUBLICATION PLOTS
# ----------------------------------------------------------------------
plt.ioff()
plt.close('all')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

col_names = ['Time', 'rpm1', 'rpm2', 'rpm3', 'rpm4',
             'Thrust1', 'Thrust2', 'Thrust3', 'Thrust4', 'Thrust_total',
             'Loadcell_Fx', 'Loadcell_Fy', 'Loadcell_Fz']
d = np.genfromtxt(csv_filename, delimiter=',', skip_header=1, names=col_names)
t = d['Time']

# ---- Fig 1: RPMs ----
fig1, ax1 = plt.subplots(figsize=(10, 4.5))
for i in range(4):
    ax1.plot(t, d[f'rpm{i+1}'], lw=1.4, label=f'Motor {i+1}')
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('RPM')
ax1.set_title(f'Motor speeds — profile: {MOTOR_PROFILE}')
ax1.legend(loc='upper right', ncol=4)
ax1.grid(True, alpha=0.4)
fig1.tight_layout(); fig1.savefig('plot_1_rpm.png', dpi=300)

# ---- Fig 2: per-motor thrust + total ----
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for i in range(4):
    ax2a.plot(t, d[f'Thrust{i+1}'], lw=1.4, label=f'Motor {i+1}')
ax2a.set_ylabel('Per-motor thrust (N)')
ax2a.set_title('Rotor thrust')
ax2a.legend(loc='upper right', ncol=4)
ax2a.grid(True, alpha=0.4)

ax2b.plot(t, d['Thrust_total'], color='k', lw=1.8)
ax2b.set_xlabel('Time (s)'); ax2b.set_ylabel('Total thrust (N)')
ax2b.set_title('Total thrust')
ax2b.grid(True, alpha=0.4)
fig2.tight_layout(); fig2.savefig('plot_2_thrust.png', dpi=300)

# ---- Fig 3: load-cell reaction ----
fig3, ax3 = plt.subplots(figsize=(10, 4.5))
ax3.plot(t, d['Loadcell_Fx'], lw=1.4, label='$F_x$')
ax3.plot(t, d['Loadcell_Fy'], lw=1.4, label='$F_y$')
ax3.plot(t, d['Loadcell_Fz'], lw=1.4, label='$F_z$')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Reaction force (N)')
ax3.set_title('Load-cell readings (world frame)')
ax3.legend(loc='upper right', ncol=3)
ax3.grid(True, alpha=0.4)
fig3.tight_layout(); fig3.savefig('plot_3_loadcell.png', dpi=300)

# ---- Fig 4: combined publication figure ----
fig4, axes4 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for i in range(4):
    axes4[0].plot(t, d[f'rpm{i+1}'], lw=1.2, label=f'M{i+1}')
axes4[0].set_ylabel('RPM'); axes4[0].legend(loc='upper right', ncol=4, fontsize=8)
axes4[0].grid(True, alpha=0.4); axes4[0].set_title('(a) Motor RPM')

for i in range(4):
    axes4[1].plot(t, d[f'Thrust{i+1}'], lw=1.2, label=f'M{i+1}')
axes4[1].plot(t, d['Thrust_total'], 'k--', lw=1.6, label='Total')
axes4[1].set_ylabel('Thrust (N)')
axes4[1].legend(loc='upper right', ncol=5, fontsize=8)
axes4[1].grid(True, alpha=0.4); axes4[1].set_title('(b) Rotor thrust')

axes4[2].plot(t, d['Loadcell_Fx'], lw=1.2, label='$F_x$')
axes4[2].plot(t, d['Loadcell_Fy'], lw=1.2, label='$F_y$')
axes4[2].plot(t, d['Loadcell_Fz'], lw=1.2, label='$F_z$')
axes4[2].set_xlabel('Time (s)'); axes4[2].set_ylabel('Force (N)')
axes4[2].legend(loc='upper right', ncol=3, fontsize=8)
axes4[2].grid(True, alpha=0.4); axes4[2].set_title('(c) Load-cell reaction')
fig4.tight_layout(); fig4.savefig('plot_4_combined.png', dpi=300)

print("\nStatic plots saved:")
print("  plot_1_rpm.png")
print("  plot_2_thrust.png")
print("  plot_3_loadcell.png")
print("  plot_4_combined.png")

plt.show()