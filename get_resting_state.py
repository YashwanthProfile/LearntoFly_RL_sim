import mujoco
import mujoco.viewer
import time

# Load the XML model
model_path = '5DOF.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Set the initial state using the saved "home" keyframe
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    print("Successfully initialized simulation to the 'home' keyframe.\n")
else:
    print("Warning: 'home' keyframe not found in XML.\n")

# Simulation parameters
simulation_time = 10.0  # Total time to run in seconds
dt = model.opt.timestep # Timestep (e.g. 0.0002)

print(f"Running simulation for {simulation_time} seconds to let it settle...\n")

# Launch the visualizer window and run the physics simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_print_time = 0.0
    
    # Loop over the duration of the simulation while the window is open
    while viewer.is_running() and data.time < simulation_time:
        step_start = time.time()
        
        # 1. Step the Physics (no ctrl forces here, it just settles)
        mujoco.mj_step(model, data)
        
        # 2. Print an update every 0.5 seconds of simulation time
        if data.time - last_print_time >= 0.5:
            print(f"--- Simulation Time: {data.time:.1f}s ---")
            def format_val(val):
                if isinstance(val, (list, tuple, type(data.qpos))):
                    return "[" + ", ".join([f"{v:.4f}" for v in val]) + "]"
                return f"{val:.4f}"

            for i in range(model.njnt):
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                qpos_adr = model.jnt_qposadr[i]
                dof_adr = model.jnt_dofadr[i]
                jnt_type = model.jnt_type[i]
                
                if jnt_type == 0: 
                    pos = data.qpos[qpos_adr : qpos_adr+7]
                    vel = data.qvel[dof_adr : dof_adr+6]
                elif jnt_type == 1: 
                    pos = data.qpos[qpos_adr : qpos_adr+4]
                    vel = data.qvel[dof_adr : dof_adr+3]
                else: 
                    pos = data.qpos[qpos_adr]
                    vel = data.qvel[dof_adr]
                    
                print(f"  {joint_name} - Pos: {format_val(pos)} | Vel: {format_val(vel)}")
            
            body_name = "Crazyflie_Combained-Body-v2"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                body_pos = data.xpos[body_id]
                body_vel = data.cvel[body_id]
                print(f"\n  {body_name} State:")
                print(f"    Position   : {format_val(body_pos)}")
                print(f"    Linear Vel : {format_val(body_vel[3:6])}")
                print(f"    Angular Vel: {format_val(body_vel[0:3])}")

            print("-----------------------------------")
            last_print_time = data.time
            
        # 3. Synchronize the viewer state with the physics state
        viewer.sync()

        # 4. Optional: Match the simulation speed to real-world time for correct visualization
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\n==== FINAL STATE (XML Keyframe Snippet) ====\n")
print("<keyframe>")
qpos_str = " ".join([f"{x:.6f}" for x in data.qpos])
print(f'    <key name="resting_state" qpos="{qpos_str}"/>')
print("</keyframe>")
