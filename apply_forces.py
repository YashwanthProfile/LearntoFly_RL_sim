import mujoco
import mujoco.viewer
import time
import numpy as np

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path('5DOF.xml')
    data = mujoco.MjData(model)

    # Start from home state
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("Reset to 'home' keyframe.")
    else:
        print("Warning: 'home' keyframe not found. Using default initial state.")

    # Duration for gradually increasing the forces
    duration = 10.0  # seconds

    # Thrust bounds in Newtons (gear Z-component = 0.27)
    min_thrust_N = 0.0500
    max_thrust_N = 0.0545
    gear_z = 0.27

    # Convert thrust bounds to ctrl values
    min_ctrl = min_thrust_N / gear_z  # ≈ 0.1852
    max_ctrl = max_thrust_N / gear_z  # ≈ 0.2963

    print(f"Starting simulation for {duration}s | thrust range: {min_thrust_N:.4f}–{max_thrust_N:.4f} N  "
          f"(ctrl: {min_ctrl:.4f}–{max_ctrl:.4f})")

    # Pre-fetch site and actuator IDs for force visualization
    motor_names = ["motor1", "motor2", "motor3", "motor4"]
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in motor_names}
    site_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in motor_names}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable built-in actuator visualization if desired
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        
        start_time = data.time
        
        while viewer.is_running() and data.time - start_time < duration:
            step_start = time.time()
            
            # Calculate progress over simulation duration
            progress = (data.time - start_time) / duration
            
            # Ramp ctrl from min_ctrl to max_ctrl over the duration
            current_ctrl = np.clip(min_ctrl + progress * (max_ctrl - min_ctrl), min_ctrl, max_ctrl)
            
            # Define which motors to fire (change this to test tilting)
            active_motors = [ "motor1", "motor2", "motor3", "motor4"]
            
            # Apply control only to active motors; zero out the others
            for name in motor_names:
                if name in active_motors:
                    data.ctrl[actuator_ids[name]] = current_ctrl
                else:
                    data.ctrl[actuator_ids[name]] = 0.0
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Print motor outputs at ~10 Hz to avoid flooding the console
            if int(data.time * 10) != int((data.time - model.opt.timestep) * 10):
                print(f"\nt={data.time:.2f}s | Motor outputs:")
                for name in motor_names:
                    ctrl = data.ctrl[actuator_ids[name]]
                    thrust = ctrl * 0.27
                    active = "✓" if name in active_motors else "✗"
                    print(f"  [{active}] {name}: ctrl={ctrl:.4f}  thrust={thrust:.4f} N")
            
            # --- Visualize forces with 3D arrows (only for active motors) ---
            viewer.user_scn.ngeom = 0  # Reset user geometries for this frame
            scale_factor = 0.2         # Scale factor for arrow length based on force magnitude

            for name in active_motors:
                site_id = site_ids[name]
                act_id = actuator_ids[name]
                
                # Site 3D position and orientation matrix
                pos = data.site_xpos[site_id]
                
                # Since gear is [0, 0, -0.27], force is applied along local -Z (upward).
                # Rotate geom matrix 180 deg around X-axis so arrow points in direction of physical force.
                mat_rot = data.site_xmat[site_id].reshape(3, 3) @ np.diag([1, -1, -1])
                mat = mat_rot.flatten()
                
                # Thrust force magnitude
                thrust = data.ctrl[act_id] * 0.27
                arrow_len = max(thrust * scale_factor, 0.001)
                
                # Initialize force arrow geometry
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([0.003, 0.003, arrow_len]),  # [radius_x, radius_y, length]
                    pos=pos,
                    mat=mat,
                    rgba=np.array([1.0, 0.1, 0.0, 0.9])         # RGBA color (bright red/orange arrow)
                )
                viewer.user_scn.ngeom += 1
            
            # Update the viewer
            viewer.sync()
            
            # Try to run at roughly real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    print("Simulation complete.")

if __name__ == "__main__":
    main()
