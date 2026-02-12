import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# Path preservation
sys.path.insert(0, r"E:\6-months-internship-projects\act2\pyroki\examples")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dual_robot_controller import DualPandaController
from dual_trajectory_planner import DualPlanner
import dual_scene_builder

def run_simulation():
    print("--- 1. BUILDING SCENE ---")
    xml_path = dual_scene_builder.create_dual_arm_scene()
    if not xml_path: return

    print("\n--- 2. INITIALIZING ---")
    controller = DualPandaController(xml_path=xml_path)
    
    # Get transforms for planner
    r1_base = controller.get_base_pose("panda1_")
    r1_rot  = controller.get_base_rotation("panda1_")
    r2_base = controller.get_base_pose("panda2_")
    r2_rot  = controller.get_base_rotation("panda2_")
    
    planner = DualPlanner(controller.kinematics_model, r1_base, r1_rot, r2_base, r2_rot)
    
    # --- TASK COORDINATES ---
    # We ask the controller for the true grasp site (Blue Line)
    if controller.grasp_site_id != -1:
        hammer_grasp_pos = controller.data.site_xpos[controller.grasp_site_id]
    else:
        hammer_grasp_pos = np.array([0.35, 0.5, 0.03])

    transfer_pos = np.array([0.0, 0.0, 0.55]) 
    place_pos    = np.array([0.0, -0.5, 0.05])
    
    print("\n--- 3. PLANNING ---")
    trajectory = planner.generate_handover_mission(hammer_grasp_pos, transfer_pos, place_pos)
    print(f"Generated {len(trajectory)} steps.")
    
    print("\n--- 4. SIMULATING ---")
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # Warmup
        for _ in range(50): mujoco.mj_step(controller.model, controller.data)
        
        traj_idx = 0
        step_counter = 0
        sim_speed = 1
        
        # Initial Gripper State: OPEN
        r1_grip = 0.04
        r2_grip = 0.04
        last_action = ""

        while viewer.is_running():
            if traj_idx < len(trajectory):
                step = trajectory[traj_idx]
                action = step["action"]
                
                # --- STATE MACHINE ---
                # We use explicit checks to ensure r2_grip is set correctly
                
                # 1. R2 APPROACH & DESCEND -> OPEN
                if action in ["r2_move_open", "r2_descend"]:
                    r2_grip = 0.04 
                    if controller.cube_attached_to == "panda2_": controller.detach()
                    
                # 2. R2 GRASP (THE WAIT PHASE) -> CLOSE
                elif action == "r2_close":
                    r2_grip = -0.01  # Force Close
                    if not controller.cube_attached_to: controller.attach("panda2_")
                    
                # 3. R2 LIFT & MOVE -> KEEP CLOSED
                elif action in ["r2_lift", "staging_move", "r1_approach"]:
                    r2_grip = -0.01 
                    r1_grip = 0.04 

                # 4. HANDOVER -> BOTH CLOSED
                elif action == "r1_grasp":
                    r1_grip = -0.01 
                    r2_grip = -0.01 
                    
                # 5. RELEASE -> R2 OPEN, R1 CLOSED
                elif action == "r2_release":
                    r2_grip = 0.04 
                    r1_grip = -0.01 
                    if controller.cube_attached_to == "panda2_":
                        controller.detach()
                        controller.attach("panda1_")
                
                # 6. RETREAT -> R2 OPEN, R1 CLOSED
                elif action in ["r2_retreat", "r1_place_move"]:
                    r2_grip = 0.04
                    r1_grip = -0.01

                # Send commands
                controller.set_commands(step["r1"], r1_grip, step["r2"], r2_grip)
                
                # VISUAL DEBUGGING
                if action != last_action:
                    print(f"Action: {action} | R2 Grip Cmd: {r2_grip}")
                    last_action = action
                
                step_counter += 1
                if step_counter >= sim_speed:
                    traj_idx += 1
                    step_counter = 0
            else:
                if last_action != "DONE":
                    print("--- MISSION COMPLETE ---")
                    last_action = "DONE"

            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    run_simulation()