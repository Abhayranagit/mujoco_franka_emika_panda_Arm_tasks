import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# Path preservation
sys.path.insert(0, r"E:\6-months-internship-projects\act2\pyroki\examples")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_robot_controller import SinglePandaController
from single_trajectory_planner import SinglePlanner
import single_scene_builder

def run_simulation():
    print("--- 1. BUILDING SCENE ---")
    # Make sure you removed cube_2 from single_scene_builder.py if you want it gone!
    xml_path = single_scene_builder.create_single_arm_scene()
    if not xml_path: return

    print("\n--- 2. INITIALIZING ---")
    controller = SinglePandaController(xml_path=xml_path)
    
    print("Settling physics (Warmup)...")
    for _ in range(200): 
        mujoco.mj_step(controller.model, controller.data)
        
    print("Reading settled cube positions...")
    
    # !!! KEY FIX: DYNAMIC LIST BUILDING !!!
    all_cubes = []
    
    # We check each potential cube. If get_object_pos returns None, we skip it.
    possible_cubes = ["cube_1", "cube_2", "cube_3", "cube_4", "cube_5"]  # Added cube_5 as well
    
    for name in possible_cubes:
        pos = controller.get_object_pos(name)
        if pos is not None:
            print(f"Found {name} at {pos}")
            all_cubes.append(pos)
        else:
            print(f"Skipping {name} (Not found in scene)")

    # If you removed cube_2, the list will now naturally have only 3 items.
    
    base_pos = controller.get_base_pose()
    base_rot = controller.get_base_rotation()
    planner = SinglePlanner(controller.kinematics_model, base_pos, base_rot)
    
    target_xy = np.array([0.6, 0.0]) 

    print("\n--- PLANNING FULL STACK ---")
    trajectory = planner.generate_full_stack_mission(target_xy, all_cubes)
    print(f"Generated {len(trajectory)} steps.")
    
    print("\n--- 6. SIMULATING ---")
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        traj_idx = 0
        step_counter = 0
        sim_speed = 1 
        gripper_state = 0.04 
        last_action = ""

        while viewer.is_running():
            if traj_idx < len(trajectory):
                step = trajectory[traj_idx]
                action = step["action"]
                
                if any(k in action for k in ["grasp", "lift", "transfer", "stack"]):
                    gripper_state = -0.01 
                else: 
                    gripper_state = 0.04  
                    if "release" in action: controller.detach() 

                controller.set_commands(step["joints"], gripper_state)
                
                if action != last_action:
                    print(f"Executing: {action}")
                    last_action = action
                
                step_counter += 1
                if step_counter >= sim_speed:
                    traj_idx += 1
                    step_counter = 0
            else:
                if last_action != "DONE":
                    print("--- STACKING COMPLETE ---")
                    last_action = "DONE"

            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    run_simulation()