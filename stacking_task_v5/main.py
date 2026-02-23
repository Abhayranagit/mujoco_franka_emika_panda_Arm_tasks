import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import h5py

# Path preservation
sys.path.insert(0, r"E:\6-months-internship-projects\act2\pyroki\examples")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_robot_controller import SinglePandaController
from single_trajectory_planner import SinglePlanner
import single_scene_builder

class DataRecorder:
    def __init__(self, model):
        self.qpos = []
        self.actions = []
        self.images = []
        
        # Setup MuJoCo Offscreen Renderer (Fast, headless rendering)
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        
        # Setup Camera to match the visualizer view
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.distance = 1.5
        self.cam.azimuth = 90
        self.cam.elevation = -40
        self.cam.lookat[:] = [0.5, 0.0, 0.0]

    def record_step(self, controller, target_joints, target_gripper):
        # 1. Capture Image
        self.renderer.update_scene(controller.data, camera=self.cam)
        rgb_array = self.renderer.render()
        self.images.append(rgb_array.copy()) # Copy to ensure array isn't overwritten

        # 2. Capture Synchronized Joint Positions
        current_qpos = []
        # The controller mapped the first 7 actuators to the arm
        for i in range(7): 
            act_id = controller.actuator_ids[i]
            # Map actuator ID -> joint ID -> qpos address
            jnt_id = controller.model.actuator_trnid[act_id, 0]
            qpos_adr = controller.model.jnt_qposadr[jnt_id]
            
            # Extract the actual physical angle
            current_qpos.append(controller.data.qpos[qpos_adr])
            
        self.qpos.append(current_qpos)

        # 3. Capture Action (Target Commands)
        action = list(target_joints) + [target_gripper]
        self.actions.append(action)

    def save_to_disk(self, filename="episode_0.hdf5"):
        print(f"\nSaving {len(self.qpos)} synchronized steps to {filename}...")
        with h5py.File(filename, 'w') as f:
            obs = f.create_group('observations')
            
            # Save joints
            obs.create_dataset('qpos', data=np.array(self.qpos, dtype=np.float32))
            
            # Save images compressed
            img_grp = obs.create_group('images')
            img_grp.create_dataset('cam_high', data=np.array(self.images, dtype=np.uint8), compression="gzip")
            
            # Save actions
            f.create_dataset('action', data=np.array(self.actions, dtype=np.float32))
            
            f.attrs['sim'] = True
            f.attrs['num_samples'] = len(self.qpos)
            
        print("Dataset saved successfully!")


def run_simulation():
    print("--- 1. BUILDING SCENE ---")
    xml_path = single_scene_builder.create_single_arm_scene()
    if not xml_path: return

    print("\n--- 2. INITIALIZING ---")
    controller = SinglePandaController(xml_path=xml_path)
    
    print("Settling physics (Warmup)...")
    for _ in range(200): 
        mujoco.mj_step(controller.model, controller.data)
        
    print("Reading settled cube positions...")
    all_cubes = []
    possible_cubes = ["cube_1", "cube_2", "cube_3", "cube_4", "cube_5"] 
    
    for name in possible_cubes:
        pos = controller.get_object_pos(name)
        if pos is not None:
            print(f"Found {name} at {pos}")
            all_cubes.append(pos)
        else:
            print(f"Skipping {name} (Not found in scene)")

    # 5. PLANNER
    base_pos = controller.get_base_pose()
    base_rot = controller.get_base_rotation()
    planner = SinglePlanner(controller.kinematics_model, base_pos, base_rot)
    
    target_xy = np.array([0.6, 0.0]) 

    print("\n--- PLANNING FULL STACK ---")
    trajectory = planner.generate_full_stack_mission(target_xy, all_cubes)
    print(f"Generated {len(trajectory)} high-frequency steps.")
    
    # 6. SIMULATION & RECORDING
    print("\n--- 6. SIMULATING & RECORDING ---")
    
    # Init Recorder
    recorder = DataRecorder(controller.model)
    
    # Config for syncing: 
    # MuJoCo default physics step is 0.002s (500Hz). 
    # We record every 20 steps to get a 25Hz recording frequency.
    RECORD_INTERVAL = 20 
    physics_step_counter = 0
    
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        traj_idx = 0
        gripper_state = 0.04 
        last_action = ""

        while viewer.is_running():
            if traj_idx < len(trajectory):
                step = trajectory[traj_idx]
                action = step["action"]
                
                # Logic to determine gripper state based on action keyword
                if any(k in action for k in ["grasp", "lift", "transfer", "stack"]):
                    gripper_state = -0.01 
                else: 
                    gripper_state = 0.04  
                    if "release" in action: controller.detach() 

                target_joints = step["joints"]
                
                # Send command to controller
                controller.set_commands(target_joints, gripper_state)
                
                # --- SYNCHRONIZED RECORDING GATE ---
                if physics_step_counter % RECORD_INTERVAL == 0:
                    recorder.record_step(controller, target_joints, gripper_state)
                
                if action != last_action:
                    print(f"Executing: {action}")
                    last_action = action
                
                traj_idx += 1
            else:
                if last_action != "DONE":
                    print("--- STACKING COMPLETE ---")
                    
                    # Save the data
                    timestamp = int(time.time())
                    recorder.save_to_disk(f"mujoco_episode_{timestamp}.hdf5")
                    
                    last_action = "DONE"
                    break # Safely exit the loop

            # Step physics and update viewer
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            
            # Increment physics counter for sync tracking
            physics_step_counter += 1
            
            # Optional: Add small sleep to slow down visualizer to real-time speed
            time.sleep(0.002) 

if __name__ == "__main__":
    run_simulation()