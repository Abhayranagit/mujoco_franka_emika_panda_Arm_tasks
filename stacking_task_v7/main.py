import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import h5py
import cv2
import google.generativeai as genai
import PIL.Image

# Path preservation
sys.path.insert(0, r"E:\6-months-internship-projects\act2\pyroki\examples")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_robot_controller import SinglePandaController
from single_trajectory_planner import SinglePlanner
import single_scene_builder

# --- 1. REAL VLM AGENT (Google Gemini) ---
# --- 1. REAL VLM AGENT (Google Gemini) ---
class VLMAgent:
    def __init__(self, api_key="API_key_here"):
        # Configure the API key
        genai.configure(api_key=api_key)
        # Load the current Gemini 2.5 Flash model
        self.model = genai.GenerativeModel('gemini-2.5-flash') 
        
        # PRAGMABOT STM: Remember what we just did so we don't repeat it!
        self.last_command = "None"

    def get_next_action(self, img_pixels, object_states):
        print("\n[VLM API] Sending image to Gemini and thinking...")
        
        # Convert the raw MuJoCo pixel array into a PIL Image format for Gemini
        img = PIL.Image.fromarray(img_pixels)
        
        # --- THE UPDATED PROMPT ---
        prompt = f"""
        You are controlling a Franka Emika Panda arm. 
        Current object states (XYZ coordinates): {object_states}
        Target Stacking Zone: [0.6, 0.0]
        
        PRAGMABOT Short-Term Memory:
        The last command you successfully executed was: {self.last_command}
        
        TASK INSTRUCTION:
        Stack the cubes at the Target Stacking Zone in this exact specific order: 
        First cube_4, then cube_3, then cube_1. 
        
        CRITICAL RULES: 
        1. Confirm first the sequence is being followed. next cube should not be picked if the previous cube in the sequence is not already stacked at the target zone.
        2. If a cube's X and Y coordinates are already at or very close to [0.6, 0.0], it is ALREADY STACKED. 
        3. Do not pick up a cube that is already stacked. Move on to the next cube in the sequence.
        4. If your last command was a 'pick', your next command MUST be a 'place' for that same cube.
        
        Look at the image, the object states, and your last command to determine what step you are on.
        You can only output ONE of these commands: 
        pick(cube_name), place(cube_name), or done.
        Output only the exact command and nothing else. Do not include markdown formatting.
        """
        
        try:
            # Call the free API
            response = self.model.generate_content([prompt, img])
            command = response.text.strip()
            
            # Update the STM for the next loop
            self.last_command = command 
            
            return command
        except Exception as e:
            print(f"[VLM API Error]: {e}")
            return "done"


# --- 2. DATA RECORDER ---
class DataRecorder:
    def __init__(self, model):
        self.qpos = []
        self.actions = []
        self.images = []
        
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.distance = 1.5
        self.cam.azimuth = 90
        self.cam.elevation = -40
        self.cam.lookat[:] = [0.5, 0.0, 0.0]

    def record_step(self, controller, target_joints, target_gripper):
        # 1. Capture Image (UPDATED)
        self.renderer.update_scene(controller.data, camera=self.cam)
        self.images.append(self.renderer.render())

        # 2. Capture Joint Positions
        current_qpos = []
        for i in range(7): 
            act_id = controller.actuator_ids[i]
            jnt_id = controller.model.actuator_trnid[act_id, 0]
            qpos_adr = controller.model.jnt_qposadr[jnt_id]
            current_qpos.append(controller.data.qpos[qpos_adr])
        self.qpos.append(current_qpos)

        # 3. Capture Action (Target Commands)
        action = list(target_joints[:7]) + [target_gripper]
        self.actions.append(action)

    def save_to_disk(self, filename="episode_0.hdf5"):
        print(f"\nSaving {len(self.qpos)} synchronized steps to {filename}...")
        with h5py.File(filename, 'w') as f:
            obs = f.create_group('observations')
            obs.create_dataset('qpos', data=np.array(self.qpos, dtype=np.float32))
            img_grp = obs.create_group('images')
            img_grp.create_dataset('cam_high', data=np.array(self.images, dtype=np.uint8), compression="gzip")
            f.create_dataset('action', data=np.array(self.actions, dtype=np.float32))
            f.attrs['sim'] = True
            f.attrs['num_samples'] = len(self.qpos)
        print("Dataset saved successfully!")


# --- 3. MAIN SIMULATION LOOP ---
def run_simulation():
    print("--- 1. BUILDING SCENE ---")
    xml_path = single_scene_builder.create_single_arm_scene()
    if not xml_path: return

    print("\n--- 2. INITIALIZING ---")
    controller = SinglePandaController(xml_path=xml_path)
    
    print("Settling physics (Warmup)...")
    for _ in range(200): 
        mujoco.mj_step(controller.model, controller.data)

    # Init Planner & VLM
    base_pos = controller.get_base_pose()
    base_rot = controller.get_base_rotation()
    planner = SinglePlanner(controller.kinematics_model, base_pos, base_rot)
    vlm = VLMAgent()
    recorder = DataRecorder(controller.model)

    # Task State tracking
    target_xy = np.array([0.6, 0.0]) 
    current_stack_z = 0.028 # Start height
    
    current_sub_plan, current_robot_pos = planner.generate_home_plan()
    
    print("\n--- 6. SIMULATING (VLM CLOSED-LOOP) ---")
    RECORD_INTERVAL = 20 
    physics_step_counter = 0
    
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        traj_idx = 0
        gripper_state = 0.04 
        last_action = ""

        while viewer.is_running():
            
            # --- VLM OBSERVATION & PLANNING PHASE ---
            if traj_idx >= len(current_sub_plan):
                # 1. Render an image for the VLM to "see" (UPDATED)
                recorder.renderer.update_scene(controller.data, camera=recorder.cam)
                img_pixels = recorder.renderer.render()
                
                # 2. Gather state data for the VLM prompt
                states = {}
                for name in ["cube_1", "cube_2", "cube_3", "cube_4"]:
                    pos = controller.get_object_pos(name)
                    if pos is not None: states[name] = pos
                
                # 3. Ask VLM what to do
                vlm_command = vlm.get_next_action(img_pixels, states)
                print(f"[VLM Outputs Command]: {vlm_command}")
                
                # 4. Parse Command & Generate Physics Sub-plan
                if "pick" in vlm_command:
                    cube_name = vlm_command.split("(")[1].split(")")[0]
                    target_pos = controller.get_object_pos(cube_name)
                    current_sub_plan, current_robot_pos = planner.generate_pick_plan(
                        current_robot_pos, target_pos, cube_name
                    )
                    traj_idx = 0 
                    
                elif "place" in vlm_command:
                    cube_name = vlm_command.split("(")[1].split(")")[0]
                    current_sub_plan, current_robot_pos = planner.generate_place_plan(
                        current_robot_pos, target_xy, current_stack_z, cube_name
                    )
                    current_stack_z += planner.CUBE_HEIGHT # Increment stack height for next time
                    traj_idx = 0
                    
                elif "done" in vlm_command:
                    print("\n--- VLM DECLARED TASK COMPLETE ---")
                    timestamp = int(time.time())
                    recorder.save_to_disk(f"mujoco_episode_{timestamp}.hdf5")
                    break 

            # --- EXECUTION PHASE ---
            if traj_idx < len(current_sub_plan):
                step = current_sub_plan[traj_idx]
                action = step["action"]
                
                if any(k in action for k in ["grasp", "lift", "transfer", "stack"]):
                    gripper_state = -0.01 
                else: 
                    gripper_state = 0.04  
                    if "release" in action: controller.detach() 

                target_joints = step["joints"]
                controller.set_commands(target_joints, gripper_state)
                
                # Sync Recording
                if physics_step_counter % RECORD_INTERVAL == 0:
                    recorder.record_step(controller, target_joints, gripper_state)
                
                if action != last_action:
                    print(f"Executing: {action}")
                    last_action = action
                
                traj_idx += 1

            # Step physics
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            physics_step_counter += 1
            time.sleep(0.002) 

if __name__ == "__main__":
    run_simulation()