import numpy as np
import sys
sys.path.insert(0, r"mujoco_franka_emika_panda_Arm_tasks\\pyroki\\examples")
import pyroki_snippets as pks

class SinglePlanner:
    def __init__(self, kin_model, base_pos, base_rot):
        self.robot = kin_model
        self.base_pos = np.array(base_pos)
        self.base_rot = np.array(base_rot)
        self.link_name = "panda_hand"
        
    def solve_ik(self, world_target, orient_mode="down"):
        rel_pos = world_target - self.base_pos
        local_pos = self.base_rot.T @ rel_pos 
        target_quat = np.array([0, 1, 0, 0]) 

        try:
            return pks.solve_ik(self.robot, self.link_name, target_quat, local_pos)
        except:
            return pks.solve_ik(robot=self.robot, target_link_name=self.link_name, target_wxyz=target_quat, target_position=local_pos)

    def interpolate_segment(self, start_pos, end_pos, steps, action_name, plan_list):
        for t in np.linspace(0, 1, int(steps)):
            curr_pos = start_pos * (1-t) + end_pos * t
            q = self.solve_ik(curr_pos, "down")
            plan_list.append({"joints": q, "action": action_name})

    def generate_full_stack_mission(self, target_xy, all_cubes_list):
        full_plan = []
        
        # --- CONFIG ---
        HOVER_HEIGHT = 0.25
        HOME_POS = self.base_pos + np.array([0.3, 0.0, 0.5]) 
        
        # Hand Length Offset (Wrist must be higher than object)
        TCP_OFFSET = 0.100
        
        CUBE_HEIGHT = 0.05
        current_stack_z = 0.028 
        
        current_pos = HOME_POS
        self.interpolate_segment(HOME_POS, HOME_POS, 10, "home", full_plan)

        for i, src_pos in enumerate(all_cubes_list):
            cube_num = i + 1
            print(f"--- PLANNING CUBE {cube_num} ---")
            
            # Points
            src_hover = np.array([src_pos[0]+0.05, src_pos[1], HOVER_HEIGHT])
            src_grasp = np.array([src_pos[0], src_pos[1], src_pos[2] + TCP_OFFSET])
            
            tgt_hover = np.array([target_xy[0], target_xy[1], HOVER_HEIGHT+0.05])
            tgt_place = np.array([target_xy[0], target_xy[1], current_stack_z + TCP_OFFSET]) 

            # --- 1. SLOW PICK SEQUENCE ---
            self.interpolate_segment(current_pos, src_hover, 150, f"move_to_c{cube_num}", full_plan)
            
            # SLOW DESCEND: Increased steps 60 -> 120
            self.interpolate_segment(src_hover, src_grasp, 150, f"descend_c{cube_num}", full_plan)
            
            # STABILIZE: Wait 80 steps (approx 0.8s) before closing gripper
            last_q = full_plan[-1]['joints']
            for _ in range(130): 
                full_plan.append({"joints": last_q, "action": f"descend_c{cube_num}"})
            
            # GRASP: Wait 100 steps (approx 1s) for fingers to fully close
            for _ in range(120): 
                full_plan.append({"joints": last_q, "action": "grasp"})
            
            # Lift
            self.interpolate_segment(src_grasp, src_hover, 150, f"lift_c{cube_num}", full_plan)

            # --- 2. SLOW PLACE SEQUENCE ---
            self.interpolate_segment(src_hover, tgt_hover, 100, f"transfer_c{cube_num}", full_plan)
            
            # SLOW STACK: Increased steps 80 -> 120
            self.interpolate_segment(tgt_hover, tgt_place, 120, f"stack_c{cube_num}", full_plan)
            
            # STABILIZE: Wait 50 steps before opening
            last_q = full_plan[-1]['joints']
            for _ in range(50): 
                full_plan.append({"joints": last_q, "action": f"stack_c{cube_num}"})
                
            # RELEASE: Wait 80 steps for fingers to fully open
            for _ in range(90): 
                full_plan.append({"joints": last_q, "action": "release"})
            
            # Retreat
            self.interpolate_segment(tgt_place, tgt_hover, 70, "retreat", full_plan)
            
            current_pos = tgt_hover
            current_stack_z += CUBE_HEIGHT 

        self.interpolate_segment(current_pos, HOME_POS, 80, "return_home", full_plan)

        return full_plan