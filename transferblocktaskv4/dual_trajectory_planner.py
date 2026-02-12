import numpy as np
import sys
# Preserving path
sys.path.insert(0, r"E:\\6-months-internship-projects\\act2\\pyroki\\examples")
import pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R

class DualPlanner:
    def __init__(self, kin_model, r1_pos, r1_rot, r2_pos, r2_rot):
        self.robot = kin_model
        self.r1_pos = np.array(r1_pos)
        self.r2_pos = np.array(r2_pos)
        self.r1_rot = np.array(r1_rot)
        self.r2_rot = np.array(r2_rot)
        self.link_name = "panda_hand"
        self.default_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
    def solve_ik(self, world_target, robot_pos, robot_rot, orient_mode="down"):
        rel_pos = world_target - robot_pos
        local_pos = robot_rot.T @ rel_pos 
        
        # --- ORIENTATION STRATEGY ---
        if orient_mode == "down":
            target_quat = np.array([0, 1, 0, 0]) # Straight Down
            
        elif orient_mode == "forward":
            # --- NEW: FRONT CLAWS (Horizontal Approach) ---
            # Start with DOWN (180 deg rotation on X)
            # Rotate -90 degrees on X to bring it up to Front
            base_r = R.from_quat([1, 0, 0, 0]) # Scipy uses [x, y, z, w], so this is w=0, x=1
            correction = R.from_euler('y', 85, degrees=True)
            
            new_r = base_r * correction
            qx, qy, qz, qw = new_r.as_quat()
            target_quat = np.array([qw, qx, qy, qz])

        elif orient_mode == "r2_present_horizontal":
            # R2 Correction (+15 deg Left)
            base_quat = [0, 0.707, 0.707, 0] 
            correction = R.from_euler('x', 15, degrees=True) 
            base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]) 
            new_rot = base_rot * correction
            qx, qy, qz, qw = new_rot.as_quat()
            target_quat = np.array([qw, qx, qy, qz])
            
        elif orient_mode == "r1_receive_vertical":
            # ORIGINAL FLAT ORIENTATION (Preserved for Home Pose)
            target_quat = np.array([0.5, 0.5, 0.5, 0.5]) 
            
        else:
            target_quat = np.array([0, 1, 0, 0])

        try:
            return pks.solve_ik(self.robot, self.link_name, target_quat, local_pos)
        except:
            return pks.solve_ik(robot=self.robot, target_link_name=self.link_name, target_wxyz=target_quat, target_position=local_pos)

    def interpolate_segment(self, start_pos, end_pos, steps, r1_args, r2_args, action_name, plan_list):
        """ Cartesian Interpolation (Good for Phase 1 & 2 straight lines) """
        for t in np.linspace(0, 1, int(steps)):
            curr_r1_pos = r1_args['start'] * (1-t) + r1_args['end'] * t
            curr_r2_pos = r2_args['start'] * (1-t) + r2_args['end'] * t
            
            q1 = self.solve_ik(curr_r1_pos, self.r1_pos, self.r1_rot, r1_args['orient'])
            q2 = self.solve_ik(curr_r2_pos, self.r2_pos, self.r2_rot, r2_args['orient'])
            
            # --- MANUAL OVERRIDE FOR R2 WRIST (JOINT 7) ---
            # Index 6 corresponds to the 7th actuator (0-6)
            if action_name in ["r2_descend", "r2_move_open"]:
                # Add rotation in RADIANS. 
                # 0.785 = 45 degrees
                # 1.57 = 90 degrees
                # -0.785 = -45 degrees
                q2[6] += 1.57  # <--- CHANGE THIS VALUE to rotate the wrist manually
                
            plan_list.append({"r1": q1, "r2": q2, "action": action_name})

    def interpolate_joint_segment(self, start_pos, end_pos, steps, r1_args, r2_args, action_name, plan_list):
        """ Joint Space Interpolation """
        q1_start = self.solve_ik(r1_args['start'], self.r1_pos, self.r1_rot, r1_args['orient'])
        q1_end   = self.solve_ik(r1_args['end'],   self.r1_pos, self.r1_rot, r1_args['orient'])
        
        q2_start = self.solve_ik(r2_args['start'], self.r2_pos, self.r2_rot, r2_args['orient'])
        q2_end   = self.solve_ik(r2_args['end'],   self.r2_pos, self.r2_rot, r2_args['orient'])

        for t in np.linspace(0, 1, int(steps)):
            q1_curr = q1_start * (1-t) + q1_end * t
            q2_curr = q2_start * (1-t) + q2_end * t
            
            plan_list.append({"r1": q1_curr, "r2": q2_curr, "action": action_name})

    def generate_handover_mission(self, hammer_handle_pos, transfer_center_ignored, place_target):
        full_plan = []
        
        # --- GLOBAL SETTINGS ---
        TRANSFER_HEIGHT = 0.55
        
        R2_TARGET = np.array([0.0, 0.0, 0.65])    
        R1_TARGET = np.array([0.0485, -0.04, TRANSFER_HEIGHT]) 
        
        r1_home = self.r1_pos + np.array([0.04, 0.3, 0.5]) 
        r2_home = self.r2_pos + np.array([0, -0.3, 0.5]) 
        
        # --- PHASE 1: PICK ---
        r2_align_high = hammer_handle_pos + np.array([0, 0, 0.30])    
        r2_pre_grasp  = hammer_handle_pos + np.array([0, 0, 0.07])   
        r2_grasp      = hammer_handle_pos + np.array([0, 0, 0.005]) 
        r2_lift_high  = hammer_handle_pos + np.array([0, 0, 0.50])    
        
        r2_final      = R2_TARGET
        r1_final      = np.array([R1_TARGET[0], 0.00, 0.5])

        print("--- PHASE 1: R2 PICK ---")
        
        self.interpolate_segment(r2_home, r2_align_high, 150, 
            {'start': r1_home, 'end': r1_home, 'orient': 'r1_receive_vertical'}, 
            {'start': r2_home, 'end': r2_align_high, 'orient': 'down_perpendicular'}, 
            "r2_move_open", full_plan)

        self.interpolate_segment(r2_align_high, r2_grasp, 200, 
            {'start': r1_home, 'end': r1_home, 'orient': 'r1_receive_vertical'}, 
            {'start': r2_align_high, 'end': r2_grasp, 'orient': 'down_perpendicular'}, 
            "r2_descend", full_plan)
        
        last = full_plan[-1]
        for _ in range(350): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r2_close"}) 

        self.interpolate_segment(r2_grasp, r2_lift_high, 200, 
            {'start': r1_home, 'end': r1_home, 'orient': 'r1_receive_vertical'}, 
            {'start': r2_grasp, 'end': r2_lift_high, 'orient': 'down_perpendicular'}, 
            "r2_lift", full_plan)

        # ==========================================================
        # PHASE 2: STAGING
        # ==========================================================
        print("--- PHASE 2: STAGING ---")
        
        self.interpolate_segment(r2_lift_high, r2_final, 150, 
            {'start': r1_home, 'end': r1_home, 'orient': 'r1_receive_vertical'}, 
            {'start': r2_lift_high, 'end': r2_final, 'orient': 'r2_present_horizontal'}, 
            "staging_move", full_plan)

        # ==========================================================
        # PHASE 3: R1 APPROACH (FRONT CLAWS / FORWARD)
        # ==========================================================
        print("--- PHASE 3: R1 APPROACH (FRONT CLAWS) ---")
        
        r1_align_safe = np.array([R1_TARGET[0], -0.35, TRANSFER_HEIGHT])
        
        r1_approach_1 = np.array([R1_TARGET[0], -0.30, TRANSFER_HEIGHT]) 
        r1_approach_2 = np.array([R1_TARGET[0], -0.27, TRANSFER_HEIGHT])
        r1_approach_3 = np.array([R1_TARGET[0], -0.25, TRANSFER_HEIGHT])
        r1_approach_4 = np.array([R1_TARGET[0], -0.23, TRANSFER_HEIGHT])
        r1_approach_5 = np.array([R1_TARGET[0], -0.19, TRANSFER_HEIGHT])
        r1_approach_6 = np.array([R1_TARGET[0], -0.15, TRANSFER_HEIGHT])
        r1_approach_7 = np.array([R1_TARGET[0], -0.12, TRANSFER_HEIGHT])
        r1_approach_8 = np.array([R1_TARGET[0], -0.10, TRANSFER_HEIGHT])
        r1_approach_9 = np.array([R1_TARGET[0], -0.09, TRANSFER_HEIGHT])
        r1_approach_10 = np.array([R1_TARGET[0], -0.05, TRANSFER_HEIGHT])
        r1_approach_11 = np.array([R1_TARGET[0], -0.02, TRANSFER_HEIGHT])
        
        # 1. Align Safe -> USING "forward" (Front Claws)
        self.interpolate_joint_segment(
            start_pos=r1_home, end_pos=r1_align_safe, steps=120,
            r1_args={'start': r1_home, 'end': r1_align_safe, 'orient': 'r1_receive_vertical'},
            r2_args={'start': r2_final, 'end': r2_final, 'orient': 'r2_present_horizontal'},
            action_name="r1_align", plan_list=full_plan
        )
        
        waypoints = [
            r1_approach_1, r1_approach_2, r1_approach_3, 
            r1_approach_4, r1_approach_5, r1_approach_6,
            r1_approach_7, r1_approach_8, r1_approach_9,
            r1_approach_10, r1_approach_11, r1_final
        ]
        
        current_start = r1_align_safe
        
        # 2. GRANULAR LOOP (All "forward")
        for i, pt in enumerate(waypoints):
            step_count = 60 if i > 6 else 80 
            step_name = f"r1_approach_step_{i+1}"
            
            self.interpolate_joint_segment(
                start_pos=current_start, end_pos=pt, steps=step_count,
                r1_args={'start': current_start, 'end': pt, 'orient': 'r1_receive_vertical'}, 
                r2_args={'start': r2_final, 'end': r2_final, 'orient': 'r2_present_horizontal'},
                action_name=step_name, 
                plan_list=full_plan
            )
            current_start = pt

        # ==========================================================
        # PHASE 4: HANDOVER EXECUTION
        # ==========================================================
        last = full_plan[-1]
        for _ in range(200): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r1_grasp"})
        for _ in range(80): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r2_release"})

        # ==========================================================
        # PHASE 5: RETREAT
        # ==========================================================
        r2_safe_height = r2_final + np.array([0, 0, 0.35])
        r2_retreat = r2_final + np.array([0, 0.3, 0])
        self.interpolate_segment(r2_final, r2_retreat, 100,
             {'start': r1_final, 'end': r1_final, 'orient': 'r1_receive_vertical'}, # Retreat in Forward/Front mode
             {'start': r2_final + np.array([0, 0, 0.3]), 'end': r2_safe_height, 'orient': 'r2_present_horizontal'},
             "r2_retreat", full_plan
        )
        print("--- R1 BACKTRACKING ---")
        
        # Define the retreat path (Reverse of approach)
        retreat_path = [
            r1_approach_9, r1_approach_8, r1_approach_7, 
            r1_approach_6, r1_approach_5, r1_approach_4,
            r1_approach_3, r1_approach_2, r1_approach_1,
            r1_align_safe, r1_home
        ]
        
        current_r1_pos = r1_final
        
        for i, pt in enumerate(retreat_path[:8]):
            step_name = f"r1_retreat_step_{i+1}"
            self.interpolate_joint_segment(
                current_r1_pos, pt, 80,
                {'start': current_r1_pos, 'end': pt, 'orient': 'r1_receive_vertical'},
                {'start': r2_safe_height, 'end': r2_safe_height, 'orient': 'r2_present_horizontal'},
                step_name, full_plan
            )
            current_r1_pos = pt
        return full_plan