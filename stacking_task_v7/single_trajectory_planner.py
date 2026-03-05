import numpy as np
import sys
sys.path.insert(0, r"E:\6-months-internship-projects\act2\pyroki\examples")
import pyroki_snippets as pks

class SinglePlanner:
    def __init__(self, kin_model, base_pos, base_rot):
        self.robot = kin_model
        self.base_pos = np.array(base_pos)
        self.base_rot = np.array(base_rot)
        self.link_name = "panda_hand"
        
        # Hand Length Offset (Wrist must be higher than object)
        self.TCP_OFFSET = 0.100
        self.HOVER_HEIGHT = 0.25
        self.CUBE_HEIGHT = 0.05
        
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

    def generate_home_plan(self):
        plan = []
        home_pos = self.base_pos + np.array([0.3, 0.0, 0.5])
        self.interpolate_segment(home_pos, home_pos, 10, "home", plan)
        return plan, home_pos

    def generate_pick_plan(self, start_pos, target_pos, cube_name):
        """Generates a trajectory to pick up a specific object."""
        plan = []
        
        src_hover = np.array([target_pos[0] + 0.05, target_pos[1], self.HOVER_HEIGHT])
        src_grasp = np.array([target_pos[0], target_pos[1], target_pos[2] + self.TCP_OFFSET])

        # --- SLOW PICK SEQUENCE ---
        self.interpolate_segment(start_pos, src_hover, 150, f"move_to_{cube_name}", plan)
        
        # Descend
        self.interpolate_segment(src_hover, src_grasp, 150, f"descend_{cube_name}", plan)
        
        # Stabilize before closing gripper
        last_q = plan[-1]['joints']
        for _ in range(130): 
            plan.append({"joints": last_q, "action": f"descend_{cube_name}"})
        
        # Grasp wait
        for _ in range(120): 
            plan.append({"joints": last_q, "action": "grasp"})
        
        # Lift
        self.interpolate_segment(src_grasp, src_hover, 150, f"lift_{cube_name}", plan)
        
        return plan, src_hover # Return plan and new robot position

    def generate_place_plan(self, start_pos, target_xy, target_z, cube_name):
        """Generates a trajectory to place an object at a target zone."""
        plan = []
        
        tgt_hover = np.array([target_xy[0], target_xy[1], self.HOVER_HEIGHT + 0.05])
        tgt_place = np.array([target_xy[0], target_xy[1], target_z + self.TCP_OFFSET]) 

        # --- SLOW PLACE SEQUENCE ---
        self.interpolate_segment(start_pos, tgt_hover, 100, f"transfer_{cube_name}", plan)
        
        # Stack
        self.interpolate_segment(tgt_hover, tgt_place, 120, f"stack_{cube_name}", plan)
        
        # Stabilize before opening
        last_q = plan[-1]['joints']
        for _ in range(50): 
            plan.append({"joints": last_q, "action": f"stack_{cube_name}"})
            
        # Release wait
        for _ in range(90): 
            plan.append({"joints": last_q, "action": "release"})
        
        # Retreat
        self.interpolate_segment(tgt_place, tgt_hover, 70, "retreat", plan)

        return plan, tgt_hover # Return plan and new robot position