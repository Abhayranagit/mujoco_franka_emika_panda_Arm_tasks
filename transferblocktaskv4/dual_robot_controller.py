import numpy as np
import mujoco
import pyroki as pk
import yourdfpy
from robot_descriptions import panda_description

class DualPandaController:
    def __init__(self, xml_path):
        self.urdf = yourdfpy.URDF.load(panda_description.URDF_PATH, build_scene_graph=True, load_meshes=False)
        self.kinematics_model = pk.Robot.from_urdf(self.urdf)
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.reset_to_ready_pose()
        mujoco.mj_forward(self.model, self.data)
        
        self.r1_prefix = "panda1_"
        self.r2_prefix = "panda2_"
        self.r1_actuator_ids = []
        self.r2_actuator_ids = []
        self._map_actuators()
        
        self.object_name = "hammer" 
        try:
            self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        except:
            self.object_body_id = -1
            
        try:
            self.grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "proper_grasp_site")
        except:
            self.grasp_site_id = -1

        # Weld Constraint ID
        try:
            self.weld_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "r2_hammer_weld")
        except:
            self.weld_eq_id = -1
            
        self.cube_attached_to = None 

    def reset_to_ready_pose(self):
        ready_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if "panda1_joint" in name:
                self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]] = ready_q[int(name[-1]) - 1]
            elif "panda2_joint" in name:
                self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]] = ready_q[int(name[-1]) - 1]

    def _map_actuators(self):
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not name: continue
            if self.r1_prefix in name: self.r1_actuator_ids.append(i)
            elif self.r2_prefix in name: self.r2_actuator_ids.append(i)

    def get_base_pose(self, prefix):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_mount")
        if bid == -1: bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix.rstrip('_')}_link0")
        return self.data.xpos[bid] if bid != -1 else np.zeros(3)

    def get_base_rotation(self, prefix):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_mount")
        return self.data.xmat[bid].reshape(3, 3) if bid != -1 else np.eye(3)

    def set_commands(self, r1_joints, r1_gripper, r2_joints, r2_gripper):
        for i, idx in enumerate(self.r1_actuator_ids):
            if i < 7: self.data.ctrl[idx] = r1_joints[i]
            else: self.data.ctrl[idx] = r1_gripper 
        
        for i, idx in enumerate(self.r2_actuator_ids):
            if i < 7: self.data.ctrl[idx] = r2_joints[i]
            else: self.data.ctrl[idx] = r2_gripper

        # Trigger Check for R2 (The Sender)
        if r2_gripper < 0.02: self._check_grasp("panda2_")
        
        # We also add a check for R1 in case you want R1 to grab it later
        if r1_gripper < 0.02: self._check_grasp("panda1_")

    def _check_grasp(self, prefix):
        if self.cube_attached_to: return
        
        finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}leftfinger")
        if finger_id == -1: finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}hand")
        if finger_id == -1: return
        
        if self.grasp_site_id != -1:
            target_pos = self.data.site_xpos[self.grasp_site_id]
        else:
            target_pos = self.data.xpos[self.object_body_id]
            
        finger_pos = self.data.xpos[finger_id]
        dist = np.linalg.norm(finger_pos - target_pos)

        if dist < 0.30:
             print(f"DEBUG: {prefix} gap: {dist:.4f}")

        # If close enough, activate the PHYSICS WELD
        if dist < 0.25: 
            print(f"!!! PHYSICS WELD ACTIVATED !!!")
            self.cube_attached_to = prefix
            
            if self.weld_eq_id != -1:
                # --- FIX 1: Use self.data instead of self.model ---
                self.data.eq_active[self.weld_eq_id] = 1

    def attach(self, robot_prefix): self.cube_attached_to = robot_prefix 
    def detach(self): 
        self.cube_attached_to = None
        if self.weld_eq_id != -1:
            # --- FIX 2: Use self.data instead of self.model ---
            self.data.eq_active[self.weld_eq_id] = 0