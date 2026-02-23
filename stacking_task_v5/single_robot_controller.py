import numpy as np
import mujoco
import pyroki as pk
import yourdfpy
from robot_descriptions import panda_description

class SinglePandaController:
    def __init__(self, xml_path):
        self.prefix = "panda_"
        self.urdf = yourdfpy.URDF.load(panda_description.URDF_PATH, build_scene_graph=True, load_meshes=False)
        self.kinematics_model = pk.Robot.from_urdf(self.urdf)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.reset_to_ready_pose()
        mujoco.mj_forward(self.model, self.data)
        
        self.actuator_ids = []
        self._map_actuators()
        
        # Track objects
        self.objects = {}
        target_objects = ["cube_1", "cube_2", "cube_3", "cube_4"]
        for name in target_objects:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid != -1:
                    self.objects[name] = {'id': bid, 'attached_to': None}
            except:
                pass

    def get_object_pos(self, obj_name):
        """Returns [x,y,z] if object exists, else None."""
        if obj_name in self.objects:
            bid = self.objects[obj_name]['id']
            return np.array(self.data.xpos[bid])
        
        # !!! KEY FIX: Return None instead of a default position !!!
        return None 

    def reset_to_ready_pose(self):
        ready_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and self.prefix in name:
                try:
                    idx = int(name[-1]) - 1
                    self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]] = ready_q[idx]
                except:
                    pass

    def _map_actuators(self):
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not name: continue
            if self.prefix in name: self.actuator_ids.append(i)

    def get_base_pose(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}base_mount")
        if bid == -1: bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix.rstrip('_')}_link0")
        return self.data.xpos[bid] if bid != -1 else np.zeros(3)

    def get_base_rotation(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}base_mount")
        return self.data.xmat[bid].reshape(3, 3) if bid != -1 else np.eye(3)

    def set_commands(self, joints, gripper):
        for i, idx in enumerate(self.actuator_ids):
            if i < 7: self.data.ctrl[idx] = joints[i]
            else: self.data.ctrl[idx] = 255 if gripper > 0.02 else 0 
        
        if gripper < 0.02: self._check_grasp()
        self._handle_attachments()

    def _check_grasp(self):
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}hand")
        if hand_id == -1: return
        hand_pos = self.data.xpos[hand_id]
        
        for name, obj_data in self.objects.items():
            if obj_data['attached_to'] is not None: continue
            obj_pos = self.data.xpos[obj_data['id']]
            if np.linalg.norm(hand_pos - obj_pos) < 0.03:
                obj_data['attached_to'] = self.prefix
                print(f"Attached {name} to {self.prefix}")
                return 

    def _handle_attachments(self):
        for name, obj_data in self.objects.items():
            if obj_data['attached_to'] == self.prefix:
                hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}hand")
                if hand_id != -1:
                    self.data.xpos[obj_data['id']] = self.data.xpos[hand_id]

    def detach(self):
        for name, obj_data in self.objects.items():
            if obj_data['attached_to'] == self.prefix:
                obj_data['attached_to'] = None
                print(f"Detached {name} from {self.prefix}")