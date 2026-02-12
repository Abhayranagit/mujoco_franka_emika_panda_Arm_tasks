import os
import copy
import xml.etree.ElementTree as ET

def create_dual_arm_scene():
    BASE_DIR = r"E:\6-months-internship-projects\act2\franka_emika_panda"
    ROBOT_XML = os.path.join(BASE_DIR, "panda.xml")
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    OUTPUT_XML = "dual_panda_scene.xml"
    
    if not os.path.exists(ROBOT_XML):
        print(f"Error: {ROBOT_XML} not found.")
        return None

    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    
    # --- PHYSICS SETTINGS ---
    def _inject_gripper_physics(xml_root):
        defaults = xml_root.find('default')
        if defaults:
            for d in defaults.iter('default'):
                if 'class' in d.attrib and 'fingertip' in d.attrib['class']:
                    for geom in d.findall('geom'):
                        geom.set('contype', '1')
                        geom.set('conaffinity', '1')
                        geom.set('friction', '2.0 0.1 0.005')
                        geom.set('solimp', '0.99 0.99 0.01')
                        geom.set('solref', '0.001 1')
    
    _inject_gripper_physics(root)

    def _boost_finger_strength(xml_root):
        actuators = xml_root.find('actuator')
        if actuators:
            for act in actuators:
                name = act.get('name', '')
                if 'finger' in name or 'actuator8' in name:
                    act.set('gainprm', '6000') 
                    act.set('biasprm', '0 -6000 -600')
                    act.set('ctrlrange', '-0.01 0.04') 

    _boost_finger_strength(root)

    robot_names = set()
    for node in root.iter():
        if 'name' in node.attrib: robot_names.add(node.attrib['name'])
    robot_names.add("link0") 

    def fix_assets(node):
        for elem in node.iter():
            if 'file' in elem.attrib:
                filename = os.path.basename(elem.attrib['file'])
                if filename.lower().endswith(('.stl', '.obj', '.png', '.jpg')):
                    new_path = os.path.join(ASSETS_DIR, filename).replace("\\", "/")
                    elem.attrib['file'] = new_path
                elif filename.lower().endswith('.xml'):
                    new_path = os.path.join(BASE_DIR, filename).replace("\\", "/")
                    elem.attrib['file'] = new_path
    fix_assets(root)

    worldbody = root.find('worldbody')
    if worldbody is None: worldbody = ET.SubElement(root, 'worldbody')
    
    robot_logic_tags = ['actuator', 'sensor', 'contact', 'equality', 'tendon']
    robot_bodies = [child for child in worldbody if child.tag == 'body']
    robot_logic = [child for child in root if child.tag in robot_logic_tags]
    
    REFERENCE_ATTRS = ['body1', 'body2', 'joint', 'joint1', 'joint2', 'site', 'tendon', 'objname', 'target', 'geom', 'geom1', 'geom2', 'actuator', 'sensor']

    def create_robot_namespace(prefix, pos, euler):
        cloned_bodies = []
        for body in robot_bodies:
            clone = copy.deepcopy(body)
            for node in clone.iter():
                if 'name' in node.attrib: node.attrib['name'] = f"{prefix}{node.attrib['name']}"
                for attr in REFERENCE_ATTRS:
                    if attr in node.attrib and node.attrib[attr] in robot_names:
                         node.attrib[attr] = f"{prefix}{node.attrib[attr]}"
            cloned_bodies.append(clone)

        cloned_logic = []
        for section in robot_logic:
            section_clone = copy.deepcopy(section)
            for node in section_clone.iter():
                if 'name' in node.attrib: node.attrib['name'] = f"{prefix}{node.attrib['name']}"
                for attr in REFERENCE_ATTRS:
                    if attr in node.attrib and node.attrib[attr] in robot_names:
                        node.attrib[attr] = f"{prefix}{node.attrib[attr]}"
            cloned_logic.append(section_clone)
        return cloned_bodies, cloned_logic

    new_root = ET.Element('mujoco', {'model': 'dual_panda_generated'})
    for child in root:
        if child.tag not in ['worldbody'] + robot_logic_tags:
            new_root.append(copy.deepcopy(child))

    new_wb = ET.SubElement(new_root, 'worldbody')
    ET.SubElement(new_wb, 'light', {'directional': 'true', 'pos': '0 0 3', 'dir': '0 0 -1'})
    ET.SubElement(new_wb, 'geom', {'name': 'floor_global', 'type': 'plane', 'pos': '0 0 0', 'size': '2 2 0.1', 'rgba': '.3 .3 .3 1'})

    r1_bodies, r1_logic = create_robot_namespace("panda1_", "0.15 -0.8 0", "0 0 1.57")
    r1_mount = ET.SubElement(new_wb, 'body', {'name': 'panda1_base_mount', 'pos': '0.15 -0.8 0', 'euler': '0 0 1.57'})
    for body in r1_bodies: r1_mount.append(body)
    for logic in r1_logic: new_root.append(logic)

    r2_bodies, r2_logic = create_robot_namespace("panda2_", "0 0.8 0", "0 0 -1.57")
    r2_mount = ET.SubElement(new_wb, 'body', {'name': 'panda2_base_mount', 'pos': '0 0.8 0', 'euler': '0 0 -1.57'})
    for body in r2_bodies: r2_mount.append(body)
    for logic in r2_logic: new_root.append(logic)

    # --- HAMMER ---
    hammer = ET.SubElement(new_wb, 'body', {'name': 'hammer', 'pos': '0.35 0.55 0.03', 'euler': '0 0 1.57'}) 
    ET.SubElement(hammer, 'freejoint')
    
    ET.SubElement(hammer, 'geom', {
        'name': 'hammer_handle', 'type': 'capsule', 'size': '0.015 0.15', 
        'rgba': '0.6 0.4 0.2 1', 'mass': '0.1', 
        'contype': '1', 'conaffinity': '1', 
        'solimp': '0.99 0.99 0.01', 'solref': '0.001 1', 'margin': '0.002',
        'friction': '5.0 0.1 0.005', 'condim': '4', 
        'pos': '0 0 0', 'quat': '0.707 0.707 0 0' 
    })
    
    ET.SubElement(hammer, 'geom', {
        'name': 'hammer_head', 'type': 'box', 'size': '0.06 0.03 0.03', 
        'rgba': '0.2 0.2 0.2 1', 'mass': '0.1', 
        'contype': '1', 'conaffinity': '1', 
        'pos': '0 0.15 0'
    })

    ET.SubElement(hammer, 'site', {'name': 'proper_grasp_site', 'pos': '0 0.075 0', 'size': '0.01', 'rgba': '0 1 0 0.8'})
    ET.SubElement(new_wb, 'site', {'name': 'transfer_zone', 'pos': '0.0 0 0.55', 'size': '0.02', 'rgba': '0 1 0 0.3'})

    # --- THE FIX: ADD EQUALITY CONSTRAINT (WELD) ---
    # Connects R2 Hand to Hammer. Initially disabled (active="false").
    # anchor="0 0 0.06" puts the pivot 6cm inside the hand (where hammer should be)
    equality = new_root.find('equality')
    if equality is None: equality = ET.SubElement(new_root, 'equality')
    
    # --- FIX: ADD TWO WELDS (One for each robot) ---
    equality = new_root.find('equality')
    if equality is None: equality = ET.SubElement(new_root, 'equality')
    
    # Weld 1: R2 (Sender)
    ET.SubElement(equality, 'weld', {
        'name': 'r2_hammer_weld',
        'body1': 'panda2_hand',
        'body2': 'hammer',
        'active': 'false', 
        'relpose': '0 0 0.06 0.707 0 0 0.707' # Adjusted for R2's grasp
    })

    # # Weld 2: R1 (Receiver) - NEW!
    # ET.SubElement(equality, 'weld', {
    #     'name': 'r1_hammer_weld',
    #     'body1': 'panda1_hand',
    #     'body2': 'hammer',
    #     'active': 'false', 
    #     # R1 grasps the hammer differently (usually rotated).
    #     # We set anchors to 0 0 0 so it locks "wherever it touches".
    #     # Or use a specific offset if you know R1's grasp is perfect.
    #     'relpose': '0 0 0.06 0 1 0 0' 
    # })

    tree = ET.ElementTree(new_root)
    tree.write(OUTPUT_XML, encoding='utf-8', xml_declaration=True)
    return OUTPUT_XML

if __name__ == "__main__":
    create_dual_arm_scene()