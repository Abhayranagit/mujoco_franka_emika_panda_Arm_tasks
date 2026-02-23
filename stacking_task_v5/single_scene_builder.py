import os
import copy
import xml.etree.ElementTree as ET

def create_single_arm_scene():
    # PATHS
    BASE_DIR = r"E:\6-months-internship-projects\act2\franka_emika_panda"
    ROBOT_XML = os.path.join(BASE_DIR, "panda.xml")
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    OUTPUT_XML = "single_panda_stacking.xml"
    
    if not os.path.exists(ROBOT_XML):
        print(f"Error: {ROBOT_XML} not found.")
        return None

    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    
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

    new_root = ET.Element('mujoco', {'model': 'single_panda_stacking'})
    for child in root:
        if child.tag not in ['worldbody', 'actuator', 'sensor', 'contact', 'equality', 'tendon']:
            new_root.append(copy.deepcopy(child))

    new_wb = ET.SubElement(new_root, 'worldbody')
    ET.SubElement(new_wb, 'light', {'directional': 'true', 'pos': '0 0 3', 'dir': '0 0 -1'})
    ET.SubElement(new_wb, 'geom', {'name': 'floor_global', 'type': 'plane', 'pos': '0 0 0', 'size': '2 2 0.1', 'rgba': '.3 .3 .3 1'})

    # Robot Cloning
    robot_bodies = [child for child in root.find('worldbody') if child.tag == 'body']
    robot_logic = [child for child in root if child.tag in ['actuator', 'sensor', 'contact', 'equality', 'tendon']]
    REF_ATTRS = ['body1', 'body2', 'joint', 'joint1', 'joint2', 'geom', 'actuator', 'tendon', 'site']

    def clone_node(node_in):
        node_out = copy.deepcopy(node_in)
        for child in node_out.iter():
            if 'name' in child.attrib: 
                child.attrib['name'] = "panda_" + child.attrib['name']
            for attr in REF_ATTRS:
                if attr in child.attrib and child.attrib[attr] in robot_names:
                    child.attrib[attr] = "panda_" + child.attrib[attr]
        return node_out

    r_mount = ET.SubElement(new_wb, 'body', {'name': 'panda_base_mount', 'pos': '0 0 0', 'euler': '0 0 0'})
    for body in robot_bodies:
        r_mount.append(clone_node(body))
    for section in robot_logic:
        new_root.append(clone_node(section))

    # --- CUBES SETUP (Edit positions here!) ---
    # Now that main.py reads this automatically, you can change these freely.
    cubes_config = [
        {'name': 'cube_1', 'pos': '0.3 0.3 0.028', 'rgba': '1 0 0 1'},
        # {'name': 'cube_2', 'pos': '0.5 0.3 0.028', 'rgba': '0 1 0 1'},
        {'name': 'cube_3', 'pos': '0.2 -0.3 0.028', 'rgba': '0 0 1 1'},
        {'name': 'cube_4', 'pos': '0.4 -0.3 0.028', 'rgba': '1 1 0 1'},
       
    ]

    for c in cubes_config:
        body = ET.SubElement(new_wb, 'body', {'name': c['name'], 'pos': c['pos']})
        ET.SubElement(body, 'freejoint')
        # SUPER HIGH FRICTION SETTINGS
        ET.SubElement(body, 'geom', {
            'name': f"{c['name']}_geom", 'type': 'box', 'size': '0.028 0.028 0.028',
            'rgba': c['rgba'], 'mass': '0.05', 
            'friction': '3.0 0.1 0.005',     # Sliding friction = 3.0 (Very sticky)
            'condim': '4',                   # Torsional sfriction enabled
            'solref': '0.01 1',              # Stiff but damped
            'solimp': '0.99 0.99 0.001',     # High impedance constraint
            'priority': '1'
        })
        
    ET.SubElement(new_wb, 'site', {'name': 'target_zone', 'pos': '0.6 0.0 0.0', 'size': '0.05', 'rgba': '0 1 1 0.3'})

    tree = ET.ElementTree(new_root)
    tree.write(OUTPUT_XML, encoding='utf-8', xml_declaration=True)
    return OUTPUT_XML

if __name__ == "__main__":
    create_single_arm_scene()