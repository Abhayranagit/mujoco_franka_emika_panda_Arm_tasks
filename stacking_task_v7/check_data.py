import h5py
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the filename (change this to your actual file's name)
filename = "mujoco_episode_1771839354.hdf5"

def explore_hdf5_structure(group, indent=""):
    """Recursively print HDF5 file structure with dataset shapes"""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print(f"{indent}📊 {key}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{indent}📁 {key}/")
            explore_hdf5_structure(item, indent + "  ")

try:
    # 2. Open the file in 'read' mode
    with h5py.File(filename, 'r') as f:
        print("=" * 70)
        print("HDF5 FILE STRUCTURE")
        print("=" * 70)
        explore_hdf5_structure(f)
        print("\n" + "=" * 70)
        print("QUICK SUMMARY: DATA COLLECTION FREQUENCIES")
        print("=" * 70)
        
        # Get quick stats
        actions_count = None
        qpos_count = None
        image_counts = {}
        
        if 'actions' in f:
            actions_count = f['actions'].shape[0]
            print(f"\nACTIONS / QPOS timesteps: {actions_count}")
        elif 'action' in f:
            actions_count = f['action'].shape[0]
            print(f"\nACTIONS / QPOS timesteps: {actions_count}")
        
        if 'observations/state' in f:
            qpos_count = f['observations/state'].shape[0]
            if actions_count is None:
                print(f"\nQPOS timesteps: {qpos_count}")
        
        if 'observations/images' in f:
            for cam in f['observations/images'].keys():
                image_counts[cam] = f[f'observations/images/{cam}'].shape[0]
            
            print(f"\nIMAGE FRAMES:")
            for cam, count in image_counts.items():
                print(f"  - {cam}: {count} frames")
                
                # Show frequency mismatch explanation
                if actions_count:
                    ratio = count / actions_count
                    print(f"    Ratio to actions: {ratio:.3f} (1 action per {1/ratio:.1f} frames OR {ratio:.1f} frames per action)")
        
        print()
        
        # 3. Get timestep information
        if 'observations' in f:
            obs_keys = list(f['observations'].keys())
            print(f"\nObservation keys: {obs_keys}")
            
            # Find a non-image observation to get timesteps
            timestep_data = None
            for key in obs_keys:
                if key != 'images':
                    try:
                        data = f[f'observations/{key}']
                        if len(data.shape) > 0:
                            timestep_data = data
                            print(f"Using '{key}' for timestep count")
                            break
                    except:
                        continue
            
            if timestep_data is not None:
                num_timesteps = timestep_data.shape[0]
                print(f"Total timesteps: {num_timesteps}")
        
        # 4. Get actions if available
        if 'actions' in f:
            actions = f['actions']
            print(f"\nActions available:")
            print(f"  - Shape: {actions.shape}")
            print(f"  - Dtype: {actions.dtype}")
            num_timesteps = actions.shape[0]
            print(f"  - Number of action timesteps: {num_timesteps}")
            
            # Show first few action values
            print(f"  - First 3 timestep actions:")
            for i in range(min(3, num_timesteps)):
                print(f"    Timestep {i}: {actions[i]}")
        
        # 5. Get state/joint information
        if 'observations/state' in f:
            state = f['observations/state']
            print(f"\nState (Joint positions) available:")
            print(f"  - Shape: {state.shape}")
            print(f"  - Dtype: {state.dtype}")
            if len(state.shape) > 0:
                print(f"  - Number of state timesteps: {state.shape[0]}")
                if len(state.shape) > 1:
                    print(f"  - Dimensions per timestep: {state.shape[1]}")
                
                # Show first few state values
                print(f"  - First 3 timestep states (joint values):")
                for i in range(min(3, state.shape[0])):
                    print(f"    Timestep {i}: {state[i]}")
        
        # 6. Map frames to timesteps with detailed analysis
        if 'observations/images' in f:
            available_cameras = list(f['observations/images'].keys())
            print(f"\n{'='*70}")
            print("FRAME TO TIMESTEP MAPPING & FREQUENCY ANALYSIS")
            print(f"{'='*70}")
            print(f"Available cameras: {available_cameras}\n")
            
            # Get reference timesteps from actions or state
            num_timesteps = None
            if 'actions' in f:
                num_timesteps = f['actions'].shape[0]
                print(f"Reference timesteps (from ACTIONS): {num_timesteps}\n")
            elif 'action' in f:
                num_timesteps = f['action'].shape[0]
                print(f"Reference timesteps (from ACTION): {num_timesteps}\n")
            elif 'observations/state' in f:
                num_timesteps = f['observations/state'].shape[0]
                print(f"Reference timesteps (from QPOS): {num_timesteps}\n")
            
            for cam in available_cameras:
                cam_frames = f[f'observations/images/{cam}'].shape[0]
                print(f"Camera '{cam}':")
                print(f"  - Total frames: {cam_frames}")
                
                if num_timesteps is not None:
                    ratio = cam_frames / num_timesteps
                    print(f"  - Frames : Timesteps ratio = {cam_frames} : {num_timesteps} = {ratio:.3f}")
                    
                    if abs(ratio - 1.0) < 0.01:
                        print(f"  - ✓ 1:1 mapping (1 frame per timestep)")
                    elif abs(ratio - 0.5) < 0.01:
                        print(f"  - ⚠ 1:2 mapping (1 frame per 2 timesteps - frames at lower frequency)")
                    elif abs(ratio - 2.0) < 0.01:
                        print(f"  - ⚠ 2:1 mapping (2 frames per timestep - frames at higher frequency)")
                    else:
                        print(f"  - ⚠ Non-standard mapping (frames captured at different frequency)")
                    
                    # Show frame-to-timestep mapping
                    print(f"\n  Timestep-to-Frame mapping:")
                    if num_timesteps <= 10:
                        # Show all timesteps
                        for ts in range(num_timesteps):
                            frame_idx = int(ts * ratio)
                            if frame_idx < cam_frames:
                                print(f"    Timestep {ts} → Frame {frame_idx}")
                    else:
                        # Show sample timesteps
                        sample_timesteps = [0, 1, 2, num_timesteps//4, num_timesteps//2, 
                                          3*num_timesteps//4, num_timesteps-3, num_timesteps-2, num_timesteps-1]
                        sample_timesteps = sorted(set([t for t in sample_timesteps if 0 <= t < num_timesteps]))
                        
                        for ts in sample_timesteps:
                            frame_idx = int(ts * ratio)
                            if frame_idx < cam_frames:
                                print(f"    Timestep {ts} → Frame {frame_idx}")
                        print(f"    ... and {num_timesteps - len(sample_timesteps)} more timesteps")
                print()
        
        # 7. Analyze qpos and action values in detail
        print(f"\n{'='*70}")
        print("DETAILED ANALYSIS: QPOS vs ACTIONS")
        print(f"{'='*70}")
        
        # Check for qpos
        qpos_data = None
        qpos_path = None
        
        if 'observations/state' in f:
            qpos_data = f['observations/state']
            qpos_path = 'observations/state'
        elif 'qpos' in f:
            qpos_data = f['qpos']
            qpos_path = 'qpos'
        elif 'observations/qpos' in f:
            qpos_data = f['observations/qpos']
            qpos_path = 'observations/qpos'
        
        # Check for actions
        actions_data = None
        actions_path = None
        if 'actions' in f:
            actions_data = f['actions']
            actions_path = 'actions'
        elif 'action' in f:
            actions_data = f['action']
            actions_path = 'action'
        
        # Display qpos information
        if qpos_data is not None:
            print(f"\nQPOS (Joint Positions):")
            print(f"  - Path: '{qpos_path}'")
            print(f"  - Shape: {qpos_data.shape}")
            print(f"  - Dtype: {qpos_data.dtype}")
            num_qpos = qpos_data.shape[0]
            num_joints = qpos_data.shape[1] if len(qpos_data.shape) > 1 else 1
            print(f"  - Total timesteps: {num_qpos}")
            print(f"  - Joint values per timestep: {num_joints}")
            print(f"  - TOTAL QPOS VALUES: {num_qpos * num_joints} (across all timesteps)")
            
            print(f"\n  Sample qpos values (first 5 timesteps):")
            for i in range(min(5, num_qpos)):
                print(f"    Timestep {i}: {qpos_data[i]}")
        else:
            print(f"\nQPOS: Not found in file")
            num_qpos = 0
            num_joints = 0
        
        # Display actions information
        if actions_data is not None:
            print(f"\nACTIONS:")
            print(f"  - Path: '{actions_path}'")
            print(f"  - Shape: {actions_data.shape}")
            print(f"  - Dtype: {actions_data.dtype}")
            num_actions = actions_data.shape[0]
            num_action_dims = actions_data.shape[1] if len(actions_data.shape) > 1 else 1
            print(f"  - Total timesteps: {num_actions}")
            print(f"  - Action dimensions per timestep: {num_action_dims}")
            print(f"  - TOTAL ACTION VALUES: {num_actions * num_action_dims} (across all timesteps)")
            
            print(f"\n  Sample action values (first 5 timesteps):")
            for i in range(min(5, num_actions)):
                print(f"    Timestep {i}: {actions_data[i]}")
        else:
            print(f"\nACTIONS: Not found in file")
            num_actions = 0
            num_action_dims = 0
        
        # Compare qpos and actions
        if qpos_data is not None and actions_data is not None:
            print(f"\n{'='*70}")
            print("COMPARISON: QPOS vs ACTIONS")
            print(f"{'='*70}")
            
            if num_joints == num_action_dims:
                print(f"✓ Same dimensions: {num_joints} values per timestep")
                print(f"  → Actions likely represent JOINT ANGLE TARGETS/COMMANDS")
            else:
                print(f"✗ Different dimensions:")
                print(f"  - QPOS: {num_joints} values per timestep")
                print(f"  - Actions: {num_action_dims} values per timestep")
                print(f"  → Actions may include additional control signals (e.g., gripper, etc.)")
            
            print(f"\nTimestep alignment:")
            if num_qpos == num_actions:
                print(f"✓ Same number of timesteps: {num_actions}")
            elif num_qpos == num_actions + 1:
                print(f"⚠ QPOS has 1 extra timestep: {num_qpos} vs {num_actions}")
                print(f"  → Initial observation before first action")
            else:
                print(f"⚠ Different counts:")
                print(f"  - QPOS timesteps: {num_qpos}")
                print(f"  - Action timesteps: {num_actions}")
                print(f"  - Difference: {abs(num_qpos - num_actions)} timesteps")
                
                # Check if there's a ratio relationship
                if num_qpos > 0 and num_actions > 0:
                    ratio = num_actions / num_qpos
                    print(f"  - Ratio (action:qpos): {ratio:.3f}")
                    if abs(ratio - 1.0) < 0.01:
                        print(f"    → Generally 1:1 mapping (minor alignment issues)")
                    elif abs(ratio - 2.0) < 0.01:
                        print(f"    → Actions at 2x frequency of qpos")
                    elif abs(ratio - 0.5) < 0.01:
                        print(f"    → Qpos at 2x frequency of actions")
        
        # Check if actions match joint angles by showing correlation
        if qpos_data is not None and actions_data is not None:
            print(f"\n{'='*70}")
            print("ACTION NATURE ANALYSIS")
            print(f"{'='*70}")
            
            # Get min/max values
            qpos_min = np.min(qpos_data)
            qpos_max = np.max(qpos_data)
            actions_min = np.min(actions_data)
            actions_max = np.max(actions_data)
            
            print(f"\nJoint position ranges (QPOS):")
            print(f"  - Min: {qpos_min:.4f}, Max: {qpos_max:.4f}")
            
            print(f"\nAction value ranges:")
            print(f"  - Min: {actions_min:.4f}, Max: {actions_max:.4f}")
            
            # Check if ranges are similar (suggesting joint angles)
            if abs(qpos_min - actions_min) < 1.0 and abs(qpos_max - actions_max) < 1.0:
                print(f"\n✓ Ranges are similar → Actions likely represent JOINT ANGLES")
            elif actions_max <= 1.0 and actions_min >= -1.0:
                print(f"\n⚠ Actions in [-1, 1] range → May be NORMALIZED joint targets or velocities")
            else:
                print(f"\n❓ Uncertain → Actions might be velocities or other control signals")
        
        # 8. Visualization: Display frame with qpos and actions
        if 'observations/images' in f:
            print(f"\n{'='*70}")
            print("VISUALIZATION: SAMPLE FRAME WITH QPOS & ACTIONS")
            print(f"{'='*70}")
            
            camera_name = list(f['observations/images'].keys())[0]
            images = f[f'observations/images/{camera_name}']
            
            total_frames = images.shape[0]
            frame_idx = total_frames // 2
            
            img_array = images[frame_idx]
            
            # Get corresponding qpos and action
            qpos_at_frame = None
            action_at_frame = None
            
            if qpos_data is not None and frame_idx < qpos_data.shape[0]:
                qpos_at_frame = qpos_data[frame_idx]
            
            if actions_data is not None and frame_idx < actions_data.shape[0]:
                action_at_frame = actions_data[frame_idx]
            
            print(f"Frame {frame_idx} (Camera: {camera_name})")
            print(f"  - QPOS: {qpos_at_frame}")
            print(f"  - Action: {action_at_frame}")
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot 1: Image
            axes[0].imshow(img_array)
            axes[0].set_title(f"Frame {frame_idx}")
            axes[0].axis('off')
            
            # Plot 2: QPOS values
            if qpos_at_frame is not None:
                axes[1].bar(range(len(qpos_at_frame)), qpos_at_frame, color='blue', alpha=0.7)
                axes[1].set_xlabel("Joint Index")
                axes[1].set_ylabel("Position (radians)")
                axes[1].set_title("QPOS (Actual Joint Positions)")
                axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Action values
            if action_at_frame is not None:
                axes[2].bar(range(len(action_at_frame)), action_at_frame, color='green', alpha=0.7)
                axes[2].set_xlabel("Action Dimension")
                axes[2].set_ylabel("Value")
                axes[2].set_title("Actions (Commands)")
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_name = f"check_frame_{frame_idx}_analysis.png"
            plt.savefig(save_name, bbox_inches='tight', dpi=100)
            print(f"\nSaved visualization as '{save_name}'")
            plt.show()

except FileNotFoundError:
    print(f"Error: Could not find '{filename}'. Check the file name and path.")
except KeyError as e:
    print(f"Error: Could not find the expected path in the HDF5 file: {e}")
except Exception as e:
    print(f"Error: {e}")