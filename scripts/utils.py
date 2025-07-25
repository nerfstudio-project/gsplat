from examples.config import Config
from typing import Optional

def extract_path_components(data_dir_path):
    """
    Extract Actor01, Sequence1, and resolution_1 from path like:
    /main/rajrup/Dropbox/Project/GsplatStream/gsplat/data/Actor01/Sequence1/0/resolution_1
    """
    parts = data_dir_path.strip('/').split('/')
    
    # Find the index of 'data' in the path
    try:
        data_index = parts.index('data')
        if data_index + 3 < len(parts):
            actor_name = parts[data_index + 1]  # Actor01
            sequence_name = parts[data_index + 2]  # Sequence1
            resolution_name = parts[data_index + 4]  # resolution_1 (skip frame_id at index 3)
            return actor_name, sequence_name, resolution_name
    except (ValueError, IndexError):
        pass
    
    return None, None, None

def set_result_dir(config: Config, exp_name: str, sub_exp_name: Optional[str] = None):
    data_dir = config.data_dir
    scene_id = config.scene_id
    actor, sequence, resolution = extract_path_components(data_dir)
    if sub_exp_name is not None:
        config.result_dir = f"./results/{exp_name}/{actor}/{sequence}/{resolution}/{scene_id}/{sub_exp_name}"
    else:
        config.result_dir = f"./results/{exp_name}/{actor}/{sequence}/{resolution}/{scene_id}"
        
def set_result_dir_voxelize(config: Config, exp_name: str, voxel_size: float, aggregate_method: str):
    data_dir = config.data_dir
    scene_id = config.scene_id
    actor, sequence, resolution = extract_path_components(data_dir)
    config.result_dir = f"./results/{exp_name}_to_voxel_size_{voxel_size}/{actor}/{sequence}/{resolution}/{scene_id}/{aggregate_method}/"