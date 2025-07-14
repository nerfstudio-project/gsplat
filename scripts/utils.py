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