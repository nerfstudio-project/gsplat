"""
Convert ActorsHQ dataset from original format to COLMAP format.

Original format:
    actorshq/
    ├── Actor01/
    │   └── Sequence1/
    │       ├── scene.json
    │       ├── aabbs.csv
    │       ├── occupancy_grids/
    │       │   ├── occupancy_grid000000.npz
    │       │   └── ...
    │       ├── 4x/
    │       │   ├── calibration.csv
    │       │   ├── light_annotations.csv
    │       │   ├── rgbs/
    │       │   │   ├── Cam001/
    │       │   │   │   ├── Cam001_rgb000000.jpg
    │       │   │   │   └── ...
    │       │   │   └── ...
    │       │   └── masks/
    │       │       ├── Cam001/
    │       │       │   ├── Cam001_mask000000.png
    │       │       │   └── ...
    │       │       └── ...
    │       ├── 2x/
    │       └── 1x/

COLMAP format:
    actorshq/
    ├── colmap/
    │   ├── Actor01/
    │   │   └── Sequence1/
    │   │       ├── metadata/
    │   │       │   ├── scene.json
    │   │       │   └── aabbs.csv
    │   │       ├── calibration_gt_4/
    │   │       │   ├── calibration.csv
    │   │       │   └── light_annotations.csv
    │   │       ├── calibration_gt_2/
    │   │       ├── calibration_gt/
    │   │       └── frames/
    │   │           ├── frame0/
    │   │           │   ├── occupancy_grid000000.npz
    │   │           │   ├── images_4/
    │   │           │   │   ├── Cam001_rgb000000.png
    │   │           │   │   └── ...
    │   │           │   ├── images_2/
    │   │           │   ├── images/
    │   │           │   ├── masks_4/
    │   │           │   ├── masks_2/
    │   │           │   └── masks/
    │   │           ├── frame1/
    │   │           │   ├── occupancy_grid000001.npz
    │   │           │   └── ...
    │   │           └── ...
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActorsHQToColmapConverter:
    def __init__(self, input_path: str, output_path: str = None, force: bool = False):
        """
        Initialize the converter.
        
        Args:
            input_path: Path to the original ActorsHQ dataset
            output_path: Path where the COLMAP format will be created (optional)
            force: Force re-conversion even if files already exist
        """
        self.input_path = Path(input_path)
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.input_path.parent / "actorshq_colmap"
        
        self.force = force
        
        # Resolution mappings
        self.res_mappings = {
            '4x': {'cal_dir': 'calibration_gt_4', 'img_dir': 'images_4', 'mask_dir': 'masks_4'},
            '2x': {'cal_dir': 'calibration_gt_2', 'img_dir': 'images_2', 'mask_dir': 'masks_2'},
            '1x': {'cal_dir': 'calibration_gt', 'img_dir': 'images', 'mask_dir': 'masks'}
        }
        
    def create_directory_structure(self, actor: str, sequence: str) -> Path:
        """Create the COLMAP directory structure for an actor/sequence."""
        colmap_base = self.output_path / "colmap" / actor / sequence
        
        # Create main directories
        colmap_base.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        metadata_dir = colmap_base / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Create calibration directories
        for res in self.res_mappings:
            cal_dir = colmap_base / self.res_mappings[res]['cal_dir']
            cal_dir.mkdir(exist_ok=True)
        
        # Create frames directory
        frames_dir = colmap_base / "frames"
        frames_dir.mkdir(exist_ok=True)
            
        return colmap_base
    
    def copy_metadata(self, src_sequence_path: Path, dst_sequence_path: Path):
        """Copy metadata files (scene.json and aabbs.csv)."""
        metadata_dir = dst_sequence_path / "metadata"
        
        # Copy scene.json
        scene_json_src = src_sequence_path / "scene.json"
        scene_json_dst = metadata_dir / "scene.json"
        if scene_json_src.exists():
            if not scene_json_dst.exists() or self.force:
                shutil.copy2(scene_json_src, scene_json_dst)
                logger.info(f"Copied scene.json to {metadata_dir}")
            else:
                logger.info(f"Skipped scene.json (already exists): {scene_json_dst}")
        else:
            logger.warning(f"scene.json not found in {src_sequence_path}")
        
        # Copy aabbs.csv if it exists, otherwise create placeholder
        aabbs_src = src_sequence_path / "aabbs.csv"
        aabbs_dst = metadata_dir / "aabbs.csv"
        if not aabbs_dst.exists() or self.force:
            if aabbs_src.exists():
                shutil.copy2(aabbs_src, aabbs_dst)
                logger.info(f"Copied aabbs.csv to {metadata_dir}")
            else:
                # Create a placeholder AABB file
                with open(aabbs_dst, 'w') as f:
                    f.write("# AABB placeholder - needs to be populated with actual scene bounds\n")
                    f.write("min_x,min_y,min_z,max_x,max_y,max_z\n")
                    f.write("-1.0,-1.0,-1.0,1.0,1.0,1.0\n")
                logger.info(f"Created placeholder aabbs.csv at {aabbs_dst}")
        else:
            logger.info(f"Skipped aabbs.csv (already exists): {aabbs_dst}")
    
    def copy_calibration_files(self, src_sequence_path: Path, dst_sequence_path: Path):
        """Copy calibration files from original format to COLMAP format."""
        for res in self.res_mappings:
            src_res_dir = src_sequence_path / res
            dst_cal_dir = dst_sequence_path / self.res_mappings[res]['cal_dir']
            
            if src_res_dir.exists():
                # Copy calibration.csv
                cal_src = src_res_dir / "calibration.csv"
                cal_dst = dst_cal_dir / "calibration.csv"
                if cal_src.exists():
                    if not cal_dst.exists() or self.force:
                        shutil.copy2(cal_src, cal_dst)
                        logger.info(f"Copied calibration.csv for {res} resolution")
                    else:
                        logger.info(f"Skipped calibration.csv for {res} resolution (already exists)")
                
                # Copy light_annotations.csv
                light_src = src_res_dir / "light_annotations.csv"
                light_dst = dst_cal_dir / "light_annotations.csv"
                if light_src.exists():
                    if not light_dst.exists() or self.force:
                        shutil.copy2(light_src, light_dst)
                        logger.info(f"Copied light_annotations.csv for {res} resolution")
                    else:
                        logger.info(f"Skipped light_annotations.csv for {res} resolution (already exists)")
    
    def get_frame_count(self, scene_json_path: Path) -> int:
        """Get the number of frames from scene.json."""
        try:
            with open(scene_json_path, 'r') as f:
                data = json.load(f)
                return data.get('num_frames', 0)
        except:
            logger.warning(f"Could not read frame count from {scene_json_path}")
            return 0
    
    def get_camera_list(self, rgbs_dir: Path) -> List[str]:
        """Get list of cameras from the rgbs directory."""
        cameras = []
        if rgbs_dir.exists():
            for cam_dir in rgbs_dir.iterdir():
                if cam_dir.is_dir() and cam_dir.name.startswith('Cam'):
                    cameras.append(cam_dir.name)
        return sorted(cameras)
    
    def convert_or_copy_image(self, src_path: Path, dst_path: Path):
        """Convert JPEG to PNG or copy PNG files using OpenCV for optimal performance."""
        if not src_path.exists():
            return False
        
        # Skip if destination already exists (unless force is enabled)
        if dst_path.exists() and not self.force:
            return True
        
        # Check file extension
        src_ext = src_path.suffix.lower()
        
        if src_ext in ['.jpg', '.jpeg']:
            # Convert JPEG to PNG using OpenCV (faster than PIL)
            try:
                # Read image with OpenCV
                img = cv2.imread(str(src_path))
                if img is None:
                    logger.error(f"Failed to read image: {src_path}")
                    return False
                
                # Write as PNG (OpenCV handles format conversion automatically)
                success = cv2.imwrite(str(dst_path), img)
                if not success:
                    logger.error(f"Failed to write PNG: {dst_path}")
                    return False
                
                return True
            except Exception as e:
                logger.error(f"Failed to convert {src_path} to PNG: {e}")
                return False
        elif src_ext == '.png':
            # Copy PNG file directly (fastest option for PNG)
            try:
                shutil.copy2(src_path, dst_path)
                return True
            except Exception as e:
                logger.error(f"Failed to copy {src_path}: {e}")
                return False
        else:
            logger.warning(f"Unsupported image format: {src_ext} for {src_path}")
            return False
    
    def copy_occupancy_grids(self, src_sequence_path: Path, dst_sequence_path: Path, num_frames: int):
        """Copy occupancy grid files from occupancy_grids/ to each frame directory."""
        src_occupancy_dir = src_sequence_path / "occupancy_grids"
        if not src_occupancy_dir.exists():
            logger.warning(f"Occupancy grids directory not found: {src_occupancy_dir}")
            return
        
        logger.info(f"Copying occupancy grids for {num_frames} frames")
        
        for frame_idx in tqdm(range(num_frames), desc="Copying occupancy grids"):
            frame_dir = dst_sequence_path / "frames" / f"frame{frame_idx}"
            
            # Copy occupancy grid file
            src_occupancy_file = src_occupancy_dir / f"occupancy_grid{frame_idx:06d}.npz"
            dst_occupancy_file = frame_dir / f"occupancy_grid{frame_idx:06d}.npz"
            
            if src_occupancy_file.exists():
                if not dst_occupancy_file.exists() or self.force:
                    shutil.copy2(src_occupancy_file, dst_occupancy_file)
                else:
                    logger.debug(f"Skipped occupancy grid for frame {frame_idx} (already exists)")
            else:
                logger.warning(f"Occupancy grid not found for frame {frame_idx}: {src_occupancy_file}")
    
    def convert_images_by_frame(self, src_sequence_path: Path, dst_sequence_path: Path):
        """Convert images from camera-centric to frame-centric organization."""
        # Get frame count from scene.json
        scene_json_path = src_sequence_path / "scene.json"
        num_frames = self.get_frame_count(scene_json_path)
        
        if num_frames == 0:
            logger.warning(f"No frames found for {src_sequence_path}")
            return
        
        # Copy occupancy grids first
        self.copy_occupancy_grids(src_sequence_path, dst_sequence_path, num_frames)
        
        # Process each resolution
        for res in sorted(self.res_mappings.keys()):
            src_res_dir = src_sequence_path / res
            if not src_res_dir.exists():
                continue
                
            src_rgbs_dir = src_res_dir / "rgbs"
            src_masks_dir = src_res_dir / "masks"
            
            if not src_rgbs_dir.exists():
                logger.warning(f"RGB directory not found: {src_rgbs_dir}")
                continue
            
            # Get camera list
            cameras = self.get_camera_list(src_rgbs_dir)
            if not cameras:
                logger.warning(f"No cameras found in {src_rgbs_dir}")
                continue
            
            logger.info(f"Converting {res} resolution with {len(cameras)} cameras and {num_frames} frames")
            
            # Process each frame
            for frame_idx in tqdm(range(num_frames), desc=f"Converting {res} frames"):
                frame_dir = dst_sequence_path / "frames" / f"frame{frame_idx}"
                frame_dir.mkdir(exist_ok=True)
                
                # Create resolution-specific directories
                img_dir = frame_dir / self.res_mappings[res]['img_dir']
                mask_dir = frame_dir / self.res_mappings[res]['mask_dir']
                img_dir.mkdir(exist_ok=True)
                mask_dir.mkdir(exist_ok=True)
                
                # Process each camera for this frame
                for camera in cameras:
                    # Convert RGB image - try both JPG and PNG extensions
                    dst_img_path = img_dir / f"{camera}_rgb{frame_idx:06d}.png"
                    
                    # Try different source file extensions
                    src_img_paths = [
                        src_rgbs_dir / camera / f"{camera}_rgb{frame_idx:06d}.jpg",
                        src_rgbs_dir / camera / f"{camera}_rgb{frame_idx:06d}.jpeg",
                        src_rgbs_dir / camera / f"{camera}_rgb{frame_idx:06d}.png"
                    ]
                    
                    converted = False
                    skipped = False
                    for src_img_path in src_img_paths:
                        if src_img_path.exists():
                            if dst_img_path.exists() and not self.force:
                                skipped = True
                                break
                            if self.convert_or_copy_image(src_img_path, dst_img_path):
                                converted = True
                                break
                    
                    if not converted and not skipped:
                        logger.warning(f"No RGB image found for {camera} frame {frame_idx}")
                    
                    # Convert mask if available - try both JPG and PNG extensions
                    if src_masks_dir.exists():
                        dst_mask_path = mask_dir / f"{camera}_mask{frame_idx:06d}.png"
                        
                        # Try different source mask file extensions
                        src_mask_paths = [
                            src_masks_dir / camera / f"{camera}_mask{frame_idx:06d}.png",
                            src_masks_dir / camera / f"{camera}_mask{frame_idx:06d}.jpg",
                            src_masks_dir / camera / f"{camera}_mask{frame_idx:06d}.jpeg"
                        ]
                        
                        converted = False
                        skipped = False
                        for src_mask_path in src_mask_paths:
                            if src_mask_path.exists():
                                if dst_mask_path.exists() and not self.force:
                                    skipped = True
                                    break
                                if self.convert_or_copy_image(src_mask_path, dst_mask_path):
                                    converted = True
                                    break
                        
                        if not converted and not skipped:
                            logger.warning(f"No mask found for {camera} frame {frame_idx}")
    
    def convert_actor_sequence(self, actor: str, sequence: str):
        """Convert a single actor/sequence from original to COLMAP format."""
        logger.info(f"Converting {actor}/{sequence}")
        
        src_sequence_path = self.input_path / actor / sequence
        if not src_sequence_path.exists():
            logger.error(f"Source sequence path does not exist: {src_sequence_path}")
            return
        
        # Create COLMAP directory structure
        dst_sequence_path = self.create_directory_structure(actor, sequence)
        
        # Copy metadata
        self.copy_metadata(src_sequence_path, dst_sequence_path)
        
        # Copy calibration files
        self.copy_calibration_files(src_sequence_path, dst_sequence_path)
        
        # Convert images to frame-centric format
        self.convert_images_by_frame(src_sequence_path, dst_sequence_path)
        
        logger.info(f"Completed conversion for {actor}/{sequence}")
    
    def convert_all(self):
        """Convert all actors and sequences in the dataset."""
        if not self.input_path.exists():
            logger.error(f"Input path does not exist: {self.input_path}")
            return
        
        # Find all actors
        actors = []
        for item in self.input_path.iterdir():
            if item.is_dir() and item.name.startswith('Actor'):
                actors.append(item.name)
        
        actors.sort()
        logger.info(f"Found actors: {actors}")
        
        # Process each actor
        for actor in actors:
            actor_path = self.input_path / actor
            
            # Find sequences for this actor
            sequences = []
            for item in actor_path.iterdir():
                if item.is_dir() and item.name.startswith('Sequence'):
                    sequences.append(item.name)
            
            sequences.sort()
            logger.info(f"Found sequences for {actor}: {sequences}")
            
            # Convert each sequence
            for sequence in sequences:
                self.convert_actor_sequence(actor, sequence)
    
    def convert_specific(self, actor: str, sequence: str = "Sequence1"):
        """Convert a specific actor/sequence."""
        self.convert_actor_sequence(actor, sequence)

def main():
    parser = argparse.ArgumentParser(description='Convert ActorsHQ dataset to COLMAP format')
    parser.add_argument('--input', type=str, default='/bigdata2/rajrup/datasets/actorshq/',
                        help='Path to the original ActorsHQ dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for COLMAP format (default: input_path/../actorshq_colmap)')
    parser.add_argument('--actor', type=str, default=None,
                        help='Specific actor to convert (e.g., Actor01)')
    parser.add_argument('--sequence', type=str, default='Sequence1',
                        help='Specific sequence to convert (default: Sequence1)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-conversion even if files already exist')
    
    args = parser.parse_args()
    
    converter = ActorsHQToColmapConverter(args.input, args.output, args.force)
    
    if args.actor:
        logger.info(f"Converting specific actor: {args.actor}")
        converter.convert_specific(args.actor, args.sequence)
    else:
        logger.info("Converting all actors and sequences")
        converter.convert_all()
    
    logger.info("Conversion completed!")

if __name__ == "__main__":
    main() 
    
'''
# Convert All Actors from Custom Input Path
python convert_actorshq_to_colmap.py --input /bigdata2/rajrup/datasets/actorshq/ --output /bigdata2/rajrup/datasets/actorshq/

# Convert Specific Actor
python convert_actorshq_to_colmap.py --actor Actor01 --input /bigdata2/rajrup/datasets/actorshq/ --output /bigdata2/rajrup/datasets/actorshq/

# Convert Specific Actor and Sequence
python convert_actorshq_to_colmap.py --actor Actor01 --sequence Sequence1 --input /bigdata2/rajrup/datasets/actorshq/ --output /bigdata2/rajrup/datasets/actorshq/
'''