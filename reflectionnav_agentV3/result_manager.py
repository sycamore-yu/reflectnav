"""
Result Manager for ReflectionNav_AgentV3
Handles categorized saving of trajectory data, RGB images, and semantic images.
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from gsamllavanav.space import Pose4D
from gsamllavanav.dataset.episode import Episode
import logging

logger = logging.getLogger(__name__)

class ResultManager:
    """Manages the saving and organization of agent results."""
    
    def __init__(self, base_output_dir: str = "/home/tong/tongworkspace/geonav/reflectionnav_agentV3/results"):
        """
        Initialize the result manager.
        
        Args:
            base_output_dir: Base directory for saving results
        """
        self.base_output_dir = base_output_dir
        self.current_episode_dir = None
        self.rgb_dir = None
        self.semantic_dir = None
        self.trajectory_file = None
        
    def initialize_episode_directory(self, episode: Episode, agent_type: str = "ReflectionNavV3") -> str:
        """
        Create and initialize directory structure for an episode.
        
        Args:
            episode: Episode object
            agent_type: Type of agent for folder naming
            
        Returns:
            Path to the episode directory
        """
        # Create timestamped directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_id_str = "_".join(map(str, episode.id))
        episode_dir_name = f"{agent_type}_{episode_id_str}_{timestamp}"
        
        # Create full directory path
        self.current_episode_dir = os.path.join(self.base_output_dir, episode_dir_name)
        os.makedirs(self.current_episode_dir, exist_ok=True)
        
        # Create subdirectories
        self.rgb_dir = os.path.join(self.current_episode_dir, "rgb_images")
        self.semantic_dir = os.path.join(self.current_episode_dir, "semantic_images")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.semantic_dir, exist_ok=True)
        
        # Initialize trajectory file
        self.trajectory_file = os.path.join(self.current_episode_dir, "trajectory.json")
        
        # Save episode metadata
        self._save_episode_metadata(episode)
        
        logger.info(f"Initialized episode directory: {self.current_episode_dir}")
        return self.current_episode_dir
    
    def _save_episode_metadata(self, episode: Episode):
        """Save episode metadata to the episode directory."""
        metadata = {
            "episode_id": episode.id,
            "target_description": episode.target_description,
            "description_landmarks": episode.description_landmarks,
            "description_target": episode.description_target,
            "description_surroundings": episode.description_surroundings,
            "map_name": episode.map_name,
            "start_pose": {
                "x": episode.start_pose.x,
                "y": episode.start_pose.y,
                "z": episode.start_pose.z,
                "yaw": episode.start_pose.yaw
            },
            "target_position": {
                "x": episode.target_position.x,
                "y": episode.target_position.y,
                "z": episode.target_position.z
            },
            "created_at": datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.current_episode_dir, "episode_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def save_rgb_image(self, rgb_image: np.ndarray, timestep: int) -> str:
        """
        Save RGB image with timestep naming.
        
        Args:
            rgb_image: RGB image array or PIL Image
            timestep: Current timestep
            
        Returns:
            Path to saved image or empty string if failed
        """
        if self.rgb_dir is None:
            logger.error("Episode directory not initialized")
            return ""
        
        filename = f"rgb_timestep_{timestep:04d}.png"
        filepath = os.path.join(self.rgb_dir, filename)
        
        try:
            # Convert and save RGB image
            if isinstance(rgb_image, np.ndarray):
                pil_image = Image.fromarray(rgb_image)
                pil_image.save(filepath)
                logger.debug(f"✅ Saved numpy RGB image: {filepath}")
                return filepath
            elif isinstance(rgb_image, Image.Image):
                rgb_image.save(filepath)
                logger.debug(f"✅ Saved PIL RGB image: {filepath}")
                return filepath
            else:
                logger.error(f"❌ Unsupported RGB image type: {type(rgb_image)}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Failed to save RGB image: {e}")
            return ""
    
    def save_semantic_image(self, semantic_image, timestep: int, image_type: str = "semantic") -> str:
        """
        Save semantic enhanced image with timestep naming.
        
        Args:
            semantic_image: Semantic image (can be PIL Image, numpy array, or base64 string)
            timestep: Current timestep
            image_type: Type of semantic image (e.g., 'semantic', 'landmark', 'search')
            
        Returns:
            Path to saved image or empty string if failed
        """
        if self.semantic_dir is None:
            logger.error("Episode directory not initialized")
            return ""
        
        filename = f"{image_type}_timestep_{timestep:04d}.png"
        filepath = os.path.join(self.semantic_dir, filename)
        
        try:
            # Handle different image types
            if isinstance(semantic_image, str):
                # Assume it's a base64 string
                import base64
                from io import BytesIO
                try:
                    # Handle data URL format (data:image/png;base64,...)
                    if ',' in semantic_image:
                        semantic_image = semantic_image.split(',')[1]
                    
                    image_bytes = base64.b64decode(semantic_image)
                    image_buffer = BytesIO(image_bytes)
                    pil_image = Image.open(image_buffer)
                    pil_image.save(filepath)
                    logger.info(f"✅ Saved base64 semantic image: {filepath}")
                    return filepath
                except Exception as e:
                    logger.error(f"❌ Failed to decode and save base64 image: {e}")
                    return ""
                    
            elif isinstance(semantic_image, np.ndarray):
                # NumPy array
                try:
                    pil_image = Image.fromarray(semantic_image)
                    pil_image.save(filepath)
                    logger.info(f"✅ Saved numpy array semantic image: {filepath}")
                    return filepath
                except Exception as e:
                    logger.error(f"❌ Failed to save numpy array image: {e}")
                    return ""
                    
            elif isinstance(semantic_image, Image.Image):
                # PIL Image
                try:
                    semantic_image.save(filepath)
                    logger.info(f"✅ Saved PIL semantic image: {filepath}")
                    return filepath
                except Exception as e:
                    logger.error(f"❌ Failed to save PIL image: {e}")
                    return ""
            else:
                logger.warning(f"⚠️  Unsupported semantic image type: {type(semantic_image)}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Unexpected error saving semantic image: {e}")
            return ""
    
    def add_trajectory_step(self, step_data: Dict[str, Any]):
        """
        Add a step to the trajectory log.
        
        Args:
            step_data: Dictionary containing step information
        """
        if self.trajectory_file is None:
            raise ValueError("Episode directory not initialized")
        
        # Load existing trajectory or create new one
        if os.path.exists(self.trajectory_file):
            with open(self.trajectory_file, 'r') as f:
                trajectory = json.load(f)
        else:
            trajectory = {
                "metadata": {
                    "agent_type": "ReflectionNavV3",
                    "framework": "LangGraph + LangChain",
                    "created_at": datetime.now().isoformat()
                },
                "steps": []
            }
        
        # Add the new step
        trajectory["steps"].append(step_data)
        
        # Save updated trajectory
        with open(self.trajectory_file, 'w') as f:
            json.dump(trajectory, f, indent=4)
    
    def finalize_episode(self, success: bool, total_steps: int, final_distance: float):
        """
        Finalize the episode by saving summary information.
        
        Args:
            success: Whether the mission was successful
            total_steps: Total number of steps taken
            final_distance: Final distance to target
        """
        if self.current_episode_dir is None:
            raise ValueError("Episode directory not initialized")
        
        summary = {
            "mission_success": success,
            "total_steps": total_steps,
            "final_distance_to_target": final_distance,
            "completed_at": datetime.now().isoformat(),
            "rgb_images_count": len([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')]) if self.rgb_dir else 0,
            "semantic_images_count": len([f for f in os.listdir(self.semantic_dir) if f.endswith('.png')]) if self.semantic_dir else 0
        }
        
        summary_file = os.path.join(self.current_episode_dir, "episode_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Episode finalized. Success: {success}, Steps: {total_steps}, Directory: {self.current_episode_dir}")
    
    def get_episode_directory(self) -> Optional[str]:
        """Get the current episode directory path."""
        return self.current_episode_dir
    
    def cleanup_incomplete_episode(self):
        """Clean up directory if episode was incomplete."""
        if self.current_episode_dir and os.path.exists(self.current_episode_dir):
            try:
                shutil.rmtree(self.current_episode_dir)
                logger.info(f"Cleaned up incomplete episode directory: {self.current_episode_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup episode directory: {e}")
        
        # Reset paths
        self.current_episode_dir = None
        self.rgb_dir = None
        self.semantic_dir = None
        self.trajectory_file = None 