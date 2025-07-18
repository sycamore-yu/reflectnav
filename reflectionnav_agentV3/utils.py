"""
Utility functions for ReflectionNav_AgentV3.
"""

import base64
import logging
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Optional, Union, Any
import os

logger = logging.getLogger(__name__)

def encode_image_from_pil(image: Image.Image) -> Optional[str]:
    """
    Encode a PIL image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string or None if error
    """
    try:
        if image is None:
            return None
            
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Encode to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return encoded_string
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

def decode_image_from_base64(encoded_string: str) -> Optional[Image.Image]:
    """
    Decode a base64 string back to PIL Image.
    
    Args:
        encoded_string: Base64 encoded image string
        
    Returns:
        PIL Image object or None if error
    """
    try:
        if not encoded_string:
            return None
            
        # Decode from base64
        image_bytes = base64.b64decode(encoded_string)
        image_buffer = BytesIO(image_bytes)
        image = Image.open(image_buffer)
        
        return image
        
    except Exception as e:
        logger.error(f"Error decoding image from base64: {e}")
        return None

def encode_numpy_image(image_array: np.ndarray) -> Optional[str]:
    """
    Encode a numpy array image to base64 string.
    
    Args:
        image_array: Numpy array representing an image
        
    Returns:
        Base64 encoded string or None if error
    """
    try:
        if image_array is None:
            return None
            
        # Convert to PIL Image first
        pil_image = Image.fromarray(image_array)
        return encode_image_from_pil(pil_image)
        
    except Exception as e:
        logger.error(f"Error encoding numpy image: {e}")
        return None

def safe_json_dumps(obj, indent=None):
    """
    Safely dump object to JSON string, handling numpy types.
    
    Args:
        obj: Object to serialize
        indent: JSON indent level
        
    Returns:
        JSON string
    """
    try:
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: safe_json_dumps(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json_dumps(item) for item in obj]
        else:
            return obj
            
    except Exception as e:
        logger.error(f"Error in safe_json_dumps: {e}")
        return str(obj)

def validate_pose(pose) -> bool:
    """
    Validate that a pose object has required attributes.
    
    Args:
        pose: Pose object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return hasattr(pose, 'x') and hasattr(pose, 'y') and hasattr(pose, 'z') and hasattr(pose, 'yaw')
    except Exception:
        return False

def validate_episode(episode) -> bool:
    """
    Validate that an episode object has required attributes.
    
    Args:
        episode: Episode object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        required_attrs = ['id', 'target_description', 'target_position', 'start_pose', 'map_name']
        return all(hasattr(episode, attr) for attr in required_attrs)
    except Exception:
        return False

def safe_distance_calculation(pos1, pos2) -> float:
    """
    Safely calculate distance between two positions.
    
    Args:
        pos1: First position (should have x, y attributes)
        pos2: Second position (should have x, y attributes)
        
    Returns:
        Distance between positions or large number if error
    """
    try:
        if hasattr(pos1, 'dist_to') and hasattr(pos2, 'xy'):
            return pos1.dist_to(pos2.xy)
        elif hasattr(pos1, 'xy') and hasattr(pos2, 'xy'):
            return pos1.xy.dist_to(pos2.xy)
        else:
            # Manual calculation
            dx = pos1.x - pos2.x
            dy = pos1.y - pos2.y
            return (dx**2 + dy**2)**0.5
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return 1000.0  # Large number to indicate error

def format_step_info(step_num: int, total_steps: int, current_goal: str, strategy: str) -> str:
    """
    Format step information for logging.
    
    Args:
        step_num: Current step number
        total_steps: Total number of steps
        current_goal: Current goal description
        strategy: Current strategy
        
    Returns:
        Formatted string
    """
    return f"Step [{step_num}/{total_steps}] - {strategy}: {current_goal}"

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value)) 

def convert_to_pil_image(image_data: Any) -> Optional[Image.Image]:
    """
    Convert various image formats to PIL.Image
    
    Args:
        image_data: Can be PIL.Image, numpy.ndarray, base64 string, or other formats
        
    Returns:
        PIL.Image object or None if conversion fails
    """
    try:
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image
            if image_data.dtype != np.uint8:
                # Normalize to 0-255 if not already uint8
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            
            if len(image_data.shape) == 2:
                # Grayscale
                return Image.fromarray(image_data, mode='L')
            elif len(image_data.shape) == 3:
                if image_data.shape[2] == 3:
                    # RGB
                    return Image.fromarray(image_data, mode='RGB')
                elif image_data.shape[2] == 4:
                    # RGBA
                    return Image.fromarray(image_data, mode='RGBA')
            return Image.fromarray(image_data)
            
        elif isinstance(image_data, str):
            # Assume it's a base64 string
            try:
                # Handle data URL format
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image_buffer = BytesIO(image_bytes)
                return Image.open(image_buffer)
            except Exception:
                # If base64 decoding fails, might be a file path
                if os.path.exists(image_data):
                    return Image.open(image_data)
                return None
        else:
            logging.warning(f"Unsupported image data type: {type(image_data)}")
            return None
            
    except Exception as e:
        logging.error(f"Failed to convert image data to PIL.Image: {e}")
        return None

def safe_image_save(image_data: Any, filepath: str) -> bool:
    """
    Safely save image data to file, handling various input formats
    
    Args:
        image_data: Image data in various formats
        filepath: Path to save the image
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        pil_image = convert_to_pil_image(image_data)
        if pil_image is not None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            pil_image.save(filepath)
            return True
        else:
            logging.error(f"Failed to convert image data for saving to {filepath}")
            return False
    except Exception as e:
        logging.error(f"Failed to save image to {filepath}: {e}")
        return False

def decode_base64_to_image(b64_string: Optional[str]) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image"""
    if not b64_string:
        return None
    try:
        # Handle data URL format
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        image_bytes = base64.b64decode(b64_string)
        image_buffer = BytesIO(image_bytes)
        return Image.open(image_buffer)
    except Exception as e:
        logging.error(f"Failed to decode base64 to image: {e}")
        return None

def encode_image_to_base64(image: Optional[Union[Image.Image, np.ndarray]]) -> Optional[str]:
    """Encode PIL Image or numpy array to base64 string"""
    if image is None:
        return None
    
    try:
        # 处理numpy数组
        if isinstance(image, np.ndarray):
            # 转换numpy数组为PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB图像
                pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
            elif len(image.shape) == 2:
                # 灰度图像
                pil_image = Image.fromarray(image.astype(np.uint8), 'L')
            else:
                logger.warning(f"Unexpected numpy array shape: {image.shape}, converting to RGB")
                pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            # 已经是PIL Image
            pil_image = image
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None
        
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        return None 

def convert_numpy_for_json(obj):
    """
    将包含numpy数组和其他不可序列化对象的Python对象转换为JSON可序列化格式。
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的JSON可序列化对象
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # 处理自定义对象，转换为字典
        return convert_numpy_for_json(obj.__dict__)
    else:
        return obj


def safe_json_dumps(obj):
    """
    安全地将对象序列化为JSON字符串，自动处理numpy数组等不可序列化对象。
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        JSON字符串
    """
    import json
    safe_obj = convert_numpy_for_json(obj)
    return json.dumps(safe_obj) 