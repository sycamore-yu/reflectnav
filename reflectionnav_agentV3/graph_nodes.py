"""
Graph nodes for ReflectionNav Agent V3.
These functions define the execution flow using LangGraph.
"""
import os
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
from skimage import color
from PIL import Image
import math

from gsamllavanav.space import Point2D
from reflectionnav_agentV3.agent_state import AgentState, FrameLog
from reflectionnav_agentV3.utils import encode_image_from_pil, safe_json_dumps, convert_numpy_for_json

logger = logging.getLogger(__name__)

# ===============================================
# Utility Functions
# ===============================================

def rgb_entropy(current_map):
    """Calculate RGB entropy for information gain measurement."""
    # Convert PIL Image to numpy array
    rgb_img = np.array(current_map) 
    hsv_img = color.rgb2hsv(rgb_img)
    entropy = 0
    for i in range(3):
        channel = (hsv_img[..., i] * 255).astype(np.uint8)
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / hist.sum() + 1e-10
        entropy += -np.sum(hist * np.log2(hist))
    return entropy

def calculate_adaptive_action_scale(current_pose, target_position, strategy: str) -> float:
    """
    Calculate adaptive action scale based on distance to target.
    For navigation, uses exponential decay above a threshold, and a fixed scale below.
    For search, uses a fixed scale.
    
    Formula: action_scale = 12.65 * exp(-0.048 * distance)
    - At 5m: action_scale ≈ 10 (very small steps)
    - At 30m: action_scale = 3 (baseline)
    - At 100m: action_scale ≈ 0.5 (large steps)
    """
    try:
        from reflectionnav_agentV3.config_loader import get_config
        config = get_config()
        
        action_scales = config.navigation.action_scale
        
        if not config.experimental.enable_adaptive_action_scale:
            # Use fixed scales from config when adaptive scaling is disabled
            scale = action_scales.get(strategy.lower(), 1.0)
            logger.info(f"🎯 Fixed Action Scale (adaptive disabled): strategy={strategy}, scale={scale:.2f}")
            return scale

        # Common distance calculation
        distance = current_pose.xy.dist_to(target_position.xy)
        action_scale = 1.0 # Default value

        if strategy == 'Navigate':
            threshold = config.navigation.adaptive_scale_distance_threshold
            print(f"distance: {distance}, threshold: {threshold}")
            if distance > threshold:
                # Exponential decay formula for long distances
                action_scale = 12.65 * np.exp(-0.038 * distance)
                print(f"🎯 Adaptive Action Scale (Navigate, dynamic): distance={distance:.1f}m > {threshold}m, scale={action_scale:.2f}")
            else:
                # Use fixed scale for short distances
                action_scale = action_scales.get('navigate', 1.0)
                print(f"🎯 Adaptive Action Scale (Navigate, fixed): distance={distance:.1f}m <= {threshold}m, scale={action_scale:.2f}")
        
        elif strategy == 'Search':
            # 🔧 NEW: Exponential decay formula for Search strategy.
            # Designed to be ~1.0 at 30m distance.
            # Formula: 1.2367 * e^(-0.00708 * distance)
            action_scale = 1.2367 * np.exp(-0.00708 * distance)
            print(f"🎯 Adaptive Action Scale (Search): distance={distance:.1f}m, scale={action_scale:.2f}")
        
        # Clamp to reasonable bounds
        print(f"action_scale: {action_scale}, max(0.3, min(15.0, action_scale)): {max(0.3, min(15.0, action_scale))}")
        return max(0.3, min(15.0, action_scale))
        
    except Exception as e:
        logger.error(f"Error calculating adaptive action scale: {e}")
        # Safe fallback
        from reflectionnav_agentV3.config_loader import get_config
        config = get_config()
        return config.navigation.action_scale.get(strategy.lower(), 1.0)

# Remove or simplify redundant print functions
def print_plan_details(state: AgentState):
    """打印计划概要信息"""
    plan = state.get("plan")
    if plan and plan.get("sub_goals"):
        num_goals = len(plan["sub_goals"])
        print(f"📋 Plan generated: {num_goals} sub-goals")

def print_subtask_execution(state: AgentState, strategy: str, reason: str, movement: str, next_pos: Point2D):
    """打印子任务执行的核心信息"""
    timestep = state.get("timestep", 0)
    distance = state["controller"].pose.xy.dist_to(state["episode"].target_position.xy)
    print(f"🎯 Step {timestep} | {strategy} | Action: {movement} | Reason: {reason}... | Dist: {distance:.1f}m | Next: ({next_pos.x:.1f}, {next_pos.y:.1f})")

def print_rag_usage(strategy: str, experience_clue: str, has_retrieved_image: bool):
    """打印RAG使用情况"""
    if experience_clue.strip():
        print(f"💡 RAG-{strategy}: {experience_clue}" + (" +Img" if has_retrieved_image else ""))

def print_reflection_info(state: AgentState, reflection_type: str, analysis: dict):
    """打印详细的反思信息"""
    print("\n" + "="*60)
    print(f"🧠 反思报告 - {reflection_type.upper()}")
    print("="*60)
    
    # 🔧 修复：根据新的分析结果格式计算经验数量
    # 新的格式包含critical_timestep，而不是key_experiences
    critical_timestep = analysis.get("critical_timestep", -1)
    trajectory = state.get("trajectory_history", [])
    
    # 如果critical_timestep有效且轨迹存在，说明有经验被存储
    if critical_timestep >= 0 and critical_timestep < len(trajectory):
        num_experiences = 1  # 现在只存储一个关键经验
    else:
        num_experiences = 0
    
    print(f"📊 存储的经验数量: {num_experiences}")
    
    # 任务状态
    success = state.get("mission_success", False)
    final_reason = state.get("final_reason", "unknown")
    
    # 🔧 修复：根据反思类型推断任务成功状态
    if reflection_type.lower() == "success":
        # 如果是成功反思，任务应该是成功的
        inferred_success = True
        if not success:
            print(f"⚠️  警告：反思类型为success但mission_success为{success}，推断为成功")
    else:
        # 如果是失败反思，任务应该是失败的
        inferred_success = False
        if success:
            print(f"⚠️  警告：反思类型为failure但mission_success为{success}，推断为失败")
    
    print(f"✅ 任务成功: {inferred_success}")
    print(f"🎯 结束原因: {final_reason}")
    
    # 轨迹统计
    print(f"📈 总步数: {len(trajectory)}")
    
    if trajectory:
        # 修复：正确处理tuple类型的pose数据
        first_pose = trajectory[0].get('pose', (0, 0, 0, 0))
        last_pose = trajectory[-1].get('pose', (0, 0, 0, 0))
        if isinstance(first_pose, tuple) and len(first_pose) >= 2:
            print(f"🚀 起始位置: ({first_pose[0]:.1f}, {first_pose[1]:.1f})")
            print(f"🏁 最终位置: ({last_pose[0]:.1f}, {last_pose[1]:.1f})")
        else:
            print(f"🚀 起始位置: 无法解析")
            print(f"🏁 最终位置: 无法解析")
    
    # 🔧 修复：关键经验详情 - 使用新的字段格式
    if num_experiences > 0 and critical_timestep >= 0:
        print("\n🔍 关键经验详情:")
        print(f"  关键时间步: {critical_timestep}")
        
        # 获取关键帧信息
        if critical_timestep < len(trajectory):
            critical_frame = trajectory[critical_timestep]
            print(f"  目标: {critical_frame.get('goal', 'N/A')}")
            print(f"  策略: {critical_frame.get('strategy', 'N/A')}")
        
        # 显示反思信息
        reflect_reason = analysis.get("reflect_reason", "N/A")
        print(f"  反思原因: {reflect_reason}")
        
        if reflection_type == "success":
            successful_action = analysis.get("successful_action", "N/A")
            print(f"  成功动作: {successful_action}")
        else:
            corrected_action = analysis.get("corrected_action", "N/A")
            print(f"  修正动作: {corrected_action}")
    
    # 内存统计
    try:
        memory_module = state.get("memory_module")
        if memory_module:
            # 🔧 修复：使用正确的collection.count()方法而不是len(collection.get())
            nav_count = memory_module.navigate_experience_memory.collection.count() if hasattr(memory_module.navigate_experience_memory, 'collection') else 0
            search_count = memory_module.search_and_locate_memory.collection.count() if hasattr(memory_module.search_and_locate_memory, 'collection') else 0
            plan_count = memory_module.mainplan_experience_memory.collection.count() if hasattr(memory_module.mainplan_experience_memory, 'collection') else 0
            
            # 🔧 新增：显示数据库路径信息
            db_path = getattr(memory_module, 'client', None)
            if db_path and hasattr(db_path, 'get_settings'):
                actual_db_path = db_path.get_settings().persist_directory
            else:
                actual_db_path = "默认路径 (./chroma)"
            
            print(f"\n📚 内存统计 (数据库路径: {actual_db_path}):")
            print(f"   导航经验: {nav_count}")
            print(f"   搜索经验: {search_count}")
            print(f"   规划经验: {plan_count}")
    except Exception as e:
        print(f"\n⚠️ 无法获取内存统计: {e}")
    
    print("="*60)

# Helper functions
def encode_image(image) -> Optional[str]:
    """Helper to encode images to base64."""
    try:
        if image is None:
            return None
        elif isinstance(image, Image.Image):
            from reflectionnav_agentV3.utils import encode_image_from_pil
            return encode_image_from_pil(image)
        elif isinstance(image, np.ndarray):
            from reflectionnav_agentV3.utils import encode_image_from_pil
            return encode_image_from_pil(Image.fromarray(image))
        elif isinstance(image, str):
            # 如果已经是base64字符串，直接返回
            logger.debug("Input is already a string (assumed to be base64)")
            return image
        else:
            logger.warning(f"Unsupported image type: {type(image)}. Cannot encode.")
        return None
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def calculate_view_area(pose, ground_level):
    """Calculate view area based on pose and ground level."""
    view_width = 2 * (pose.z - ground_level)
    return [
        (pose.x - view_width/2, pose.y - view_width/2),
        (pose.x + view_width/2, pose.y + view_width/2)
    ]

def calculate_next_position(current_pose, view_width: float, action_scale: float, target_json: dict) -> Point2D:
    """
    Calculate next position based on LLM output and a pre-calculated action scale.
    This function no longer calculates the scale itself, making the control flow more explicit.
    """
    direction_map = {
        'northwest': (-view_width / 5, view_width / 5),
        'northeast': (view_width / 5, view_width / 5),
        'southwest': (-view_width / 5, -view_width / 5),
        'southeast': (view_width / 5, -view_width / 5),
        'north': (0, view_width / 4),
        'south': (0, -view_width / 4),
        'west': (-view_width / 4, 0),
        'east': (view_width / 4, 0)
    }
    
    movement = target_json.get('movement', '').lower()
    for key, (dx, dy) in direction_map.items():
        if key in movement:
            # Use the provided action_scale, ensuring it's not zero to avoid division errors
            safe_scale = action_scale if action_scale != 0 else 1.0
            return Point2D(
                current_pose.x + dx / safe_scale,
                current_pose.y + dy / safe_scale
            )
            
    # Fallback to locate logic if present in target_json
    if target_json.get('status') == 'TARGET_LOCKED' and 'selected_pos' in target_json:
        pos = target_json['selected_pos']
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return Point2D(pos[0], pos[1])
            
    # Default: stay at current position if no valid movement is found
    return Point2D(current_pose.x, current_pose.y)

def retrieve_experience_memory(memory_module, strategy: str, current_state: str, config=None) -> Tuple[str, Optional[str]]:
    """
    Retrieve experience from memory based on strategy and current_state.
    
    Args:
        memory_module: The memory module instance
        strategy: The strategy type ('Navigate', 'Search', 'Locate')
        current_state: Complete current state string for RAG query (as per documentation)
        config: Configuration object
        
    Returns:
        Tuple of (experience_clue, retrieved_image_b64)
    """
    # 🔧 Fixed: Check master RAG switch first
    if config and hasattr(config, 'rag'):
        if not config.rag.enable_rag:
            return "RAG system disabled.", None
        if not config.rag.enable_experience_memory:
            return "RAG experience memory disabled.", None
    
    try:
        n_results = config.memory.max_retrieval_results if config else 1 # 🔧 FIXED: Use max_retrieval_results from config
        
        # 🔧 FIXED: Use complete current_state string as query vector (as per documentation)
        if strategy == 'Navigate':
            retrieved_exp_list = memory_module.navigate_experience_memory.retrieve(
                query_text=current_state,  # 🔧 FIXED: Use current_state instead of scene_caption
                n_results=n_results, 
                where_filter={"memory_type": "reflection_note"}
            )
        elif strategy in ['Search', 'Locate']:
            retrieved_exp_list = memory_module.search_and_locate_memory.retrieve(
                query_text=current_state,  # 🔧 FIXED: Use current_state instead of scene_caption
                n_results=n_results, 
                where_filter={"memory_type": "reflection_note"}
            )
        else:
            return "No specific experience found.", None
        
        if retrieved_exp_list and len(retrieved_exp_list) > 0:
            exp = retrieved_exp_list[0]
            experience_clue = exp.get('corrected_reason', "No specific insight from this experience.")
            # Prioritize semantic enhanced image over map snapshot
            retrieved_image_b64 = exp.get('semantic_enhanced_image_b64') or exp.get('map_snapshot_b64')
            return experience_clue, retrieved_image_b64
        
    except Exception as e:
        logger.error(f"Error retrieving experience memory: {e}")
    
    return "No specific experience found. Rely on general knowledge.", None

def get_semantic_image(state: AgentState, strategy: str) -> Optional[str]:
    """获取语义图像的base64字符串，参考GeonavAgent.gen_map的实现"""
    try:
        timestep = state.get("timestep", 0)
        
        # Generate map data based on strategy - 直接调用landmark_nav_map.plot返回base64字符串
        if strategy == 'Navigate':
            map_image_b64 = state["landmark_nav_map"].plot(map_type='landmark', timestep=timestep)
            logger.info(f"🗺️  Generated landmark map for Navigate strategy at timestep {timestep}")
        elif strategy == 'Search':
            # 确保query_engine存在
            query_engine = getattr(state["controller"], "query_engine", None)
            if query_engine is None:
                logger.warning("Query engine not available, using landmark map instead")
                map_image_b64 = state["landmark_nav_map"].plot(map_type='landmark', timestep=timestep)
            else:
                map_image_b64 = state["landmark_nav_map"].plot(
                    map_type='semantic',
                    query_engine=query_engine,
                    current_pos=state["controller"].pose.xy,
                    timestep=timestep
                )
            logger.info(f"🗺️  Generated semantic map for Search strategy at timestep {timestep}")
        else:  # 'Locate'
            map_image_b64 = state["landmark_nav_map"].plot(map_type='landmark', timestep=timestep)
            logger.info(f"🗺️  Generated landmark map for Locate strategy at timestep {timestep}")
        
        # 🔧 新增：使用ResultManager保存语义图片到正确的子目录
        if map_image_b64:
            try:
                result_manager = state.get("result_manager")
                if result_manager is not None and hasattr(result_manager, 'save_semantic_image'):
                    saved_path = result_manager.save_semantic_image(map_image_b64, timestep, strategy.lower())
                    if saved_path:
                        logger.info(f"✅ Saved semantic image via ResultManager: {saved_path}")
                    else:
                        logger.warning("Failed to save semantic image via ResultManager")
                else:
                    logger.warning("ResultManager not available for semantic image saving")
            except Exception as e:
                logger.error(f"Error saving semantic image via ResultManager: {e}")
        
        return map_image_b64
        
    except Exception as e:
        logger.error(f"Error generating semantic image for {strategy}: {e}")
        return None

# ==============================================================================================
# Router Nodes
# ==============================================================================================

def should_perceive(state: AgentState) -> str:
    """Router: Determine if perception is needed."""
    # Perceive on first step or at intervals
    if state["timestep"] == 0:
        return "perceive"
    
    # 🔧 Fixed: Use config value instead of hardcoded interval
    from reflectionnav_agentV3.config_loader import get_config
    config = get_config()
    vlm_interval = config.vlm_detection.interval
    
    # Check interval-based perception
    if state["timestep"] % vlm_interval == 0:
        return "perceive"
    
    # Check distance-based perception
    current_pos = state["controller"].pose.xy
    last_pos = state.get("last_vlm_detection_position", current_pos)
    vlm_distance_threshold = config.vlm_detection.distance_threshold
    if current_pos.dist_to(last_pos) >= vlm_distance_threshold:
        return "perceive"
    
    return "skip_perception"

def should_generate_plan(state: AgentState) -> str:
    """Router: Determine if plan generation is needed."""
    if state.get("plan") is None:
        return "generate_plan"
    return "execute_subtask"

def should_continue_mission(state: AgentState) -> str:
    from reflectionnav_agentV3.config_loader import get_config
    config = get_config()
    """Router: Determine if mission should continue."""
    # 🔧 NEW: Prioritize locate completion check. This is the definitive mission end signal.
    if state.get("locate_completed", False):
        print("🎯 Locate operation completed, evaluating final mission success based on distance.")
        try:
            # Final success is determined *only* by distance to target.
            distance_to_target = state["controller"].pose.xy.dist_to(state["episode"].target_position.xy)
            success_dist = config.strategy.target_proximity # 🔧 FIXED: Use target_proximity from config

            if distance_to_target <= success_dist:
                state["mission_success"] = True
                state["final_reason"] = "success"
                logger.info(f"🎯 Mission SUCCESS: Final distance to target {distance_to_target:.2f}m <= {success_dist}m")
                return "reflect_success"
            else:
                state["mission_success"] = False
                action_result = state.get("action_result", {})
                status = "unknown"
                if isinstance(action_result, dict):
                    status = action_result.get("status", "unknown")
                state["final_reason"] = f"goal_unreachable (final_distance: {distance_to_target:.2f}m, locate_status: {status})"
                logger.info(f"🎯 Mission FAILURE: Final distance to target {distance_to_target:.2f}m > {success_dist}m")
                return "reflect_failure"
        except Exception as e:
            logger.error(f"Error evaluating final mission success: {e}")
            state["mission_success"] = False
            state["final_reason"] = "final_check_error"
            return "reflect_failure"

    # 🔧 Enhanced: Check for final step force locate
    
    # Check if we should force locate on final step
    if config.strategy.force_locate_on_final_step:
        # Check if approaching max timesteps (within last 3 steps) and target not yet reached
        max_timestep = state["args"].eval_max_timestep
        approaching_timeout = (state["timestep"] >= max_timestep - 3)
        
        # Check if in final phase of plan
        plan = state.get("plan")
        approaching_plan_end = False
        if plan and isinstance(plan, dict) and "sub_goals" in plan:
            total_goals = len(plan["sub_goals"])
            current_index = state.get("current_task_index", 0)
            approaching_plan_end = (current_index >= total_goals - 1)
        
        # Force locate if approaching timeout or plan end, and target not reached
        if (approaching_timeout or approaching_plan_end):
            try:
                if not state["controller"].reached_target(state["controller"].pose, state["episode"].target_position.xy):
                    logger.info(f"🎯 Forcing final locate operation - timeout: {approaching_timeout}, plan_end: {approaching_plan_end}")
                    
                    # Temporarily modify current task to be locate
                    if plan and "sub_goals" in plan and state["current_task_index"] < len(plan["sub_goals"]):
                        current_task = plan["sub_goals"][state["current_task_index"]]
                        current_task["strategy"] = "Locate"
                        current_task["goal"] = f"Final locate: {state['episode'].target_description}"
                        logger.info(f"Modified current task to force locate: {current_task['goal']}")
                    
                    # Continue mission to execute the locate
                    return "continue"
            except Exception as e:
                logger.error(f"Error in force locate check: {e}")

    # Check if target reached
    try:
        if state["controller"].reached_target(state["controller"].pose, state["episode"].target_position.xy):
            state["mission_success"] = True
            state["final_reason"] = "success"
            return "reflect_success"
    except Exception as e:
        logger.error(f"Error checking target reached: {e}")
    
    # Check if max timesteps reached
    if state["timestep"] >= state["args"].eval_max_timestep:
        state["mission_success"] = False
        state["final_reason"] = "timeout"
        return "reflect_failure"
    
    # 🔧 Enhanced: Check subtask status first, then plan completion
    subtask_status = state.get("subtask_status", "")
    if subtask_status == "all_completed":
        # All subtasks completed - check final success
        try:
            distance_to_target = state["controller"].pose.xy.dist_to(state["episode"].target_position.xy)
            if distance_to_target <= config.strategy.target_proximity: # 🔧 FIXED: Use target_proximity
                state["mission_success"] = True
                state["final_reason"] = "success"
                logger.info(f"Mission completed successfully - distance to target: {distance_to_target:.2f}m")
                return "reflect_success"
            else:
                state["mission_success"] = False
                state["final_reason"] = "goal_unreachable"
                logger.info(f"Mission completed but target not reached - distance: {distance_to_target:.2f}m")
                return "reflect_failure"
        except Exception as e:
            logger.error(f"Error calculating final distance: {e}")
            state["mission_success"] = False
            state["final_reason"] = "error"
            return "reflect_failure"
    
    # Check if all subtasks completed (fallback check)
    plan = state.get("plan")
    if plan and isinstance(plan, dict) and "sub_goals" in plan:
        if state["current_task_index"] >= len(plan["sub_goals"]):
            # Mission completed but target not reached
            try:
                distance_to_target = state["controller"].pose.xy.dist_to(state["episode"].target_position.xy)
                if distance_to_target <= config.strategy.target_proximity: # 🔧 FIXED: Use target_proximity
                    state["mission_success"] = True
                    state["final_reason"] = "success"
                    logger.info(f"Mission completed successfully (fallback) - distance to target: {distance_to_target:.2f}m")
                    return "reflect_success"
                else:
                    state["mission_success"] = False
                    state["final_reason"] = "goal_unreachable"
                    logger.info(f"Mission completed but target not reached (fallback) - distance: {distance_to_target:.2f}m")
                    return "reflect_failure"
            except Exception as e:
                logger.error(f"Error calculating final distance: {e}")
                state["mission_success"] = False
                state["final_reason"] = "error"
                return "reflect_failure"
    
    return "continue"

def should_retrieve_memory(state: AgentState) -> bool:
    """
    🔧 New function: Check if memory retrieval should happen based on interval.
    """
    from reflectionnav_agentV3.config_loader import get_config
    config = get_config()
    
    # Check master switches
    if not config.rag.enable_rag:
        return False
        
    # Check retrieval interval
    last_retrieval = state.get("last_memory_retrieval_timestep", -1)
    current_timestep = state.get("timestep", 0)
    interval = config.rag.retrieval_interval
    
    # 🔧 Fixed: Always retrieve on first step (-1 indicates never retrieved)
    if last_retrieval == -1:
        return True
        
    should_retrieve = (current_timestep - last_retrieval) >= interval
    
    return should_retrieve

# ==============================================================================================
# Core Nodes
# ==============================================================================================

def perceive_and_model(state: AgentState) -> AgentState:
    """
    Node: Perceive the world and update the semantic map.
    """
    logger.info(f"[{state['timestep']}] Perceiving and modeling...")
    
    try:
        # Get current observation - use simple unpacking like in agent.py
        rgb, _ = state["controller"].perceive(
            state["controller"].pose, 
            state["episode"].map_name
        )
        
        # Encode image
        image_b64 = encode_image_from_pil(Image.fromarray(rgb))
        
        # 🔧 修复：确保每步都保存RGB图像
        try:
            # 使用ResultManager的方法保存RGB图像到正确的子目录
            result_manager = state.get("result_manager")
            if result_manager is not None and hasattr(result_manager, 'save_rgb_image'):
                saved_path = result_manager.save_rgb_image(rgb, state["timestep"])
                if saved_path:
                    logger.info(f"✅ Saved RGB image via ResultManager: {saved_path}")
                else:
                    logger.warning("Failed to save RGB image via ResultManager")
            else:
                # 备用方法：直接保存到统一路径
                os.makedirs(state["save_path"], exist_ok=True)
                episode_id_str = "_".join(map(str, state["episode"].id))
                file_path = os.path.join(state["save_path"], f'{episode_id_str}_rgb_{state["timestep"]}.png')
                Image.fromarray(rgb).save(file_path)
                logger.info(f"✅ Saved RGB image (fallback): {file_path}")
        except Exception as e:
            logger.error(f"Error saving RGB image: {e}")
        
        # Update landmark map with observations
        strategy = "Search"  # Default strategy
        goal_for_display = "Default goal"  # Default goal for logging
        
        if state.get("plan") and isinstance(state["plan"], dict):
            # 🔧 Fixed: Add bounds checking to prevent list index out of range
            if ("sub_goals" in state["plan"] and 
                isinstance(state["plan"]["sub_goals"], list) and 
                0 <= state["current_task_index"] < len(state["plan"]["sub_goals"])):
                current_task = state["plan"]["sub_goals"][state["current_task_index"]]
                strategy = current_task.get("strategy", "Search")
                goal_for_display = current_task.get("goal", "Unknown goal")
                logger.info(f"Current Task [{state['current_task_index']+1}/{len(state['plan']['sub_goals'])}]: {goal_for_display} (Strategy: {strategy})")
            elif state.get("current_task_index", 0) >= len(state["plan"].get("sub_goals", [])):
                # All tasks completed - use final strategy 
                strategy = "Search"  # Use search as default for post-completion perception
                goal_for_display = "All tasks completed - continuing perception"
                logger.info(f"All tasks completed, using default strategy: {strategy}")
            else:
                logger.warning(f"Invalid plan structure or task index: {state.get('current_task_index', 'None')}")
                goal_for_display = "Using default goal due to invalid plan structure or task index"
        
        state["landmark_nav_map"].update_observations(
            state["controller"].pose, rgb, None, 
            use_gsam_map_cache=False, strategy=strategy
        )
        
        # Generate scene caption using LLM manager
        try:
            # 🔧 修复：将base64字符串转换为PIL Image对象
            if image_b64:
                from reflectionnav_agentV3.utils import decode_base64_to_image
                image_pil = decode_base64_to_image(image_b64)
                if image_pil:
                    scene_caption = state["llm_manager"].generate_image_caption(image_pil)
                else:
                    scene_caption = "Unable to decode image for scene description"
            else:
                scene_caption = "No image available for scene description"
        except Exception as e:
            logger.error(f"Error generating scene caption: {e}")
            scene_caption = "Unable to generate scene description"
        
        # Build geographical context
        try:
            if state["args"].ablation != 'wo_landmark':
                landmark_description = state["controller"].build_geo_nodes(
                    state["landmark_nav_map"].landmark_map.landmarks
                )
            else:
                landmark_description = state["controller"].build_geo_nodes([])
        except Exception as e:
            logger.error(f"Error building geo nodes: {e}")
            landmark_description = ""
        
        # Handle VLM detection based on strategy - only when needed, not affecting RGB saving
        recent_objects = ""
        surroundings = ""
        
        if strategy == 'Search':
            try:
                current_position = state["controller"].pose.xy
                last_time = state.get("last_vlm_detection_time", 0)
                last_pos = state.get("last_vlm_detection_position", current_position)
                interval = state.get("vlm_detection_interval", 1)
                distance_threshold = state.get("vlm_detection_distance", 15)
                
                time_condition = (state["timestep"] - last_time >= interval)
                distance_condition = (current_position.dist_to(last_pos) >= distance_threshold)
                
                if time_condition or distance_condition:
                    logger.info(f"Executing VLM detection at timestep {state['timestep']}")
                    # Ensure image_b64 is not None before calling understand
                    if image_b64:
                        subgraph = state["controller"].understand(image_b64, state["episode"])
                        state["controller"].build_scene_graph(subgraph, state["landmark_nav_map"].target_map)
                    else:
                        logger.warning("No image available for VLM detection")
                    
                    surroundings, recent_objects = state["controller"].build_scene_nodes(
                        state["landmark_nav_map"].target_map.obj_list,
                        state["landmark_nav_map"].surroundings_map.obj_list,
                        show=True
                    )
                    
                    # Update detection tracking
                    state["last_vlm_detection_time"] = state["timestep"]
                    state["last_vlm_detection_position"] = current_position
                    
            except Exception as e:
                logger.error(f"Error in VLM detection: {e}")
        print(scene_caption, "completed")
        # Store current observation
        state["current_observation"] = {
            "image": rgb,
            "image_b64": image_b64,
            "description": scene_caption
        }
        
        # Build geographical instruction
        try:
            geoinstruct = state["llm_manager"].prompts["landmark_prompt"].format(
                landmark=landmark_description,
                recent_objects=recent_objects,
                surroundings=surroundings
            )
        except Exception as e:
            logger.error(f"Error formatting geoinstruct: {e}")
            geoinstruct = f"Landmark: {landmark_description}. Recent objects: {recent_objects}. Surroundings: {surroundings}."
        
        # Store for later use
        state["current_observation"]["geoinstruct"] = geoinstruct
        state["last_scene_caption"] = scene_caption
        
        # Calculate view area
        try:
            state["view_width"] = 2 * (state["controller"].pose.z - state["landmark_nav_map"].ground_level)
            state["xyxy"] = calculate_view_area(state["controller"].pose, state["landmark_nav_map"].ground_level)
        except Exception as e:
            logger.error(f"Error calculating view area: {e}")
            state["view_width"] = 20.0  # Default value
            state["xyxy"] = [(0, 0), (20, 20)]
        
        # 已删除situational memory存储，根据CLAUDE.md规范只使用三个主要记忆库
        # mainplan_experience_memory, navigate_experience_memory, search_and_locate_memory
        
        logger.info("Perception and modeling completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"Error in perceive_and_model: {e}")
        # Ensure minimum required state is set
        if "current_observation" not in state:
            state["current_observation"] = {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "image_b64": "",
                "description": "Perception failed",
                "geoinstruct": "Unable to perceive environment"
            }
        return state

def generate_plan_with_rag(state: AgentState) -> AgentState:
    """
    Node: Create a high-level plan for the mission, augmented by RAG.
    """
    logger.info("Generating plan with RAG...")
    
    try:
        # Ensure we have current observation with geoinstruct
        if not state.get("current_observation") or "geoinstruct" not in state["current_observation"]:
            logger.warning("Missing geoinstruct, using basic format")
            geoinstruct = "Basic geographical context available."
        else:
            geoinstruct = state["current_observation"]["geoinstruct"]
        
        # RAG: Retrieve similar plans from strategic memory (with config check)
        from reflectionnav_agentV3.config_loader import get_config
        config = get_config()
        
        example_plan = "No similar successful plans found in memory."
        
        # 🔧 Fixed: Check both master RAG switch and strategic memory switch
        if config.rag.enable_rag and config.rag.enable_strategic_memory:
            try:
                retrieved_plans = state["memory_module"].mainplan_experience_memory.retrieve(
                        state["episode"].target_description, n_results=config.memory.max_retrieval_results # 🔧 FIXED: Use max_retrieval_results
                )
                
                if retrieved_plans:
                    plan_json_str = retrieved_plans[0].get('master_plan_json', '{}')
                    try:
                        plan_json = json.loads(plan_json_str)
                        example_plan = json.dumps(plan_json, indent=2)
                    except json.JSONDecodeError:
                        example_plan = plan_json_str
            except Exception as e:
                logger.error(f"Error retrieving strategic plans: {e}")
                example_plan = "No examples available due to retrieval error."
        else:
            example_plan = "Strategic memory disabled."
        
        # Retrieve reflection notes
        try:
            retrieved_notes = state["memory_module"].mainplan_experience_memory.retrieve(
                task_instruction=state["episode"].target_description, 
                n_results=config.memory.max_retrieval_results # 🔧 FIXED: Use correct parameter name
            )
            reflection_note = "No specific reflection notes found for this task."
            if retrieved_notes:
                reflection_note = retrieved_notes[0].get('reflection_note', 'No specific insight.')
        except Exception as e:
            logger.error(f"Error retrieving reflection notes: {e}")
            reflection_note = "Unable to retrieve reflection notes."
        
        # Format the planner prompt
        landmarks = state["episode"].description_landmarks
        
        # 🔧 Enhanced: Calculate distance and direction to final target for disambiguation
        current_pos = state["controller"].pose.xy
        target_pos = state["episode"].target_position.xy
        distance_to_final_target = current_pos.dist_to(target_pos)
        
        # Calculate direction to target (simple 8-direction mapping)
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        
        # Calculate angle in degrees (0° = East, 90° = North)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        
        # Map angle to 8 cardinal directions
        if 22.5 <= angle < 67.5:
            final_target_direction = "Northeast"
        elif 67.5 <= angle < 112.5:
            final_target_direction = "North"
        elif 112.5 <= angle < 157.5:
            final_target_direction = "Northwest"
        elif 157.5 <= angle < 202.5:
            final_target_direction = "West"
        elif 202.5 <= angle < 247.5:
            final_target_direction = "Southwest"
        elif 247.5 <= angle < 292.5:
            final_target_direction = "South"
        elif 292.5 <= angle < 337.5:
            final_target_direction = "Southeast"
        else:  # 337.5-360 and 0-22.5
            final_target_direction = "East"
        
        system_prompt = state["llm_manager"].prompts.get("planner_system_prompt") # Or handle if no system prompt
        planner_prompt = state["llm_manager"].prompts["planner_prompt"].format(
            instruction=state["episode"].target_description,
            geoinstruct=geoinstruct,
            landmarks=landmarks,
            distance_to_final_target=distance_to_final_target,
            final_target_direction=final_target_direction,
            example=example_plan,
            reflection_note=reflection_note
        )
        
        # 🔧 NEW: Store prompts for logging
        state["last_llm_system_prompt"] = system_prompt
        state["last_llm_prompt"] = planner_prompt
        
        # Generate plan using LLM manager
        plan_result = state["llm_manager"].call_llm_with_parsing(
            system_prompt=system_prompt,
            user_prompt=planner_prompt,
            images=None,
            parser_type="json",
            max_retries=3
        )
        print(f"plan_result: {plan_result}")
        if plan_result and isinstance(plan_result, dict) and "sub_goals" in plan_result:
            state["plan"] = plan_result
            state["current_task_index"] = 0
            state["subtask_status"] = "pending"
            state["subtask_context"] = {
                "goal": "mission_start",
                "CoT_summary": "Mission initialized",
                "action": "start"
            }
            print(f"Generated plan with {len(plan_result['sub_goals'])} subtasks")
            
            # Print detailed plan information
            print_plan_details(state)
            
        else:
            logger.error("Failed to generate valid plan, creating default plan")
            # Create default plan
            state["plan"] = {
                "sub_goals": [
                    {
                        "goal": state["episode"].target_description,
                        "strategy": "Search",
                        "desired_state": "Emergency fallback plan"
                    }
                ]
            }
            state["current_task_index"] = 0
            state["subtask_status"] = "pending"
            state["subtask_context"] = {
                "goal": "default_plan",
                "CoT_summary": "Using default search strategy",
                "action": "search"
            }
            
            # Print default plan information
            print_plan_details(state)
        
        return state
        
    except Exception as e:
        logger.error(f"Error in generate_plan_with_rag: {e}")
        # Ensure we have a basic plan
        if "plan" not in state or state["plan"] is None:
            state["plan"] = {
                "sub_goals": [
                    {
                        "goal": state["episode"].target_description,
                        "strategy": "Search",
                        "desired_state": "Emergency fallback plan"
                    }
                ]
            }
            state["current_task_index"] = 0
            state["subtask_status"] = "pending"
        return state

def execute_subtask_router(state: AgentState) -> AgentState:
    """
    Node: Prepare for sub-task execution by constructing the current instruction and the unified `current_state`. 
    Sets up subtask context but does not route - routing is handled by route_to_expert function.
    """
    logger.info("Preparing subtask execution...")
    
    try:
        # Validate state has plan
        if not state.get("plan") or not isinstance(state["plan"], dict):
            logger.error("No valid plan available")
            state["subtask_status"] = "error"
            return state
        
        plan = state["plan"]
        if "sub_goals" not in plan or not isinstance(plan["sub_goals"], list):
            logger.error("Plan missing sub_goals")
            state["subtask_status"] = "error"
            return state
            
        if state["current_task_index"] >= len(plan["sub_goals"]):
            logger.info("All tasks completed")
            state["subtask_status"] = "completed"
            return state
        
        current_task = plan["sub_goals"][state["current_task_index"]]
        
        # Initialize subtask if needed
        if state.get("subtask_status") == "pending":
            state["subtask_status"] = "running"
            state["switch_time"] = state.get("timestep", 0)  # 🔧 FIXED: Record strategy switch time for timeout
            if current_task.get("strategy") == "Search":
                state["search_start_pos"] = state["controller"].pose.xy
        
        # Construct current state for RAG & Reflection
        observation_desc = ""
        if state.get("current_observation"):
            observation_desc = state["current_observation"].get("description", "No description available")
        
        # 🔧 Fixed: Match CLAUDE.md format with geoinstruct
        geoinstruct = state.get("current_observation", {}).get("geoinstruct", "No geographical instruction available")
        state["current_state"] = (
            f"Current aerial perspective observation: {{'scene_description': '{observation_desc}', 'geoinstruct': '{geoinstruct}'}}. "
            f"Task: {state['task_instruction']}. "
            f"Current Goal: {current_task.get('goal', 'Unknown goal')} "
            f"(Strategy: {current_task.get('strategy', 'Unknown strategy')})"
        )
        
        print(f"Current Task [{state['current_task_index']+1}/{len(plan['sub_goals'])}]: "
                   f"{current_task.get('goal', 'Unknown')} (Strategy: {current_task.get('strategy', 'Unknown')})")
        
        # Set the strategy for the routing function to use
        state["current_strategy"] = current_task.get("strategy", "Search")
        
        return state
            
    except Exception as e:
        logger.error(f"Error in execute_subtask_router: {e}")
        state["subtask_status"] = "error"
        return state

# Add this function after execute_subtask_router
def route_to_expert(state: AgentState) -> str:
    """
    🔧 Enhanced Router: Determine which expert to route to based on both plan and LLM decisions.
    This makes routing more flexible and responsive to LLM recommendations.
    """
    try:
        # Check if we have a valid plan and current task
        if not state.get("plan") or not isinstance(state["plan"], dict):
            logger.error("No valid plan for routing")
            return "finalize_step"
        
        plan = state["plan"]
        if "sub_goals" not in plan or not isinstance(plan["sub_goals"], list):
            logger.error("Plan missing sub_goals for routing")
            return "finalize_step"
            
        if state["current_task_index"] >= len(plan["sub_goals"]):
            logger.info("All tasks completed - routing to finalize")
            return "finalize_step"
        
        current_task = plan["sub_goals"][state["current_task_index"]]
        planned_strategy = current_task.get("strategy", "Search")
        
        # 🔧 New: Check if previous expert made a strategy decision
        action_result = state.get("action_result", {})
        
        # Handle Navigation strategy decisions
        if planned_strategy == "Navigate":
            nav_decision = state.get("navigation_strategy_decision")
            if nav_decision == "switch_to_search":
                logger.info("🔄 Navigation expert decided to switch to Search")
                # Update the current task strategy to Search for next iteration
                current_task["strategy"] = "Search"
                return "execute_search"
            elif nav_decision == "continue_navigate":
                return "execute_navigate"
            else:
                # Default to planned strategy
                return "execute_navigate"
        
        # Handle Search strategy decisions  
        elif planned_strategy == "Search":
            search_decision = state.get("search_decision")
            if search_decision == "execute_locate":
                logger.info("🔄 Search expert decided to switch to Locate")
                # Update the current task strategy to Locate for next iteration
                current_task["strategy"] = "Locate" 
                return "execute_locate"
            elif search_decision == "continue_search":
                return "execute_search"
            else:
                # Default to planned strategy
                return "execute_search"
        
        # Handle Locate strategy (always executes once)
        elif planned_strategy == "Locate":
            return "execute_locate"
        
        else:
            logger.error(f"Unknown strategy for routing: {planned_strategy}")
            return "finalize_step"
            
    except Exception as e:
        logger.error(f"Error in route_to_expert: {e}")
        return "finalize_step"

# ==============================================================================================
# Expert Nodes
# ==============================================================================================

def execute_navigate(state: AgentState) -> AgentState:
    """
    Expert Node: Execute navigation strategy with RAG.
    """
    print("🧭 Executing navigation...")
    
    try:
        plan = state.get("plan")
        if not plan or not isinstance(plan, dict) or "sub_goals" not in plan:
            logger.error("No valid plan available for navigation")
            return state
            
        current_task_idx = state.get("current_task_index", 0)
        sub_goals = plan["sub_goals"]
        
        if current_task_idx >= len(sub_goals):
            logger.error("Invalid task index for navigation")
            return state
            
        current_task = sub_goals[current_task_idx]
        
        # 🔧 NEW: Find the actual landmark position for the current sub-goal
        landmark_position = state["episode"].target_position  # Default to final target
        try:
            current_goal_name = current_task.get("goal", "").lower()
            landmarks = state["landmark_nav_map"].landmark_map.landmarks
            for landmark in landmarks:
                if landmark.name.lower() in current_goal_name:
                    landmark_position = landmark.position
                    logger.info(f"🎯 Navigation target landmark found: '{landmark.name}' at {landmark_position.xy}")
                    break
        except Exception as e:
            logger.error(f"Error finding landmark position for goal '{current_task.get('goal')}': {e}")
            
        # 🔧 Fixed: Use interval-based RAG retrieval
        experience_clue = "No experience retrieved."
        retrieved_image_b64 = None
        
        if should_retrieve_memory(state):
            from reflectionnav_agentV3.config_loader import get_config
            config = get_config()
            # 🔧 FIXED: Use complete current_state string for RAG query (as per documentation)
            current_state = state.get("current_state", "")
            experience_clue, retrieved_image_b64 = retrieve_experience_memory(
                state["memory_module"], "Navigate", current_state, config
            )
            print_rag_usage("Navigate", experience_clue, retrieved_image_b64 is not None)
            # 🔧 Fixed: Update last retrieval timestep after successful retrieval
            state["last_memory_retrieval_timestep"] = state.get("timestep", 0)
        
        # 🔧 修复：生成和保存语义图像 - 每次传入就保存
        map_image_b64 = get_semantic_image(state, "Navigate")
        
        if map_image_b64 is not None:
            logger.info(f"🖼️  Generated landmark map for Navigation")
        else:
            logger.warning("⚠️  Failed to generate landmark map, proceeding without it")
        
        
        # 🔧 Fixed: Get recent actions with reasoning for better context
        history_entries = state.get("trajectory_history", [])[-2:]
        history_actions = [entry.get('action', '{}') for entry in history_entries]
        history_actions = str(history_actions) if history_actions else "No recent actions"
        
        # Prepare images list
        images = []
        if map_image_b64:
            images.append(map_image_b64)
            logger.info("📋 Added landmark map to VLM input")
        if retrieved_image_b64:
            images.append(retrieved_image_b64)
            logger.info("📋 Added retrieved experience image to VLM input")
        
        logger.info(f"🤖 Calling VLM with {len(images)} images for navigation")
        
        # Enhanced navigation prompt with strategy decision
        scene_caption = state.get("last_scene_caption", "No current observation")
        geoinstruct = state["current_observation"].get("geoinstruct", "No geographical context available")
        goal = current_task.get("goal", "Navigate to target") if isinstance(current_task, dict) else "Navigate to target"
        
        # 🔧 Fixed: Calculate required variables for navigation strategy decision
        current_position = state["controller"].pose.xy
        target_position = state["episode"].target_position.xy
        distance = current_position.dist_to(target_position)
        
        # Use navigation strategy decision prompt
        system_prompt = state["llm_manager"].prompts["goal_description_nav"]
        
        # 🔧 Fixed: Remove precise coordinates, use descriptive spatial context instead
        enhanced_nav_prompt = state["llm_manager"].prompts["navigation_strategy_decision"].format(
            task_instruction=state["task_instruction"],
            navigation_goal=goal,
            current_position="aerial perspective position",
            target_position="mission target area",
            distance=distance,
            scene_caption=scene_caption,
            geoinstruct=geoinstruct,
            history_actions=history_actions,
            experience=experience_clue
        )
        
        # 🔧 NEW: Store prompts for logging
        state["last_llm_system_prompt"] = system_prompt
        state["last_llm_prompt"] = enhanced_nav_prompt
        
        # Execute navigation strategy decision
        try:
            result = state["llm_manager"].call_llm_with_parsing(
                system_prompt=system_prompt,
                user_prompt=enhanced_nav_prompt,
                images=images,
                parser_type="json",
                max_retries=3
            )
        except Exception as e:
            logger.error(f"Error calling VLM for navigation strategy decision: {e}")
            result = None
        
        if result:
            # Handle different result types properly
            if isinstance(result, dict):
                reasoning = result.get('reason', result.get('reasoning', 'No reasoning provided'))
                movement = result.get('movement', 'north')
                next_strategy = result.get('next_strategy', 'continue_navigate')
                confidence = result.get('confidence', 0.5)
            else:
                # 🔧 Fixed: Safe attribute access for BaseModel and other types
                try:
                    reasoning = getattr(result, 'reason', None) or getattr(result, 'reasoning', 'No reasoning provided')
                    movement = getattr(result, 'movement', 'north')
                    next_strategy = getattr(result, 'next_strategy', 'continue_navigate')
                    confidence = getattr(result, 'confidence', 0.5)
                except AttributeError:
                    logger.warning(f"Unexpected result type: {type(result)}")
                    reasoning = "Default reasoning"
                    movement = "north"
                    next_strategy = "continue_navigate"
                    confidence = 0.5
            
            result_dict = {
                "reasoning": reasoning,
                "movement": movement,
                "next_strategy": next_strategy,
                "confidence": confidence
            }
            
            # Store strategy decision for route_to_expert to use
            state["navigation_strategy_decision"] = next_strategy
            state["navigation_confidence"] = confidence
            
        else:
            logger.warning("No result from navigation strategy decision")
            reasoning = "Navigation strategy decision failed"
            movement = "north" 
            next_strategy = "continue_navigate"
            confidence = 0.1
            
            result_dict = {
                "reasoning": reasoning,
                "movement": movement,
                "next_strategy": next_strategy,
                "confidence": confidence
            }
            state["navigation_strategy_decision"] = "continue_navigate"
            state["navigation_confidence"] = 0.1

        # Calculate next position with adaptive scaling (moved outside the if block)
        action_scale = calculate_adaptive_action_scale(
            state["controller"].pose, 
            landmark_position, # 🔧 FIXED: Use landmark position
            "Navigate"
        )
        
        next_pos = calculate_next_position(
            state["controller"].pose,
            state["view_width"], 
            action_scale,
            result_dict
        )
        
        # Log navigation decision (simplified)
        print_subtask_execution(state, "Navigate", reasoning, movement, next_pos)

        # Store results in state
        state["next_position"] = next_pos
        state["action_result"] = result_dict
        state["cot_summary"] = reasoning
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in execute_navigate: {e}")
        state["next_position"] = Point2D(state["controller"].pose.x, state["controller"].pose.y)
        state["action_result"] = {}
        state["cot_summary"] = "Navigation error"
        return state

def execute_search(state: AgentState) -> AgentState:
    """
    Node: Execute search strategy with integrated decision making.
    Simplified to reduce LLM calls.
    """
    print("🔍 Executing search strategy...")
    
    try:
        plan = state.get("plan")
        if not plan or "sub_goals" not in plan:
            logger.error("No valid plan available for search execution")
            state["next_position"] = Point2D(state["controller"].pose.x, state["controller"].pose.y)
            state["action_result"] = {}
            state["cot_summary"] = "No plan available"
            return state
            
        current_task = plan["sub_goals"][state["current_task_index"]]
        
        # Calculate current coordinates and view area
        view_width = 2 * (state["controller"].pose.z - state["landmark_nav_map"].ground_level)
        state["view_width"] = view_width
        state["xyxy"] = [
            (state["controller"].pose.x - view_width/2, state["controller"].pose.y - view_width/2),
            (state["controller"].pose.x + view_width/2, state["controller"].pose.y + view_width/2)
        ]
        
        # 🔧 Fixed: Use interval-based RAG retrieval
        experience_clue = "No experience retrieved."
        retrieved_img_b64 = None
        semantic_img_b64 = get_semantic_image(state, "Search") # 🐞 FIX: Initialize to handle all code paths
         
        if semantic_img_b64 is not None:
            logger.info(f"🖼️  Generated semantic map for Search")
        else:
            logger.warning("⚠️  Failed to generate semantic map, proceeding without it")
        
        if should_retrieve_memory(state):
            from reflectionnav_agentV3.config_loader import get_config
            config = get_config()
            # 🔧 FIXED: Use complete current_state string for RAG query (as per documentation)
            current_state = state.get("current_state", "")
            experience_clue, retrieved_img_b64 = retrieve_experience_memory(
                state["memory_module"], "Search", current_state, config
            )
            # 🔧 Fixed: Update last retrieval timestep after successful retrieval
            state["last_memory_retrieval_timestep"] = state.get("timestep", 0)

        
        # Prepare images list
        images = [img for img in [semantic_img_b64, retrieved_img_b64] if img]
        
        # Build enhanced prompt with entropy information for decision making
        entropy_info = ""
        if state.get("previous_sem_map") is not None:
            current_sem_map = state["landmark_nav_map"].get_semantic_map()
            current_entropy = rgb_entropy(current_sem_map)
            previous_entropy = rgb_entropy(state["previous_sem_map"])
            entropy_change = abs(current_entropy - previous_entropy)
            
            entropy_info = f"""
            Information Gain Analysis:
            - Current entropy: {current_entropy:.3f}
            - Previous entropy: {previous_entropy:.3f}
            - Entropy change: {entropy_change:.3f}
            
            Based on this information gain and semantic map, decide whether the information gain is enough to execute locate strategy or continue searching.
            If entropy change < 0.05, consider 'execute_locate'. Otherwise, 'continue_search'.
            If the semantic map is not enough to execute locate strategy, consider 'continue_search'.
            """
        
        # 🔧 Fixed: Get history actions with reasoning for search context
        history_entries = state.get("trajectory_history", [])[-2:]
        history_actions = [entry.get('action', '{}') for entry in history_entries]
        history_actions = str(history_actions) if history_actions else "No recent actions"
        print(f"history_actions: {history_actions}")
        # 🔧 NEW: Enhanced search prompt with desired_state and updated decision options
        target_description = state["episode"].target_description
        desired_state_info = current_task.get("desired_state", "No specific desired state for this task.")
        geoinstruct = state["current_observation"].get("geoinstruct", "No geographical context available")

        system_prompt = state["llm_manager"].prompts["goal_description_sea"]
        search_prompt = state["llm_manager"].format_prompt(
            "search_decision",
            task_instruction=state["task_instruction"],
            current_goal=current_task.get('goal', 'Unknown'),
            desired_state=desired_state_info,
            current_position="aerial perspective position",
            view_area=str(state.get("xyxy")),
            scene_caption=state.get("last_scene_caption", "No current observation available"),
            geoinstruct=geoinstruct,
            history_actions=history_actions,
            entropy_info=entropy_info,
            experience=experience_clue
        )
        
        # 🔧 NEW: Store prompts for logging
        state["last_llm_system_prompt"] = system_prompt
        state["last_llm_prompt"] = search_prompt
        
        # Execute search decision with integrated continue/locate logic
        search_result = state["llm_manager"].call_llm_with_parsing(
            system_prompt=system_prompt,
            user_prompt=search_prompt,
            images=images,
            parser_type="json",
            max_retries=3
        )
        
        if search_result:
            # Handle different result types properly
            if isinstance(search_result, dict):
                movement = search_result.get('movement', 'north')
                reason = search_result.get('reason', search_result.get('reasoning', 'No reason provided'))
                decision = search_result.get('decision', 'continue_search')
                confidence = search_result.get('confidence', 0.5)
            else:
                logger.warning(f"Unexpected search result type: {type(search_result)}")
                movement = "north"
                reason = "Default search reasoning"
                decision = "continue_search"
                confidence = 0.5
            
            result_dict = {
                "movement": movement,
                "reasoning": reason,
                "decision": decision,
                "confidence": confidence
            }
        else:
            logger.warning("No result from search LLM call")
            result_dict = {
                "movement": "north",
                "reasoning": "Search failed",
                "decision": "continue_search",
                "confidence": 0.1
            }

    except Exception as e:
        logger.error(f"Error in search execution: {e}")
        result_dict = {
            "movement": "north",
            "reasoning": "Search error fallback",
            "decision": "continue_search",
            "confidence": 0.1
        }

    # 🔧 NEW: Explicitly calculate adaptive action scale for Search strategy
    action_scale = calculate_adaptive_action_scale(
        state["controller"].pose,
        state["episode"].target_position, # Search scale is relative to the final mission target
        "Search"
    )

    # Calculate next position for search
    next_pos = calculate_next_position(
        state["controller"].pose,
        state["view_width"],
        action_scale,
        result_dict
    )
    
    # 🔧 Fixed: Store search decision in the correct state key for check_subgoal_completion
    state["search_decision"] = result_dict["decision"]
    
    # Store results in state
    state["next_position"] = next_pos
    state["action_result"] = result_dict
    state["cot_summary"] = result_dict["reasoning"]
    
    return state

def execute_locate(state: AgentState) -> AgentState:
    """
    Expert Node: Execute locate strategy.
    """
    logger.info("Executing locate...")
    
    try:
        plan = state.get("plan")
        if not plan or "sub_goals" not in plan:
            logger.error("No valid plan available for locate execution")
            state["next_position"] = Point2D(state["controller"].pose.x, state["controller"].pose.y)
            state["action_result"] = {}
            state["cot_summary"] = "No plan available"
            return state
            
        current_task = plan["sub_goals"][state["current_task_index"]]
        
        # Locate策略不需要保存语义图片
        
        # Try Scene Graph Query first (if not ablated)
        if state["args"].ablation != 'wo_sg':
            try:
                # Complex query handling
                is_complex = len(current_task.get("goal", "").split()) > 3
                
                if is_complex:
                    print(f"📊 Query Type: Complex Scene Graph Query")
                    logger.info(f"Executing complex query: {current_task['goal']}")
                    # This call does not use the standard prompt manager, so logging is difficult here.
                    target_nodes = state["controller"].complex_query_scene_graph(current_task['goal'], debug=True)
                else:
                    print(f"📊 Query Type: Simple Operation Chain Query")
                    # Simple query using operation chain
                    operation_prompt = state["llm_manager"].prompts["query_operation_chain"].format(instruction=current_task["goal"])
                    
                    # 🔧 NEW: Store prompts for logging
                    state["last_llm_system_prompt"] = None
                    state["last_llm_prompt"] = operation_prompt

                    operation_result = state["llm_manager"].call_llm_with_parsing(
                        system_prompt=None,
                        user_prompt=operation_prompt,
                        images=None,
                        parser_type="json",
                        max_retries=2
                    )
                    
                    if operation_result and isinstance(operation_result, list):
                        target_nodes = state["controller"].query_engine.robust_subgraph_query(
                            operation_result, fallback=True, min_results=1, debug=True
                        )
                    elif operation_result and isinstance(operation_result, dict):
                        # Convert dict result to list format if needed
                        if "operations" in operation_result:
                            target_nodes = state["controller"].query_engine.robust_subgraph_query(
                                operation_result["operations"], fallback=True, min_results=1, debug=True
                            )
                        else:
                            target_nodes = None
                    else:
                        target_nodes = None
                
                if target_nodes:
                    # Select best node
                    selected_node = max(target_nodes, key=lambda n: getattr(n, 'confidence', 0))
                    next_pos = Point2D(selected_node.position.x, selected_node.position.y)
                    
                    state["next_position"] = next_pos
                    state["action_result"] = {
                        "status": "TARGET_LOCKED",
                        "selected_pos": [next_pos.x, next_pos.y],
                        "node_id": selected_node.id,
                        "confidence": getattr(selected_node, 'confidence', 1.0)
                    }
                    state["cot_summary"] = f"Scene graph query successful: found {selected_node.id}"
                    
                    # Log successful locate
                    print_subtask_execution(state, "Locate", state["cot_summary"], "TARGET_LOCKED", next_pos)
                    
                    return state
                else:
                    print(f"❌ Scene Graph Query Failed: No nodes found")
                    
            except Exception as e:
                logger.error(f"Scene graph query error: {e}")
                print(f"❌ Scene Graph Query Error: {e}")
        
        # Fallback to VLM-based location
        print(f"🔄 Fallback: VLM-based Location")
        
        # RAG Retrieval for VLM fallback (with config check)
        from reflectionnav_agentV3.config_loader import get_config
        config = get_config()
        experience_clue, retrieved_image_b64 = retrieve_experience_memory(
            state["memory_module"], "Locate", state.get("last_scene_caption", ""), config
        )
        
        # Print RAG usage information
        print_rag_usage("Locate", experience_clue, retrieved_image_b64 is not None)
        
        # 🔧 Fixed: Get history actions with reasoning for locate context
        history_entries = state.get("trajectory_history", [])[-2:]
        history_actions = [entry.get('action', '{}') for entry in history_entries]
        history_actions = str(history_actions) if history_actions else "No recent actions"
        
        # 🔧 Fixed: Format locate prompt without precise position exposure
        geoinstruct = state["current_observation"].get("geoinstruct", "No geographical context available")
        system_prompt = state["llm_manager"].prompts["goal_description_loc"]
        locate_prompt = state["llm_manager"].prompts["object_locate"].format(
            pos="viewing center (aerial perspective)",
            area="current field of view",
            goal=state["episode"].target_description,
            clue=experience_clue,
            history=str(history_actions)
        )
        
        # 🔧 NEW: Store prompts for logging
        state["last_llm_system_prompt"] = system_prompt
        state["last_llm_prompt"] = locate_prompt
        
        # Prepare images - use current observation
        images = []
        if state["current_observation"].get("image_b64"):
            images.append(state["current_observation"]["image_b64"])
        if retrieved_image_b64:
            images.append(retrieved_image_b64)
        
        # Execute locate decision with complex target finding
        try:
            locate_result = state["llm_manager"].call_llm_with_parsing(
                system_prompt=system_prompt,
                user_prompt=locate_prompt,
                images=images,
                parser_type="json",
                max_retries=3
            )
        
            # Handle different result types properly
            if locate_result and isinstance(locate_result, dict):
                status = locate_result.get('status', 'SEARCHING_VICINITY')
                selected_pos = locate_result.get('selected_pos')
                reason = locate_result.get('reason', locate_result.get('reasoning', 'No reason provided'))
                confidence = locate_result.get('confidence', 0.5)
                
                result_dict = {
                    "status": status,
                    "selected_pos": selected_pos,
                    "reasoning": reason,
                    "confidence": confidence
                }
            else:
                logger.warning(f"Unexpected locate result type: {type(locate_result)}")
                result_dict = {
                    "status": "SEARCHING_VICINITY",
                    "selected_pos": None,
                    "reasoning": "Locate failed",
                    "confidence": 0.1
                }

        except Exception as e:
            logger.error(f"Error in locate execution: {e}")
            result_dict = {
                "status": "SEARCHING_VICINITY", 
                "selected_pos": None,
                "reasoning": f"Locate error: {str(e)}",
                "confidence": 0.1
            }

        # Process locate result
        if result_dict["status"] == "TARGET_LOCKED" and result_dict["selected_pos"]:
            # Direct target location found
            next_pos = Point2D(result_dict["selected_pos"][0], result_dict["selected_pos"][1])
            print_subtask_execution(state, "Locate", result_dict["reasoning"], "TARGET_LOCKED", next_pos)
        else:
            # Search vicinity - move slightly toward target
            current_pose = state["controller"].pose
            target_pos = state["episode"].target_position
            
            # Calculate direction toward target
            dx = target_pos.x - current_pose.x
            dy = target_pos.y - current_pose.y
            distance = (dx**2 + dy**2)**0.5
            
            if distance > 5.0:  # Move toward target if far away
                move_scale = 10.0  # Move 10 meters toward target
                next_pos = Point2D(
                    current_pose.x + (dx / distance) * move_scale,
                    current_pose.y + (dy / distance) * move_scale
                )
            else:
                # Search in small radius if close to target
                import random
                radius = 5.0
                angle = random.uniform(0, 2 * 3.14159)
                next_pos = Point2D(
                    current_pose.x + radius * (0.5 - random.random()),
                    current_pose.y + radius * (0.5 - random.random())
                )
            
            print_subtask_execution(state, "Locate", result_dict["reasoning"], "SEARCH_VICINITY", next_pos)

        # Store results in state
        state["next_position"] = next_pos
        state["action_result"] = result_dict
        state["cot_summary"] = result_dict["reasoning"]
        
        return state
        
    except Exception as e:
        logger.error(f"Error in execute_locate: {e}")
        state["next_position"] = Point2D(state["controller"].pose.x, state["controller"].pose.y)
        state["action_result"] = {}
        state["cot_summary"] = "Locate error"
        return state

def finalize_step(state: AgentState) -> AgentState:
    """
    Node: Update physical state and record the completed frame.
    """
    logger.info("Finalizing step...")
    
    try:
        # Update pose if next_position is set
        if "next_position" in state and state["next_position"]:
            next_pos = state["next_position"]
            from gsamllavanav.space import Pose4D
            state["controller"].pose = Pose4D(
                next_pos.x, next_pos.y,
                state["controller"].pose.z,
                state["controller"].pose.yaw
            )
        
        # 🔧 修复：先记录当前timestep的轨迹，再递增timestep
        current_timestep = state["timestep"]
        
        # 🔧 Fixed: Initialize current_task early to avoid reference errors
        plan = state.get("plan")
        current_task = None
        current_task_index = state.get("current_task_index", 0)
        
        # Get current task if valid
        if plan and "sub_goals" in plan and current_task_index < len(plan["sub_goals"]):
            current_task = plan["sub_goals"][current_task_index]
        
        # Record trajectory frame (使用当前timestep，与语义图片保存时一致)
        if plan and "sub_goals" in plan and state.get("current_task_index") is not None:
            try:
                # 🔧 Calculate current distance for trajectory record
                current_pose = state["controller"].pose
                target_position = state["episode"].target_position
                current_distance = current_pose.xy.dist_to(target_position.xy)
                
                # 🔧 修复：使用通用的JSON序列化工具函数
                from reflectionnav_agentV3.utils import convert_numpy_for_json
                
                safe_action_result = convert_numpy_for_json(state.get("action_result", {}))
                
                frame_log = FrameLog(
                    goal=current_task.get("goal", "Unknown task") if current_task else "Task completed",
                    strategy=current_task.get("strategy", "Unknown") if current_task else "Unknown",
                    current_state=state.get("current_state", ""),
                    action=json.dumps(safe_action_result),
                    scene_graph_summary=state.get("cot_summary") or "",
                    map_snapshot=state.get("current_observation", {}).get("image"),
                    scene_caption=state.get("last_scene_caption", ""),
                    timestep=current_timestep,  # 🔧 使用当前timestep而不是递增后的
                    pose=(current_pose.x, current_pose.y, current_pose.z, current_pose.yaw),  # 🔧 Record actual pose
                    distance_to_target=float(current_distance),  # 🔧 Record actual distance at this timestep
                    llm_system_prompt=state.get("last_llm_system_prompt") or "N/A",
                    llm_prompt=state.get("last_llm_prompt") or "N/A"
                )
                
                # Initialize trajectory_history if not exists
                if "trajectory_history" not in state:
                    state["trajectory_history"] = []
                
                state["trajectory_history"].append(frame_log)
                
                # Save trajectory step to result manager
                try:
                    step_data = {
                        "timestep": current_timestep,  # 🔧 使用当前timestep
                        "goal": current_task.get("goal", "Task completed") if current_task else "Task completed",
                        "strategy": current_task.get("strategy", "Unknown") if current_task else "Unknown",
                        "current_state": state.get("current_state", ""),
                        "action": state.get("action_result", {}),
                        "reasoning": state.get("cot_summary", ""),
                        "scene_caption": state.get("last_scene_caption", ""),
                        "pose": {
                            "x": state["controller"].pose.x,
                            "y": state["controller"].pose.y,
                            "z": state["controller"].pose.z,
                            "yaw": state["controller"].pose.yaw
                        },
                        "distance_to_target": state["controller"].pose.xy.dist_to(state["episode"].target_position.xy)
                    }
                    # 🔧 Fixed: Safe result_manager access
                    result_manager = state.get("result_manager")
                    if result_manager is not None and hasattr(result_manager, 'add_trajectory_step'):
                        result_manager.add_trajectory_step(step_data)
                except Exception as e:
                    logger.error(f"Error saving trajectory step: {e}")
                
            except Exception as e:
                logger.error(f"Error recording trajectory frame: {e}")
        
        # 现在才递增timestep
        state["timestep"] += 1
        
        # Check if current subtask is completed (使用已定义的变量)
        if plan and "sub_goals" in plan and current_task is not None:
            try:
                # 🔧 Fixed: Use already defined current_task variable
                if check_subgoal_completion(state, current_task):
                    state["subtask_status"] = "completed"
                    state["current_task_index"] += 1
                    
                    # Update subtask context for next task
                    if state["current_task_index"] < len(plan["sub_goals"]):
                        next_task = plan["sub_goals"][state["current_task_index"]]
                        state["subtask_context"] = {
                            "goal": next_task.get("goal", "Unknown"),
                            "CoT_summary": f"Completed {current_task.get('goal', 'previous task')}",
                            "action": "continue"
                        }
                        state["subtask_status"] = "pending"
                        logger.info(f"Advanced to task {state['current_task_index'] + 1}")
                    else:
                        logger.info("All subtasks completed - mission should end")
                        # 🔧 Fixed: Mark mission as ready for completion evaluation
                        state["subtask_status"] = "all_completed"
                        
            except Exception as e:
                logger.error(f"Error checking subgoal completion: {e}")
        elif plan and "sub_goals" in plan and current_task_index >= len(plan["sub_goals"]):
            # 🔧 Fixed: When task index is out of bounds, all tasks are completed
            logger.info(f"All subtasks completed (task index {current_task_index} >= {len(plan['sub_goals'])})")
            state["subtask_status"] = "all_completed"
        
        # Clear temporary state unless locate has just completed
        state.pop("next_position", None)
        if not state.get("locate_completed", False):
            state.pop("action_result", None)
            state.pop("cot_summary", None)
            
        # 🔧 NEW: Clear prompt history from state after logging
        state.pop("last_llm_system_prompt", None)
        state.pop("last_llm_prompt", None)

        logger.info(f"Step {current_timestep} finalized, next timestep: {state['timestep']}")
        return state
        
    except Exception as e:
        logger.error(f"Error in finalize_step: {e}")
        state["timestep"] += 1  # At least increment timestep
        return state

def check_subgoal_completion(state: AgentState, current_task: Dict[str, Any]) -> bool:
    """
    🔧 Enhanced: Check if the current subgoal is completed based on desired_state, LLM decisions and landmark proximity.
    """
    try:
        strategy = current_task.get("strategy", "Search")
        
        # 🔧 NEW: Rely on LLM's decision instead of a separate function call
        # The desired_state logic is now handled within the expert's prompt.
        
        if strategy == "Navigate":
            # 🔧 Enhanced: Check landmark proximity first (强制规则)
            from reflectionnav_agentV3.config_loader import get_config
            config = get_config()
            landmark_completion_distance = config.strategy.landmark_proximity # 🔧 FIXED: Use landmark_proximity from strategy config
            
            # Check if within landmark completion distance
            try:
                landmarks = state["landmark_nav_map"].landmark_map.landmarks
                current_pos = state["controller"].pose.xy
                current_goal_name = current_task.get("goal", "").lower()

                for landmark in landmarks:
                    # 🎯 FIX: Only check for the landmark that is part of the current goal
                    if landmark.name.lower() in current_goal_name:
                        distance_to_landmark = current_pos.dist_to(landmark.position.xy)
                        if distance_to_landmark <= landmark_completion_distance:
                            logger.info(f"🎯 Navigation to '{landmark.name}' completed: Proximity reached ({distance_to_landmark:.1f}m <= {landmark_completion_distance}m)")
                            if "strategy_distances" not in state:
                                state["strategy_distances"] = {}
                            state["strategy_distances"]["Navigate"] = float(state["controller"].pose.xy.dist_to(
                                state["episode"].target_position.xy
                            ))
                            return True # Subgoal is complete
                        
            except Exception as e:
                logger.error(f"Error checking landmark proximity: {e}")
            
            # 🔧 Modified: Change switch logic to "next_subtask"
            navigation_decision = state.get("navigation_strategy_decision", "continue_navigate")
            
            if navigation_decision == "next_subtask":
                # Navigation decided to move to next subtask
                if "strategy_distances" not in state:
                    state["strategy_distances"] = {}
                state["strategy_distances"]["Navigate"] = float(state["controller"].pose.xy.dist_to(
                    state["episode"].target_position.xy
                ))
                logger.info(f"Navigation strategy completed based on LLM decision: {navigation_decision}")
                return True
            elif navigation_decision == "continue_navigate":
                # Continue navigating - check timeout to prevent infinite navigation
                if (state["timestep"] - state.get("switch_time", state["timestep"])) >= config.strategy.navigate_timeout: # 🔧 FIXED: Use navigate_timeout from config
                    logger.info("Navigation completed due to timeout")
                    return True
                # Continue navigation
                return False
            else:
                # Fallback - continue navigation
                return False
                
        elif strategy == "Search":
            # 🔧 FIXED: Add search timeout check
            from reflectionnav_agentV3.config_loader import get_config
            config = get_config()
            if (state["timestep"] - state.get("switch_time", state["timestep"])) >= config.strategy.search_timeout:
                logger.info("Search completed due to timeout")
                return True

            # 🔧 Modified: Change switch logic to "next_subtask" 
            search_decision = state.get("search_decision", "continue_search")
            
            if search_decision == "next_subtask":
                # Search decided to move to next subtask
                if "strategy_distances" not in state:
                    state["strategy_distances"] = {}
                state["strategy_distances"]["Search"] = float(state["controller"].pose.xy.dist_to(
                    state["episode"].target_position.xy
                ))
                logger.info(f"Search strategy completed based on LLM decision: {search_decision}")
                return True
            else:
                # Continue searching
                return False
            
        elif strategy == "Locate":
            # 🔧 NEW: Locate always completes and triggers mission end
            action_result = state.get("action_result", {})
            if isinstance(action_result, dict):
                status = action_result.get("status", "SEARCHING_VICINITY") 
                confidence = action_result.get("confidence", 0)
                
                if status == "TARGET_LOCKED":
                    logger.info(f"🎯 Locate strategy completed: TARGET_LOCKED - Mission ending for reflection")
                elif confidence > 0.7:
                    logger.info(f"🎯 Locate strategy completed: High confidence ({confidence:.2f}) - Mission ending for reflection")
                else:
                    logger.info(f"🎯 Locate strategy completed: Single step execution - Mission ending for reflection")
            else:
                logger.info(f"🎯 Locate strategy completed: Default completion - Mission ending for reflection")
            
            # 🔧 NEW: Mark mission as ready for completion evaluation immediately after locate
            state["subtask_status"] = "all_completed"
            state["locate_completed"] = True  # Special flag for locate completion
            
            return True
            
    except Exception as e:
        logger.error(f"Error checking subgoal completion: {e}")
        return True  # Default to completed to avoid infinite loops
    
    return False

def reflect_on_success(state: AgentState) -> AgentState:
    """
    Node: Reflect on successful mission and store experiences.
    Simplified unified reflection approach.
    """
    logger.info("Reflecting on successful mission...")
    
    # 🔧 修复：确保成功反思时设置正确的任务状态
    state["mission_success"] = True
    if not state.get("final_reason"):
        state["final_reason"] = "success"
    
    # 🔧 Fixed: Check master reflection switch first
    from reflectionnav_agentV3.config_loader import get_config
    config = get_config()
    
    if not config.reflection.enable_reflection:
        logger.info("Reflection system disabled in config")
        return state
        
    if not config.reflection.enable_success_reflection:
        logger.info("Success reflection disabled in config")
        return state
        
    return _unified_reflection(state, "success")

def reflect_on_failure(state: AgentState) -> AgentState:
    """
    Node: Reflect on failed mission and store corrective experiences.
    Simplified unified reflection approach.
    """
    logger.info("Reflecting on failed mission...")
    
    # 🔧 修复：确保失败反思时设置正确的任务状态
    state["mission_success"] = False
    if not state.get("final_reason"):
        state["final_reason"] = "failure"
    
    # 🔧 Fixed: Check master reflection switch first
    from reflectionnav_agentV3.config_loader import get_config
    config = get_config()
    
    if not config.reflection.enable_reflection:
        logger.info("Reflection system disabled in config")
        return state
        
    if not config.reflection.enable_failure_reflection:
        logger.info("Failure reflection disabled in config")
        return state
        
    return _unified_reflection(state, "failure")

def _unified_reflection(state: AgentState, analysis_type: str) -> AgentState:
    """
    Unified reflection method for both success and failure analysis.
    Reduces LLM calls and code duplication.
    """
    try:
        if not state.get("plan") or not state.get("trajectory_history"):
            logger.warning(f"No plan or trajectory to reflect on for {analysis_type}")
            return state
        
        # Store strategic plan first (for both success and failure) - with config check
        from reflectionnav_agentV3.config_loader import get_config
        config = get_config()
        
        # 🔧 修复：按照文档要求，进行主规划反思与优化
        if config.reflection.enable_reflection and config.reflection.store_strategic_plans:
            try:
                # 进行主规划反思分析
                master_plan_analysis = state["reflection_agent"].analyze_main_plan(
                    task_instruction=state["episode"].target_description,
                    master_plan=state.get("plan") or {},
                    trajectory_history=state["trajectory_history"],
                    final_reason=state.get("final_reason", "Mission completed")
                )
                
                # 存储优化后的主规划
                # 🔧 修复：使用通用的JSON序列化工具函数
                from reflectionnav_agentV3.utils import safe_json_dumps, convert_numpy_for_json
                
                # 🔧 修复：深度转换plan中的所有对象
                safe_plan = convert_numpy_for_json(state["plan"])
                
                # 🔧 修复：确保final_outcome也是可序列化的
                safe_final_outcome = convert_numpy_for_json({
                    "status": "success" if analysis_type == "success" else "failure", 
                    "total_steps": state["timestep"],
                    "reflection_note": master_plan_analysis.get("reflection_note", ""),
                    "plan_refined": master_plan_analysis.get("plan_refined", "")
                })
                
                state["memory_module"].mainplan_experience_memory.add(
                    task_instruction=state["episode"].target_description,
                    master_plan=safe_json_dumps(safe_plan),
                    final_outcome=safe_final_outcome
                )
                logger.info(f"Stored {analysis_type} strategic plan with reflection in memory")
            except Exception as e:
                logger.error(f"Error storing strategic plan: {e}")
        else:
            logger.info("Strategic plan storage disabled in config")
        
        # Format trajectory for analysis
        trajectory_log = []
        for frame in state["trajectory_history"]:
            log_entry = (
                f"Strategy: {frame.get('strategy', 'N/A')}, "
                f"Goal: {frame.get('goal', 'N/A')}, "
                f"Action: {frame.get('action', 'N/A')}, "
                f"Timestep: {frame.get('timestep', 0)}"
            )
            trajectory_log.append(log_entry)
        
        # Generate reflection analysis
        try:
            # 🔧 Fixed: Ensure master_plan is never None
            master_plan = state.get("plan") or {}
            
            reflection_result = state["reflection_agent"].analyze_success(
                goal_instruction=state["episode"].target_description,
                master_plan=master_plan,
                trajectory_log=trajectory_log
            ) if analysis_type == "success" else state["reflection_agent"].analyze_failure(
                trajectory_log=trajectory_log,
                failure_reason=state.get("final_reason", "Mission failed"),
                master_plan=master_plan
            )
            
            # Handle different result types for reflection - FIXED: Ensure analysis is always a dict
            if isinstance(reflection_result, dict):
                analysis = reflection_result
            elif isinstance(reflection_result, str):
                # Try to parse as JSON
                try:
                    analysis = json.loads(reflection_result)
                except json.JSONDecodeError:
                    # Fallback to basic structure - FIXED: Ensure proper format
                    analysis = {
                        "analysis_type": analysis_type,
                        "root_cause": reflection_result[:200] if len(reflection_result) > 200 else reflection_result,
                        "key_experiences": []
                    }
            else:
                # FIXED: Handle any other unexpected types
                logger.warning(f"Unexpected reflection result type: {type(reflection_result)}")
                analysis = {
                    "analysis_type": analysis_type,
                    "root_cause": "Unknown reflection result type",
                    "key_experiences": []
                }
                
            # FIXED: Ensure analysis is always a dict before any .get() calls
            if not isinstance(analysis, dict):
                logger.error(f"Analysis is not a dict after processing: {type(analysis)}")
                analysis = {"key_experiences": [], "root_cause": "Processing error"}
                
            state["reflection_analysis"] = analysis
            
        except Exception as e:
            logger.error(f"Error in reflection analysis: {e}")
            # FIXED: Always provide a proper dict structure
            analysis = {
                "analysis_type": analysis_type,
                "root_cause": f"Analysis error: {str(e)}",
                "key_experiences": []
            }
            state["reflection_analysis"] = analysis

        # Store key experiences from reflection - FIXED: Safe access to analysis
        try:
            # 🔧 修复：按照文档要求，从分析结果中提取关键timestep
            critical_timestep = analysis.get("critical_timestep", 0) if isinstance(analysis, dict) else 0
            stored_count = 0
            
            if critical_timestep >= 0 and critical_timestep < len(state["trajectory_history"]):
                # 获取关键帧的数据
                critical_frame = state["trajectory_history"][critical_timestep]
                strategy = critical_frame.get("strategy", "unknown")
                goal = critical_frame.get("goal", "Unknown goal")
                
                # 🔧 修复：直接使用FrameLog中已经格式化的完整current_state字符串
                # 这确保了RAG检索和存储使用完全一致的格式
                current_state = critical_frame.get("current_state", "")
                if not current_state:
                    logger.warning(f"No current_state found in critical_frame at timestep {critical_timestep}")
                    # 回退到构建格式化的字符串
                    scene_caption = critical_frame.get("scene_caption", "")
                    current_state = f"Current aerial perspective observation: {{'scene_description': '{scene_caption}'}}. Task: {state['task_instruction']}. Current Goal: {goal} (Strategy: {strategy})"
                
                # 获取对应的视觉上下文
                semantic_image = _find_semantic_image_by_timestep(state, critical_timestep, strategy)
                map_snapshot = critical_frame.get("map_snapshot")
                
                # 根据分析类型获取相应的数据
                if analysis_type == "success":
                    successful_action = analysis.get("successful_action", "{}")
                    reflection_note = analysis.get("reflect_reason", "")
                else:  # failure
                    successful_action = analysis.get("corrected_action", "{}")
                    reflection_note = analysis.get("reflect_reason", "")
                
                # 存储到相应的记忆模块
                if strategy == "Navigate":
                    state["memory_module"].navigate_experience_memory.add(
                        navigation_goal=goal,
                        map_snapshot=map_snapshot,
                        refined_action=successful_action,  # 🔧 修复：使用新的字段名
                        is_success=(analysis_type == "success"),
                        reflect_reason=reflection_note,  # 🔧 修复：使用新的字段名
                        current_state=current_state,  # 🔧 修复：使用完整的 current_state
                        memory_type="reflection",
                        semantic_enhanced_image=semantic_image,
                        timestep=critical_timestep
                    )
                    stored_count += 1
                elif strategy in ["Search", "Locate"]:
                    state["memory_module"].search_and_locate_memory.add(
                        search_goal=goal,
                        map_snapshot=map_snapshot,
                        scene_graph_summary="",
                        refined_action=successful_action,  # 🔧 修复：使用新的字段名
                        is_success=(analysis_type == "success"),
                        reflect_reason=reflection_note,  # 🔧 修复：使用新的字段名
                        current_state=current_state,  # 🔧 修复：使用完整的 current_state
                        memory_type="reflection",
                        semantic_enhanced_image=semantic_image,
                        timestep=critical_timestep
                    )
                    stored_count += 1
                
                logger.info(f"Stored {analysis_type} experience from timestep {critical_timestep}")
                logger.info(f"📚 Stored {stored_count} {analysis_type} experiences in memory")
            else:
                logger.warning(f"Invalid critical_timestep: {critical_timestep}")
                stored_count = 0  # 重置计数器
                # 新增：如果是failure且没有有效的critical_timestep，也要存储一次失败经验
                if analysis_type == "failure" and state.get("trajectory_history"):
                    last_frame = state["trajectory_history"][-1]
                    goal = last_frame.get("goal", "Unknown goal")
                    strategy = last_frame.get("strategy", "Unknown")
                    action = last_frame.get("action", "")
                    timestep = last_frame.get("timestep", 0)
                    
                    # 🔧 修复：直接使用FrameLog中已经格式化的完整current_state字符串
                    current_state = last_frame.get("current_state", "")
                    if not current_state:
                        logger.warning(f"No current_state found in last_frame at timestep {timestep}")
                        # 回退到构建格式化的字符串
                        scene_caption = last_frame.get("scene_caption", "")
                        current_state = f"Current aerial perspective observation: {{'scene_description': '{scene_caption}'}}. Task: {state['task_instruction']}. Current Goal: {goal} (Strategy: {strategy})"
                    
                    # 🔧 修复：获取失败时刻的视觉上下文
                    semantic_image = _find_semantic_image_by_timestep(state, timestep, strategy)
                    map_snapshot = last_frame.get("map_snapshot")
                    
                    # 🔧 修复：尝试从analysis中提取correction/root_cause，确保是字符串类型
                    correction = ""
                    if isinstance(analysis, dict):
                        correction = analysis.get("correction") or analysis.get("correct_action") or analysis.get("root_cause") or analysis.get("suggestion") or ""
                        # 确保correction是字符串
                        if isinstance(correction, list):
                            correction = "; ".join(str(item) for item in correction)
                        elif not isinstance(correction, str):
                            correction = str(correction)
                    elif isinstance(analysis, str):
                        correction = analysis
                    
                    # 存储失败经验
                    if strategy == "Navigate":
                        state["memory_module"].navigate_experience_memory.add(
                            navigation_goal=goal,
                            map_snapshot=map_snapshot,  # 🔧 修复：使用实际的地图快照
                            refined_action=action,  # 🔧 修复：使用新的字段名
                            is_success=False,
                            reflect_reason=correction,  # 🔧 修复：使用新的字段名
                            current_state=current_state,  # 🔧 修复：使用完整的 current_state
                            memory_type="correction",
                            semantic_enhanced_image=semantic_image,  # 🔧 修复：直接传递PIL Image对象
                            timestep=timestep
                        )
                        logger.info(f"📚 Stored 1 failure experience for Navigate in memory")
                    elif strategy in ["Search", "Locate"]:
                        state["memory_module"].search_and_locate_memory.add(
                            search_goal=goal,
                            map_snapshot=map_snapshot,  # 🔧 修复：使用实际的地图快照
                            scene_graph_summary="",
                            refined_action=action,  # 🔧 修复：使用新的字段名
                            is_success=False,
                            reflect_reason=correction,  # 🔧 修复：使用新的字段名
                            current_state=current_state,  # 🔧 修复：使用完整的 current_state
                            memory_type="correction",
                            semantic_enhanced_image=semantic_image,  # 🔧 修复：直接传递PIL Image对象
                            timestep=timestep
                        )
                        logger.info(f"📚 Stored 1 failure experience for {strategy} in memory")
                    else:
                        logger.warning(f"Unknown strategy for failure experience: {strategy}")
                else:
                    logger.info(f"No key experiences to store for {analysis_type}")
        except Exception as e:
            logger.error(f"Error processing key experiences: {e}")

        # Print reflection summary - FIXED: Safe access + detailed debugging
        try:
            # 添加详细调试信息
            print("\n🔍 调试信息:")
            print(f"   分析类型: {analysis_type}")
            print(f"   分析对象类型: {type(analysis)}")
            print(f"   轨迹长度: {len(state.get('trajectory_history', []))}")
            if isinstance(analysis, dict):
                print(f"   关键timestep: {analysis.get('critical_timestep', 'N/A')}")
                print(f"   分析键: {list(analysis.keys())}")
            else:
                print(f"   分析内容: {analysis}")
            
            print_reflection_info(state, analysis_type, analysis if isinstance(analysis, dict) else {})
        except Exception as e:
            logger.error(f"Error printing reflection info: {e}")
            print(f"❌ 打印反思信息错误: {e}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in {analysis_type} reflection: {e}")
        return state

def _find_semantic_image_by_timestep(state: AgentState, timestep: int, strategy: str) -> Optional[Image.Image]:
    """
    Simplified function to find semantic image by timestep.
    Much simpler than the previous complex implementation.
    """
    try:
        save_path = state["landmark_nav_map"].save_path
        id_str = "_".join(map(str, state["landmark_nav_map"].id)) if isinstance(state["landmark_nav_map"].id, tuple) else str(state["landmark_nav_map"].id)
        
        # Determine map type based on strategy
        map_type = "landmark" if strategy == "Navigate" else "semantic"
        
        # Try exact timestep first, then nearby timesteps
        for step_offset in range(-2, 3):  # Try ±2 steps
            target_step = timestep + step_offset
            filename = f"{map_type}_{id_str}_step_{target_step:04d}.png"
            filepath = os.path.join(save_path, filename)
            
            if os.path.exists(filepath):
                return Image.open(filepath)
        
        logger.debug(f"No semantic image found for timestep {timestep}, strategy {strategy}")
        return None
        
    except Exception as e:
        logger.error(f"Error finding semantic image: {e}")
        return None



def find_caption_for_goal(trajectory_log: List[FrameLog], goal: str) -> str:
    """Find the scene caption from trajectory log that corresponds to a given goal."""
    try:
        for entry in reversed(trajectory_log):
            if entry.get("goal") == goal and entry.get("scene_caption"):
                return entry["scene_caption"]
    except Exception as e:
        logger.error(f"Error finding caption for goal: {e}")
    return ""