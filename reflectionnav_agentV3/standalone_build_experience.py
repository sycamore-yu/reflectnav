#!/usr/bin/env python3
"""
独立的经验构建脚本，避免导入有依赖问题的模块
"""

import os
import json
import glob
import argparse
import logging
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
from typing import Optional, Dict, Any, List

# 直接导入需要的模块，避免导入整个包
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_memory import MultiModalMemory, OpenAIEmbeddingProvider
from reflection import ReflectionAgent
from config_loader import get_config
from utils import encode_image_to_base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_trajectory_for_llm(steps: list) -> list:
    """
    Formats the trajectory steps into a readable string for the LLM prompt.
    Updated for V3 format with enhanced step information.
    """
    formatted_log = []
    for step in steps:
        # Handle both V2 and V3 format
        if 'plan' in step:
            # V2 format
            plan = step.get('plan', {})
            log_entry = (
                f"Strategy: {plan.get('strategy', 'N/A')}, "
                f"Goal: {plan.get('goal', 'N/A')}, "
                f"Action: {step.get('action_suggestion', 'N/A')}"
            )
        else:
            # V3 format
            log_entry = (
                f"Strategy: {step.get('strategy', 'N/A')}, "
                f"Goal: {step.get('goal', 'N/A')}, "
                f"Action: {step.get('action', 'N/A')}, "
                f"Scene: {step.get('scene_caption', 'N/A')[:100]}..."  # Truncate long captions
            )
        formatted_log.append(log_entry)
    return formatted_log

def build_experience_from_trajectory(results_dir: str, db_path: Optional[str] = None):
    """
    🔧 重构：使用现有的反思方法构建初始经验数据库
    
    Args:
        results_dir: 结果文件目录
        db_path: 数据库路径（可选，默认使用配置文件中的路径）
    """
    if not os.path.isdir(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return 0

    # 🔧 修复：使用配置文件管理数据库路径
    if db_path is None:
        config = get_config()
        db_path = config.memory.experience_db_path
        logger.info(f"Using database path from config: {db_path}")

    logger.info("Initializing agents and memory...")
    embedding_provider = OpenAIEmbeddingProvider()
    memory = MultiModalMemory(embedding_provider, db_path=db_path)
    reflection_agent = ReflectionAgent()

    json_files = glob.glob(os.path.join(results_dir, '**/*.json'), recursive=True)
    logger.info(f"Found {len(json_files)} potential result files to analyze.")

    processed_files = 0
    successful_files = 0
    
    for file_path in tqdm(json_files, desc="Processing result files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read or parse {file_path}: {e}")
            continue

        # 🔧 修复：检查任务成功状态 - 支持多种格式，暂时处理所有任务
        is_success = data.get("success", False) or data.get("mission_success", False)
        logger.info(f"File: {os.path.basename(file_path)}, Success: {is_success}")
        # 暂时处理所有任务，包括失败的，以便构建经验库
        # if not is_success:
        #     logger.debug(f"Skipping unsuccessful run: {file_path}")
        #     continue

        # Extract all necessary info directly from the result file - 支持多种格式
        target_description = data.get("target_description") or data.get("goal") or "Unknown target"
        master_plan = data.get("master_plan") or data.get("plan")
        steps = data.get("steps") or data.get("trajectory")
        agent_version = data.get("agent_version", "ReflectionNavV3")

        if not all([target_description, steps]):
            logger.warning(f"Result file {file_path} is missing required data (target_description or steps). Skipping.")
            continue

        logger.info(f"Processing {agent_version} result file: {os.path.basename(file_path)}")
        processed_files += 1
        
        # 🔧 修复：使用现有的反思方法存储主规划经验
        if master_plan:
            try:
                # 使用现有的主规划反思方法
                master_plan_analysis = reflection_agent.analyze_main_plan(
                    task_instruction=target_description,
                    master_plan=master_plan,
                    trajectory_history=steps,
                    final_reason="success"
                )
                
                # 🔧 修复：使用通用的JSON序列化工具函数
                from utils import safe_json_dumps, convert_numpy_for_json
                
                safe_master_plan = convert_numpy_for_json(master_plan)
                safe_final_outcome = convert_numpy_for_json({
                    "status": "success", 
                    "total_steps": len(steps),
                    "agent_version": agent_version,
                    "file_source": os.path.basename(file_path),
                    "reflection_note": master_plan_analysis.get("reflection_note", ""),
                    "plan_refined": master_plan_analysis.get("plan_refined", "")
                })
                
                memory.mainplan_experience_memory.add(
                    task_instruction=target_description,
                    master_plan=safe_json_dumps(safe_master_plan),
                    final_outcome=safe_final_outcome
                )
                logger.debug(f"Stored strategic plan for: {target_description}")
            except Exception as e:
                logger.error(f"Error storing strategic plan for {file_path}: {e}")

        # 🔧 修复：根据任务成功状态选择反思方法
        try:
            # Format trajectory for LLM analysis
            trajectory_log = format_trajectory_for_llm(steps)
            
            if is_success:
                # 使用现有的成功反思方法
                analysis = reflection_agent.analyze_success(
                    goal_instruction=target_description,
                    master_plan=master_plan or {"sub_goals": [{"strategy": "unknown", "goal": target_description}]},
                    trajectory_log=trajectory_log
                )
            else:
                # 使用失败反思方法
                final_reason = data.get("final_reason", "unknown failure")
                analysis = reflection_agent.analyze_failure(
                    trajectory_log=trajectory_log,
                    failure_reason=final_reason,
                    master_plan=master_plan
                )
            
            if not analysis:
                logger.warning(f"LLM analysis returned no results for {file_path}. Skipping experience generation.")
                continue
                
            # 🔧 修复：使用新的字段格式存储经验
            critical_timestep = analysis.get("critical_timestep", 0)
            if critical_timestep >= 0 and critical_timestep < len(steps):
                # 获取关键帧的数据
                critical_step = steps[critical_timestep]
                strategy = critical_step.get("strategy", "unknown")
                goal = critical_step.get("goal", "Unknown goal")
                
                # 🔧 修复：优先使用已有的格式化current_state字符串，确保格式一致性
                current_state = critical_step.get("current_state", "")
                if not current_state:
                    logger.warning(f"No current_state found in critical_step at timestep {critical_timestep}")
                    # 回退到构建格式化的字符串
                    scene_caption = critical_step.get("scene_caption", "")
                    scene_description = scene_caption or "No scene description available"
                    geoinstruct = critical_step.get("geoinstruct", "No geographical context available")
                    task_instruction = target_description
                    current_state = f"Current aerial perspective observation: {{'scene_description': '{scene_description}', 'geoinstruct': '{geoinstruct}'}}. Task: {task_instruction}. Current Goal: {goal} (Strategy: {strategy})"
                
                # 获取对应的视觉上下文
                map_snapshot = None
                semantic_enhanced_image = None
                
                try:
                    # 获取RGB图像作为地图快照
                    episode_dir = os.path.dirname(file_path)
                    rgb_img_path = os.path.join(episode_dir, "rgb_images", f"rgb_timestep_{critical_timestep:04d}.png")
                    if os.path.exists(rgb_img_path):
                        map_snapshot = Image.open(rgb_img_path)
                    
                    # 获取语义增强图像
                    map_type = "landmark" if strategy == "Navigate" else "semantic"
                    episode_name = os.path.basename(episode_dir)
                    semantic_img_path = os.path.join(episode_dir, "semantic_images", f"{map_type}_{episode_name.replace('ReflectionNavV3_', '')}_step_{critical_timestep:04d}.png")
                    if os.path.exists(semantic_img_path):
                        semantic_enhanced_image = Image.open(semantic_img_path)
                    else:
                        # 尝试替代命名模式
                        semantic_img_path = os.path.join(episode_dir, "semantic_images", f"{map_type}_birmingham_block_1_16_4_step_{critical_timestep:04d}.png")
                        if os.path.exists(semantic_img_path):
                            semantic_enhanced_image = Image.open(semantic_img_path)
                            
                except Exception as e:
                    logger.warning(f"Failed to load visual context for experience: {e}")
                
                # 存储经验到相应的记忆模块
                try:
                    # 根据成功/失败状态选择字段
                    if is_success:
                        action_field = analysis.get("successful_action", "{}")
                        memory_type = "experience"
                    else:
                        action_field = analysis.get("corrected_action", "{}")
                        memory_type = "reflection_note"
                    
                    if strategy == 'Navigate':
                        memory.navigate_experience_memory.add(
                            navigation_goal=goal,
                            map_snapshot=map_snapshot,
                            refined_action=action_field,  # 🔧 修复：根据状态选择字段
                            is_success=is_success,
                            reflect_reason=analysis.get("reflect_reason", ""),  # 🔧 修复：使用新的字段名
                            current_state=current_state,  # 🔧 修复：使用完整的 current_state
                            memory_type=memory_type,
                            semantic_enhanced_image=semantic_enhanced_image,
                            timestep=critical_timestep
                        )
                    elif strategy in ['Search', 'Locate']:
                        memory.search_and_locate_memory.add(
                            search_goal=goal,
                            map_snapshot=map_snapshot,
                            scene_graph_summary="",
                            refined_action=action_field,  # 🔧 修复：根据状态选择字段
                            is_success=is_success,
                            reflect_reason=analysis.get("reflect_reason", ""),  # 🔧 修复：使用新的字段名
                            current_state=current_state,  # 🔧 修复：使用完整的 current_state
                            memory_type=memory_type,
                            semantic_enhanced_image=semantic_enhanced_image,
                            timestep=critical_timestep
                        )
                    
                    successful_files += 1
                    logger.info(f"Stored {strategy} experience from timestep {critical_timestep}")
                    
                except Exception as e:
                    logger.error(f"Error storing experience: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    logger.info(f"Experience building complete. Processed {processed_files} files, stored {successful_files} experiences.")
    return successful_files

def validate_experience_db(db_path: Optional[str] = None):
    """
    🔧 重构：验证经验数据库
    
    Args:
        db_path: 数据库路径（可选，默认使用配置文件中的路径）
    """
    # 🔧 修复：使用配置文件管理数据库路径
    if db_path is None:
        config = get_config()
        db_path = config.memory.experience_db_path
        logger.info(f"Using database path from config: {db_path}")
    
    try:
        embedding_provider = OpenAIEmbeddingProvider()
        memory = MultiModalMemory(embedding_provider, db_path=db_path)
        
        # 获取内存统计
        stats = memory.get_memory_stats()
        
        logger.info("Experience Database Validation Results:")
        logger.info("=" * 50)
        
        total_experiences = 0
        for collection_name, count in stats.items():
            logger.info(f"{collection_name}: {count} entries")
            total_experiences += count
        
        logger.info("=" * 50)
        logger.info(f"Total experiences: {total_experiences}")
        
        if total_experiences > 0:
            logger.info("✅ Experience database validation successful")
            return True
        else:
            logger.warning("⚠️  Experience database is empty")
            return False
            
    except Exception as e:
        logger.error(f"❌ Experience database validation failed: {e}")
        return False

def main():
    """主函数：构建经验数据库"""
    parser = argparse.ArgumentParser(description="Build experience database from trajectory files")
    parser.add_argument("--results_dir", type=str, help="Directory containing result files")
    parser.add_argument("--db_path", type=str, help="Database path (optional, uses config default)")
    parser.add_argument("--validate", action="store_true", help="Validate existing database")
    
    args = parser.parse_args()
    
    if args.validate:
        logger.info("Validating experience database...")
        validate_experience_db(args.db_path)
        return
    
    if not args.results_dir:
        logger.error("--results_dir is required when not using --validate")
        return
    
    logger.info("Building experience database...")
    
    # 构建成功经验
    successful_experiences = build_experience_from_trajectory(args.results_dir, args.db_path)
    
    # 验证数据库
    logger.info("Validating final database...")
    validate_experience_db(args.db_path)

if __name__ == "__main__":
    main() 