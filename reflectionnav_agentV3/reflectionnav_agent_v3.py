"""
ReflectionNav Agent V3: Advanced navigation agent with multi-modal RAG and reflection capabilities.

This module implements the main agent class that orchestrates:
- LangGraph workflow for task execution
- Multi-modal RAG for experience retrieval  
- Reflection-based learning from successes and failures
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os

from langgraph.graph import StateGraph, START, END
from scenegraphnav.agent import Agent, ExplorationAnalyzer
from scenegraphnav.parser import ExperimentArgs as scenegraphnav_ExperimentArgs
from gsamllavanav.parser import ExperimentArgs as gsamllavanav_ExperimentArgs
from scenegraphnav.llm_controller import LLMController
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.space import Pose4D
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap

from reflectionnav_agentV3.agent_state import AgentState
from reflectionnav_agentV3.multimodal_memory import MultiModalMemory, OpenAIEmbeddingProvider
from reflectionnav_agentV3.llm_manager import LLMManager
from reflectionnav_agentV3.reflection import ReflectionAgent
from reflectionnav_agentV3.result_manager import ResultManager
from reflectionnav_agentV3.config_loader import get_config
from reflectionnav_agentV3.graph_nodes import (
    # Core nodes
    perceive_and_model,
    generate_plan_with_rag,
    execute_subtask_router,
    route_to_expert,
    execute_navigate,
    execute_search,
    execute_locate,
    finalize_step,
    reflect_on_success,
    reflect_on_failure,
    # Router functions
    should_perceive,
    should_generate_plan,
    should_continue_mission
)

logger = logging.getLogger(__name__)

# ===============================================
# Main Agent Class  
# ===============================================

class ReflectionNav_AgentV3(Agent):
    """
    Advanced navigation agent with LangGraph workflow, multi-modal RAG, and reflection learning.
    """
    
    def __init__(self, 
                 args: scenegraphnav_ExperimentArgs, 
                 initial_pose: Pose4D, 
                 episode: Episode, 
                 vlmodel, 
                 set_height=None,
                 embedding_provider=OpenAIEmbeddingProvider()):
        """
        Initialize the ReflectionNav Agent V3.
        
        Args:
            args: Experiment arguments
            initial_pose: Starting pose
            episode: Episode data
            vlmodel: Vision-Language model client
            set_height: Optional height override
            embedding_provider: Embedding provider for memory
        """
        super().__init__(args, initial_pose, episode)
        
        try:
            # Validate vlmodel
            if vlmodel is None:
                raise ValueError("vlmodel cannot be None")
                
            # Basic setup
            self.episode = episode
            self.target = self.episode.target_position
            self.set_height = set_height
            if set_height is not None:
                initial_pose.with_z(set_height)
            
            # Load configuration
            self.config = get_config()
            
            # Initialize result manager and get unique episode save path
            self.result_manager = ResultManager(base_output_dir=args.output_dir)
            self.save_path = self.result_manager.initialize_episode_directory(self.episode)
            
            # Initialize tracking variables (similar to original agents)
            self.history = {'decision': [], 'action': [], 'observation': []}
            self.action = ''
            self.observation = ''
            self.plan = None
            self.current_task_index = 0
            self.subtask_status = "pending"
            
            # Use configuration instead of hardcoded values
            self.search_threshold = self.config.navigation.search_threshold
            self.max_search_radius = self.config.navigation.max_search_radius
            self.search_start_pos = None
            self.strategy_distances = {"Start": initial_pose.xy.dist_to(self.episode.target_position.xy)}
            self.trajectory_log = []  # for reflection
            
            # VLM detection control variables from config
            self.last_vlm_detection_time = 0
            self.vlm_detection_interval = self.config.vlm_detection.interval
            self.vlm_detection_distance = self.config.vlm_detection.distance_threshold
            self.last_vlm_detection_position = initial_pose.xy
            self.last_scene_caption = ""
            
            # Initialize core components  
            # Convert scenegraphnav_ExperimentArgs to gsamllavanav_ExperimentArgs for LLMController
            gsamllavanav_args = self._convert_args_for_llm_controller(args)
            self.controller = LLMController(gsamllavanav_args, initial_pose)
            self.model = vlmodel
            
            # Initialize memory system
            memory_db_path = getattr(args, 'multimodal_db_path', None)
            self.memory = MultiModalMemory(embedding_provider, db_path=memory_db_path)
            
            # Initialize reflection agent
            vl_api_key = getattr(args, 'vl_api_key', None)
            base_url = getattr(args, 'base_url', None)
            self.reflection_agent = ReflectionAgent(vl_api_key=vl_api_key, base_url=base_url)
            
            # Initialize prompts and LLM manager
            self.prompts = self._init_prompts()
            
            # Initialize LLM manager with validation
            if not self.prompts:
                raise ValueError("Failed to initialize prompts")
            
            self.llm_manager = LLMManager(self.model, self.prompts)
            
            # Initialize landmark navigation map - save_path is now semantic_images dir
            self.landmark_nav_map = LandmarkNavMap(
                episode.map_name, args.map_shape, args.map_pixels_per_meter,
                episode.description_landmarks, episode.description_target, 
                episode.description_surroundings, args.gsam_params, 
                id=episode.id, save_path=self.result_manager.semantic_dir
            )
            
            # Initialize exploration analyzer
            self.exploration_analyzer = ExplorationAnalyzer(threshold=1e-8)
            self.previous_sem_map = np.zeros((*self.landmark_nav_map.shape, 3), dtype=np.float32)
            
            # Build the LangGraph
            self.graph = self._build_graph()
            self.app = self.graph.compile()
            
            # Results tracking
            self.results = {
                "target": (self.target.x, self.target.y),
                "steps": []
            }
            
            logger.info("ReflectionNav_AgentV3 initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during ReflectionNav_AgentV3 initialization: {e}")
            raise RuntimeError(f"Failed to initialize ReflectionNav_AgentV3: {e}") from e
    
    def _convert_args_for_llm_controller(self, args: scenegraphnav_ExperimentArgs) -> gsamllavanav_ExperimentArgs:
        """Convert scenegraphnav ExperimentArgs to gsamllavanav ExperimentArgs for LLMController compatibility."""
        return gsamllavanav_ExperimentArgs(
            output_dir=args.output_dir,  # 🔧 Fixed: Add missing output_dir parameter
            seed=args.seed,
            mode=args.mode,
            model=args.model,
            log=args.log,
            silent=args.silent,
            resume_log_id=args.resume_log_id,
            map_size=args.map_size,
            map_meters=args.map_meters,
            map_update_interval=args.map_update_interval,
            max_depth=args.max_depth,
            altitude=args.altitude,
            ablate=args.ablate,
            alt_env=args.alt_env,
            gsam_rgb_shape=args.gsam_rgb_shape,
            gsam_use_segmentation_mask=args.gsam_use_segmentation_mask,
            gsam_use_bbox_confidence=args.gsam_use_bbox_confidence,
            gsam_use_map_cache=args.gsam_use_map_cache,
            gsam_box_threshold=args.gsam_box_threshold,
            gsam_text_threshold=args.gsam_text_threshold,
            gsam_max_box_size=args.gsam_max_box_size,
            gsam_max_box_area=args.gsam_max_box_area,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            epochs=args.epochs,
            checkpoint=args.checkpoint,
            save_every=args.save_every,
            train_trajectory_type=args.train_trajectory_type,
            train_episode_sample_size=args.train_episode_sample_size,
            eval_every=args.eval_every,
            eval_batch_size=args.eval_batch_size,
            eval_at_start=args.eval_at_start,
            eval_max_timestep=args.eval_max_timestep,
            eval_client=args.eval_client,
            success_dist=args.success_dist,
            success_iou=args.success_iou,
            move_iteration=args.move_iteration,
            progress_stop_val=args.progress_stop_val,
            eval_goal_selector=args.eval_goal_selector,
            gps_noise_scale=args.gps_noise_scale,
            sim_ip=args.sim_ip,
            sim_port=args.sim_port
        )
    
    def _init_prompts(self) -> Dict[str, str]:
        """Initialize all task-related prompt templates using PromptManager."""
        from reflectionnav_agentV3.prompt_manager import get_prompt_templates
        
        # Use the unified PromptManager system
        return get_prompt_templates()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph execution flow."""
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("perceive_and_model", perceive_and_model)
        workflow.add_node("generate_plan_with_rag", generate_plan_with_rag)
        workflow.add_node("execute_subtask_router", execute_subtask_router)
        workflow.add_node("execute_navigate", execute_navigate)
        workflow.add_node("execute_search", execute_search)
        workflow.add_node("execute_locate", execute_locate)
        workflow.add_node("finalize_step", finalize_step)
        workflow.add_node("reflect_on_success", reflect_on_success)
        workflow.add_node("reflect_on_failure", reflect_on_failure)
        
        # Set entry point
        workflow.set_entry_point("perceive_and_model")
        
        # Main flow: perceive -> plan check -> subtask execution -> finalize -> continue check
        workflow.add_conditional_edges(
            "perceive_and_model",
            should_generate_plan,
            {
                "generate_plan": "generate_plan_with_rag",
                "execute_subtask": "execute_subtask_router"
            }
        )
        
        # Add edge for plan generation to subtask execution
        workflow.add_edge("generate_plan_with_rag", "execute_subtask_router")
        
        # Add conditional edges from subtask router to experts
        workflow.add_conditional_edges(
            "execute_subtask_router", 
            route_to_expert,  # This returns the strategy
            {
                "execute_navigate": "execute_navigate",
                "execute_search": "execute_search", 
                "execute_locate": "execute_locate",
                "finalize_step": "finalize_step"  # Handle error cases
            }
        )
        
        # Add edges from experts to finalize_step
        workflow.add_edge("execute_navigate", "finalize_step")
        workflow.add_edge("execute_search", "finalize_step")
        workflow.add_edge("execute_locate", "finalize_step")
        
        # Add conditional edge from finalize_step for mission continuation
        workflow.add_conditional_edges(
            "finalize_step",
            should_continue_mission,
            {
                "continue": "perceive_and_model",  # Loop back to perception
                "reflect_success": "reflect_on_success",
                "reflect_failure": "reflect_on_failure"
            }
        )
        
        # Add end edges for reflection nodes
        workflow.add_edge("reflect_on_success", END)
        workflow.add_edge("reflect_on_failure", END)
        
        return workflow
    
    def _create_initial_state(self) -> AgentState:
        """Create the initial state for the agent."""
        # 🔧 Enhanced: Calculate target location information for task instruction
        import math
        current_pos = self.controller.pose.xy
        target_pos = self.episode.target_position.xy
        distance_to_target = current_pos.dist_to(target_pos)
        
        # Calculate direction to target
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        
        # Map to cardinal direction
        if 22.5 <= angle < 67.5:
            target_direction = "Northeast"
        elif 67.5 <= angle < 112.5:
            target_direction = "North"
        elif 112.5 <= angle < 157.5:
            target_direction = "Northwest"
        elif 157.5 <= angle < 202.5:
            target_direction = "West"
        elif 202.5 <= angle < 247.5:
            target_direction = "Southwest"
        elif 247.5 <= angle < 292.5:
            target_direction = "South"
        elif 292.5 <= angle < 337.5:
            target_direction = "Southeast"
        else:
            target_direction = "East"
        
        # Enhanced task instruction with global target location
        enhanced_task_instruction = (
            f"{self.episode.target_description}. "
            f"Final Target Location: Approximately {distance_to_target:.1f} meters to your {target_direction}."
        )
        
        return {
            # Static components
            "controller": self.controller,
            "save_path": self.save_path,
            "landmark_nav_map": self.landmark_nav_map,
            "episode": self.episode,
            "args": self.args,
            "task_instruction": enhanced_task_instruction,  # 🔧 Enhanced: Include target location info
            "memory_module": self.memory,
            "llm_manager": self.llm_manager,
            "reflection_agent": self.reflection_agent,
            "exploration_analyzer": self.exploration_analyzer,
            "result_manager": self.result_manager,
            
            # Dynamic state - initialize with our instance variables
            "plan": self.plan,  # Initially None
            "current_task_index": self.current_task_index,  # Initially 0
            "timestep": 0,
            "current_observation": {},
            "current_state": "",
            "subtask_context": {},
            "trajectory_history": [],
            
            # Mission control - use our instance variables
            "subtask_status": self.subtask_status,  # Initially "pending"
            "search_start_pos": self.search_start_pos,  # Initially None
            "search_threshold": self.search_threshold,  # 0.05
            "max_search_radius": self.max_search_radius,  # 30.0
            "switch_time": 0,
            "locate_completed": None,  # 🔧 NEW: Initially None, set to True when locate completes
            
            # VLM detection control - use our instance variables
            "last_vlm_detection_time": self.last_vlm_detection_time,  # 0
            "vlm_detection_interval": self.vlm_detection_interval,  # 4
            "vlm_detection_distance": self.vlm_detection_distance,  # 15.0
            "last_vlm_detection_position": self.last_vlm_detection_position,
            
            # 🔧 New: Memory retrieval tracking for interval-based RAG
            "last_memory_retrieval_timestep": -1,  # Initialize to -1 so first step always retrieves
            
            # ===== LLM I/O logging =====
            "last_llm_system_prompt": None,  # The last system prompt sent to LLM
            "last_llm_prompt": None,         # The last user prompt sent to LLM
            
            # Final result
            "mission_success": False,
            "final_reason": "",
            
            # Working variables - ensure proper types
            "previous_sem_map": self.previous_sem_map,
            "last_scene_caption": self.last_scene_caption,  # Initially ""
            "strategy_distances": {k: float(v) for k, v in self.strategy_distances.items()},  # Ensure all values are float
            "view_width": 0.0,
            "xyxy": [],
            
            # Additional working variables
            "current_strategy": None,
            "next_position": None,
            "action_result": None,
            "cot_summary": None,
            
            # Search Decision Variables
            "search_decision": None,
            "search_confidence": None,
            
            # Navigation Decision Variables (required by AgentState)
            "navigation_strategy_decision": None,
            "navigation_confidence": None,
            
            # Reflection Variables  
            "reflection_analysis": None
        }
    
    def run(self) -> tuple[bool, List[Pose4D]]:
        """
        Run the agent using LangGraph execution.
        
        Returns:
            Tuple of (success, position_log)
        """
        logger.info("Starting ReflectionNav_AgentV3 execution...")
        
        # Create initial state
        initial_state = self._create_initial_state()
        
        # Log the episode directory that was already created in __init__
        if self.result_manager:
            episode_dir = self.result_manager.get_episode_directory()
            if episode_dir:
                logger.info(f"Using episode directory: {episode_dir}")
            else:
                logger.warning("No episode directory available")
        
        try:
            # Execute the graph with increased recursion limit
            final_state = self.app.invoke(
                initial_state,
                config={"recursion_limit": 100}  # Increase from default 25 to 100
            )
            
            # Extract results - ensure final_state is treated as AgentState
            success = final_state.get("mission_success", False)
            timestep = final_state.get("timestep", 0)
            
            # Build position log from trajectory
            pos_log = []
            for frame in final_state.get("trajectory_history", []):
                # Reconstruct poses from trajectory (this is a simplified approach)
                # In practice, you might want to store poses explicitly
                pos_log.append(self.controller.pose)
            
            # Store final position
            pos_log.append(self.controller.pose)
            
            # Update results for saving - cast to AgentState type
            self._update_results_from_state(final_state)
            
            # Finalize episode with result manager (optional)
            if self.result_manager:
                try:
                    final_distance = float(self.controller.pose.xy.dist_to(self.episode.target_position.xy))
                    self.result_manager.finalize_episode(success, timestep, final_distance)
                except Exception as e:
                    logger.error(f"Error finalizing episode: {e}")
            
            # Save results
            self.save_results(success)
            
            logger.info(f"Mission completed. Success: {success}, Steps: {timestep}")
            return success, pos_log
            
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            # Cleanup incomplete episode (optional)
            if self.result_manager:
                try:
                    self.result_manager.cleanup_incomplete_episode()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {cleanup_error}")
            
            self.save_results(False)
            return False, [self.controller.pose]
    
    def _update_results_from_state(self, final_state: Dict[str, Any]):
        """Update results dictionary from final state."""
        for i, frame in enumerate(final_state.get("trajectory_history", [])):
            # 🔧 Fixed: Use pose and distance from trajectory record, not final pose
            if "pose" in frame and "distance_to_target" in frame:
                # Use recorded pose and distance
                pose = frame["pose"]
                distance_to_target = frame["distance_to_target"]
            else:
                # Fallback to current pose (for backward compatibility)
                pose = (
                    self.controller.pose.x, self.controller.pose.y,
                    self.controller.pose.z, self.controller.pose.yaw
                )
                distance_to_target = self.controller.pose.xy.dist_to(self.target.xy)
            
            step_data = {
                "time_step": frame.get("timestep", i),  # 🔧 Use recorded timestep
                "pose": pose,
                "distance_to_target": distance_to_target,  # 🔧 Use recorded distance
                "goal": frame.get("goal", ""),
                "strategy": frame.get("strategy", ""),
                "action": frame.get("action", ""),
                "scene_caption": frame.get("scene_caption", ""),
                "observation_suggestion": frame.get("scene_graph_summary", "")
            }
            self.results["steps"].append(step_data)
    
    def save_results(self, success: bool):
        """Save mission results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ReflectionNavAgentV3_{self.episode.id}_{timestamp}.json"
        filepath = os.path.join(self.save_path, filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Add success status and context information
        self.results["success"] = success
        self.results["episode_id"] = self.episode.id
        self.results["target_description"] = self.episode.target_description
        self.results["agent_version"] = "ReflectionNav_AgentV3"
        self.results["framework"] = "LangGraph + LangChain"
        
        # Save results to file
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"Results saved to {filepath}")

    @property
    def state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging."""
        return {
            "pose": (self.controller.pose.x, self.controller.pose.y, 
                    self.controller.pose.z, self.controller.pose.yaw),
            "target": (self.target.x, self.target.y),
            "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
            "episode_id": self.episode.id,
            "target_description": self.episode.target_description
        }


# ==============================================================================================
# Factory function for backward compatibility
# ==============================================================================================

def create_reflectionnav_agent_v3(args: scenegraphnav_ExperimentArgs,
                                  initial_pose: Pose4D,
                                  episode: Episode,
                                  vlmodel,
                                  set_height=None,
                                  embedding_provider=None) -> ReflectionNav_AgentV3:
    """
    Factory function to create ReflectionNav_AgentV3 instance.
    
    This provides a clean interface for creating the agent and maintains
    backward compatibility with existing code.
    """
    if embedding_provider is None:
        embedding_provider = OpenAIEmbeddingProvider()
    
    return ReflectionNav_AgentV3(
        args=args,
        initial_pose=initial_pose,
        episode=episode,
        vlmodel=vlmodel,
        set_height=set_height,
        embedding_provider=embedding_provider
    ) 