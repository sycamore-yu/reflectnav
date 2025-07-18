"""
State definitions for ReflectionNav_AgentV3 using TypedDict for LangGraph compatibility.
Based on the flowchart specification in the project rules.
"""

from typing import TypedDict, Optional, List, Dict, Any

from torch import embedding
from gsamllavanav.space import Pose4D, Point2D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.llm_controller import LLMController
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from reflectionnav_agentV3.multimodal_memory import MultiModalMemory,OpenAIEmbeddingProvider
from reflectionnav_agentV3.llm_manager import LLMManager
from reflectionnav_agentV3.reflection import ReflectionAgent
from reflectionnav_agentV3.result_manager import ResultManager
from scenegraphnav.agent import ExplorationAnalyzer
from scenegraphnav.parser import ExperimentArgs as scenegraphnav_ExperimentArgs

class FrameLog(TypedDict):
    """Log entry for each frame in the trajectory."""
    goal: str
    strategy: str
    current_state: str
    action: str
    scene_caption: str
    scene_graph_summary: str
    map_snapshot: Any  # PIL Image or base64 string
    timestep: int  # Timestep when this frame was recorded
    pose: tuple  # (x, y, z, yaw) - Current pose at this timestep
    distance_to_target: float  # Distance to target at this timestep
    llm_system_prompt: str  # The system prompt sent to the LLM
    llm_prompt: str       # The user prompt sent to the LLM

class AgentState(TypedDict):
    """
    The central data hub for the ReflectionNav Agent.
    Based on the flowchart specification.
    """
    
    # Core components (static during episode)
    controller: LLMController
    save_path: str
    landmark_nav_map: LandmarkNavMap
    episode: Episode
    args: Any  # ExperimentArgs
    task_instruction: str
    memory_module: MultiModalMemory
    llm_manager: LLMManager
    reflection_agent: ReflectionAgent
    exploration_analyzer: ExplorationAnalyzer
    result_manager: Optional[Any]  # 简化为可选字段
    
    # ===== Dynamic State: Updated During Mission =====
    plan: Optional[Dict[str, Any]]  # High-level plan, generated once
    current_task_index: int  # Pointer to current sub-task in plan
    timestep: int  # Current step in simulation
    current_observation: Dict[str, Any]  # {'image': ndarray, 'description': str}
    current_state: str  # State snapshot for RAG & Reflection: location, observation_description, task_instruction, subtask_goal
    subtask_context: Dict[str, Any]  # Expert handoff memo: {goal, CoT_summary, action} from previous step
    trajectory_history: List[FrameLog]  # Logs each frame's {current_state, CoT, action} for reflection
    
    # ===== Mission Control Variables =====
    subtask_status: str  # pending/running/completed
    search_start_pos: Optional[Point2D]  # Search starting position
    search_threshold: float  # Information gain threshold
    max_search_radius: float  # Maximum search radius in meters
    switch_time: int  # Time when strategy switched
    locate_completed: Optional[bool]  # 🔧 NEW: Flag to indicate locate operation completion
    
    # ===== LLM I/O logging =====
    last_llm_system_prompt: Optional[str] # The last system prompt sent to LLM
    last_llm_prompt: Optional[str]        # The last user prompt sent to LLM
    
    # ===== VLM Detection Control =====
    last_vlm_detection_time: int
    vlm_detection_interval: int
    vlm_detection_distance: float
    last_vlm_detection_position: Point2D
    
    # ===== Memory Retrieval Control =====
    last_memory_retrieval_timestep: int  # 🔧 Added: Track last memory retrieval for interval-based RAG
    
    # ===== Final Result: Set at Mission End =====
    mission_success: bool
    final_reason: str  # e.g., 'success', 'timeout', 'goal_unreachable'
    
    # ===== Working Variables =====
    previous_sem_map: Any  # Previous semantic map for comparison
    last_scene_caption: str  # Last generated scene caption
    strategy_distances: Dict[str, float]  # Distance tracking for different strategies
    view_width: float  # Current view width based on altitude
    xyxy: List[tuple]  # Current viewing area coordinates
    
    # ===== Additional Working Variables =====
    current_strategy: Optional[str]  # Current strategy being executed
    next_position: Optional[Point2D]  # Next position to move to
    action_result: Optional[Dict[str, Any]]  # Result from expert execution
    cot_summary: Optional[str]  # Chain of thought summary
    
    # ===== Search Decision Variables =====
    search_decision: Optional[str]  # Search continuation decision: 'continue_search' or 'execute_locate'
    search_confidence: Optional[float]  # Confidence in search decision
    
    # ===== Navigation Decision Variables =====
    navigation_strategy_decision: Optional[str]  # Navigation strategy decision: 'continue_navigate' or 'next_subtask'
    navigation_confidence: Optional[float]  # Confidence in navigation decision
    
    # ===== Reflection Variables =====
    reflection_analysis: Optional[Dict[str, Any]]  # Reflection analysis results

class PerceptionResult(TypedDict):
    """Result from perception node."""
    rgb_image: Any
    image_b64: str
    scene_caption: str
    geoinstruct: str

class PlanResult(TypedDict):
    """Result from planning node."""
    plan: Dict[str, Any]
    current_task_index: int
    subtask_context: Dict[str, Any]

class SubtaskResult(TypedDict):
    """Result from subtask execution."""
    next_position: Point2D
    action: str
    observation: str
    cot_summary: str
    
class ReflectionResult(TypedDict):
    """Result from reflection process."""
    analysis: Dict[str, Any]
    stored_experiences: List[Dict[str, Any]] 