"""
Configuration loader for ReflectionNav Agent V3.
Manages YAML configuration files and provides easy access to settings.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class VLMDetectionConfig:
    """VLM detection configuration."""
    interval: int = 4
    distance_threshold: float = 15.0

@dataclass
class NavigationConfig:
    """Navigation and search configuration."""
    max_search_radius: float = 30.0
    search_threshold: float = 0.05
    adaptive_scale_distance_threshold: float = 80.0
    action_scale: Dict[str, float] = field(default_factory=lambda: {"navigate": 1.0, "search": 3.0})
    view_width_multiplier: float = 2.0

@dataclass
class StrategyConfig:
    """Strategy switching configuration."""
    navigate_timeout: int = 6
    search_timeout: int = 6
    landmark_proximity: float = 30.0
    target_proximity: float = 5.0
    force_locate_on_final_step: bool = True  # 🔧 新增：强制最后一步进行locate

@dataclass
class MemoryConfig:
    """Memory and experience configuration."""
    experience_db_path: str = "experience_database"
    max_retrieval_results: int = 3
    similarity_threshold: float = 0.7

@dataclass
class LLMConfig:
    """LLM and parsing configuration."""
    max_retries: int = 3
    max_tokens: int = 500
    temperature: float = 0.1

@dataclass
class PathsConfig:
    """Path settings configuration."""
    semantic_images_save_dir: str = "V3"
    rgb_images_save_dir: str = "V3" 
    results_save_dir: str = "V3"

@dataclass
class RAGConfig:
    """RAG settings configuration."""
    enable_rag: bool = True
    retrieval_interval: int = 5
    context_window_size: int = 500
    enable_experience_memory: bool = True
    enable_strategic_memory: bool = True

@dataclass
class ReflectionConfig:
    """Reflection settings configuration."""
    enable_reflection: bool = True
    enable_success_reflection: bool = True
    enable_failure_reflection: bool = True
    max_key_experiences: int = 5
    store_strategic_plans: bool = True

@dataclass
class LoggingConfig:
    """Logging and output configuration."""
    level: str = "INFO"
    enable_trajectory_logging: bool = True
    save_semantic_images: bool = True
    save_rgb_images: bool = False

@dataclass
class ExperimentalConfig:
    """Experimental features configuration."""
    enable_adaptive_action_scale: bool = False
    enable_multi_step_planning: bool = False
    enable_scene_graph_caching: bool = True

@dataclass
class AgentConfig:
    """Main configuration class for ReflectionNav Agent V3."""
    vlm_detection: VLMDetectionConfig = field(default_factory=VLMDetectionConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)  # 🔧 新增：路径配置
    rag: RAGConfig = field(default_factory=RAGConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)

class ConfigLoader:
    """Configuration loader for ReflectionNav Agent V3."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config_path = config_path
        self._config: Optional[AgentConfig] = None
    
    def load_config(self) -> AgentConfig:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config
            
        if not os.path.exists(self.config_path):
            print(f"Config file not found at {self.config_path}, using default configuration")
            self._config = AgentConfig()
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Parse configuration sections
            vlm_detection = VLMDetectionConfig(**yaml_data.get('vlm_detection', {}))
            navigation = NavigationConfig(**yaml_data.get('navigation', {}))
            strategy = StrategyConfig(**yaml_data.get('strategy', {}))
            memory = MemoryConfig(**yaml_data.get('memory', {}))
            llm = LLMConfig(**yaml_data.get('llm', {}))
            paths = PathsConfig(**yaml_data.get('paths', {}))  # 🔧 新增：路径配置解析
            rag = RAGConfig(**yaml_data.get('rag', {}))
            reflection = ReflectionConfig(**yaml_data.get('reflection', {}))
            logging_config = LoggingConfig(**yaml_data.get('logging', {}))
            experimental = ExperimentalConfig(**yaml_data.get('experimental', {}))
            
            self._config = AgentConfig(
                vlm_detection=vlm_detection,
                navigation=navigation,
                strategy=strategy,
                memory=memory,
                llm=llm,
                paths=paths,  # 🔧 新增：路径配置
                rag=rag,
                reflection=reflection,
                logging=logging_config,
                experimental=experimental
            )
            
            print(f"Configuration loaded from {self.config_path}")
            return self._config
            
        except Exception as e:
            print(f"Error loading configuration: {e}, using default configuration")
            self._config = AgentConfig()
            return self._config
    
    def get_config(self) -> AgentConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, config: AgentConfig):
        """Save configuration to YAML file."""
        try:
            config_dict = {
                'vlm_detection': {
                    'interval': config.vlm_detection.interval,
                    'distance_threshold': config.vlm_detection.distance_threshold
                },
                'navigation': {
                    'max_search_radius': config.navigation.max_search_radius,
                    'search_threshold': config.navigation.search_threshold,
                    'adaptive_scale_distance_threshold': config.navigation.adaptive_scale_distance_threshold,
                    'action_scale': config.navigation.action_scale,
                    'view_width_multiplier': config.navigation.view_width_multiplier
                },
                'strategy': {
                    'navigate_timeout': config.strategy.navigate_timeout,
                    'search_timeout': config.strategy.search_timeout,
                    'landmark_proximity': config.strategy.landmark_proximity,
                    'target_proximity': config.strategy.target_proximity,
                    'force_locate_on_final_step': config.strategy.force_locate_on_final_step  # 🔧 新增
                },
                'memory': {
                    'experience_db_path': config.memory.experience_db_path,
                    'max_retrieval_results': config.memory.max_retrieval_results,
                    'similarity_threshold': config.memory.similarity_threshold
                },
                'llm': {
                    'max_retries': config.llm.max_retries,
                    'max_tokens': config.llm.max_tokens,
                    'temperature': config.llm.temperature
                },
                'paths': {  # 🔧 新增：路径配置保存
                    'semantic_images_save_dir': config.paths.semantic_images_save_dir,
                    'rgb_images_save_dir': config.paths.rgb_images_save_dir,
                    'results_save_dir': config.paths.results_save_dir
                },
                'rag': {
                    'retrieval_interval': config.rag.retrieval_interval,
                    'context_window_size': config.rag.context_window_size
                },
                'reflection': {
                    'enable_success_reflection': config.reflection.enable_success_reflection,
                    'enable_failure_reflection': config.reflection.enable_failure_reflection,
                    'max_key_experiences': config.reflection.max_key_experiences
                },
                'logging': {
                    'level': config.logging.level,
                    'enable_trajectory_logging': config.logging.enable_trajectory_logging,
                    'save_semantic_images': config.logging.save_semantic_images,
                    'save_rgb_images': config.logging.save_rgb_images
                },
                'experimental': {
                    'enable_adaptive_action_scale': config.experimental.enable_adaptive_action_scale,
                    'enable_multi_step_planning': config.experimental.enable_multi_step_planning,
                    'enable_scene_graph_caching': config.experimental.enable_scene_graph_caching
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            print(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")

# Global config instance
_config_loader = ConfigLoader()

def get_config() -> AgentConfig:
    """Get the global configuration instance."""
    return _config_loader.get_config()

def reload_config() -> AgentConfig:
    """Reload configuration from file."""
    global _config_loader
    _config_loader._config = None
    return _config_loader.load_config() 

def set_config_value(key_path: str, value) -> None:
    """
    Dynamically set a config value using dot notation.
    
    Args:
        key_path: Dot-separated path like 'rag.enable_rag' or 'reflection.enable_reflection'
        value: New value to set
    """
    global _config_loader
    
    config = _config_loader.get_config()
    keys = key_path.split('.')
    
    if len(keys) != 2:
        raise ValueError(f"Invalid key path: {key_path}. Expected format: 'section.key'")
    
    section_name, key_name = keys
    
    # Get the section object
    if hasattr(config, section_name):
        section = getattr(config, section_name)
        
        # Set the value if the attribute exists
        if hasattr(section, key_name):
            setattr(section, key_name, value)
            print(f"Config updated: {key_path} = {value}")
        else:
            raise ValueError(f"Unknown config key: {key_name} in section {section_name}")
    else:
        raise ValueError(f"Unknown config section: {section_name}")

def get_config_value(key_path: str):
    """
    Get a config value using dot notation.
    
    Args:
        key_path: Dot-separated path like 'rag.enable_rag'
    
    Returns:
        The config value
    """
    config = _config_loader.get_config()
    keys = key_path.split('.')
    
    if len(keys) != 2:
        raise ValueError(f"Invalid key path: {key_path}. Expected format: 'section.key'")
    
    section_name, key_name = keys
    
    if hasattr(config, section_name):
        section = getattr(config, section_name)
        if hasattr(section, key_name):
            return getattr(section, key_name)
        else:
            raise ValueError(f"Unknown config key: {key_name} in section {section_name}")
    else:
        raise ValueError(f"Unknown config section: {section_name}") 