import chromadb
import numpy as np
from PIL import Image
import sys
import os
import base64
from io import BytesIO
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_dashscope import DashScopeEmbeddings
import torch
from typing import Optional, Dict, Any
import logging
from abc import ABC, abstractmethod
from reflectionnav_agentV3.utils import encode_image_to_base64
from reflectionnav_agentV3.prompt_manager import prompt_manager
from reflectionnav_agentV3.llm_manager import LLMManager
import uuid  # 🔧 新增：导入uuid库用于生成唯一ID
# ================== Embedding Generation (Placeholder) ==================
# We assume a CLIP model for multimodal embeddings.
# In a real implementation, this might be a more sophisticated model or a service call.

class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI models"""
    
    def __init__(self):
        # # Load Qwen3-Embedding model
        # self.model_name = "Qwen/Qwen3-Embedding-4B"
        # # 移除弃用的device参数
        # model_kwargs={'device': 'cpu'} 
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # encode_kwargs = {'normalize_embeddings': False}
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=self.model_name,
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
        )
        # Initialize OpenLLM client for image captioning
        self.image_captioner = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE")
        )
        print(f"api_key: {os.getenv('DASHSCOPE_API_KEY')}")
        print(f"api_base: {os.getenv('DASHSCOPE_API_BASE')}")
        self.logger = logging.getLogger(__name__)
        # Removed the problematic logger.info call

    def generate_image_caption(self, image: Image.Image) -> str:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Generate caption 
        # 🔧 Fixed: Use prompt_manager to get scene_caption template
        scene_caption_template = prompt_manager.format_prompt("scene_caption")
        response = self.image_captioner.chat.completions.create(
            model="qwen-vl-max",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": scene_caption_template},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }]
        )
        return response.choices[0].message.content or ""

    def get_text_embedding(self, text: str):
        return self.embeddings.embed_query(text)

    def get_image_embedding(self, image: Image.Image):
        caption = self.generate_image_caption(image)
        return self.get_text_embedding(caption)

    def get_multimodal_embedding(self, text: str, image: Optional[Image.Image], image_caption: str = ""):
        """
        Generate multimodal embedding by combining text and image information.
        
        Args:
            text: Text to embed
            image: Optional image to embed
            image_caption: Pre-generated image caption (to avoid redundant LLM calls)
        """
        try:
            # Use provided caption or fallback to simple description
            if image_caption:
                combined_text = f"{text} [Image: {image_caption}]"
            elif image is not None:
                combined_text = f"{text} [Image: visual content available]"
            else:
                combined_text = text
                
            return self.get_text_embedding(combined_text)
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal embedding: {e}")
            # Fallback to text-only embedding
            return self.get_text_embedding(text)



def _decode_image_from_base64(b64_string: Optional[str]) -> Optional[Image.Image]:
    """Decodes a base64 string to a PIL image."""
    if not b64_string:
        return None
    try:
        image_bytes = base64.b64decode(b64_string)
        image_buffer = BytesIO(image_bytes)
        return Image.open(image_buffer)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to decode base64 to image: {e}")
        return None

# ================== Base Memory Class ==================

class ExperienceMemory(ABC):
    """Base abstract class for all memory types.
    
    Defines the interface for adding and retrieving memory entries.
    All memory types must implement the add and retrieve methods.
    """
    def __init__(self, client, collection_name: str, embedding_provider: OpenAIEmbeddingProvider):
        self.client = client
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_provider = embedding_provider
        self.memory_counter = self.collection.count()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def add(self, *args, **kwargs):
        """Add a new entry to memory.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, *args, **kwargs):
        """Retrieve entries from memory based on query.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            List of retrieved memory entries.
        """
        raise NotImplementedError

# ================== Specialized Memory Stores ==================

class MainplanExperienceMemory(ExperienceMemory):
    """Memory for storing and retrieving strategic planning information."""
    def __init__(self, client, embedding_provider):
        super().__init__(client, "mainplan_experience_memory", embedding_provider)

    def add(self, task_instruction: str, master_plan: str, final_outcome: dict):
        """Add a strategic plan to memory."""
        try:
            # Use text embedding for strategic plan
            embedding = self.embedding_provider.get_text_embedding(task_instruction)
            
            # Store the master plan
            metadata = {
                "task_instruction": task_instruction,
                "master_plan": master_plan,
                "reflection_note": final_outcome.get("reflection_note", ""),
                "plan_refined": final_outcome.get("plan_refined", ""),
                "status": final_outcome.get("status", "unknown")
            }
            
            # 🔧 修复：使用UUID生成唯一ID，避免重复写入
            doc_id = str(uuid.uuid4())

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[f"Strategic plan for: {task_instruction}"]
            )
        except Exception as e:
            print(f"Error adding strategic plan: {e}")

    def retrieve(self, task_instruction: str, n_results: int = 1) -> list:
        """Retrieve strategic plan entries similar to the given task instruction.
        
        Args:
            task_instruction: The task instruction to find similar plans for
            n_results: Number of results to return
            
        Returns:
            List of matching strategic plan entries
        """
        try:
            embedding = self.embedding_provider.get_text_embedding(task_instruction)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["metadatas"]
            )
            return results['metadatas'][0] if results['metadatas'] else []
        except Exception as e:
            self.logger.error(f"Failed to retrieve strategic plan memory entries: {str(e)}")
            return []

class NavigateExperienceMemory(ExperienceMemory):
    """Stores and retrieves experiences related to the 'Navigate' strategy."""
    def __init__(self, client, embedding_provider):
        super().__init__(client, "navigate_experience_memory", embedding_provider)

    def add(self, navigation_goal: str, map_snapshot: Optional[Image.Image], refined_action: str, is_success: bool, reflect_reason: str = "", trajectory: str = "", current_state: str = "", memory_type: str = "experience", semantic_enhanced_image: Optional[Image.Image] = None, timestep: int = 0):
        """Add a navigation experience entry to memory.
        
        Args:
            navigation_goal: The specific navigation goal (e.g., "go to the intersection of A and B").
            map_snapshot: PIL Image of the map at the decision point. Can be None.
            refined_action: For failures, the corrected action; for successes, the original effective action.
            is_success: Boolean indicating if the outcome was successful.
            reflect_reason: Core lesson learned from the experience.
            trajectory: A JSON string representing the sequence of actions taken.
            current_state: Detailed composite string describing the current state snapshot.
            memory_type: The type of memory, e.g., 'experience' or 'reflection_note'.
            semantic_enhanced_image: PIL Image of the semantic enhanced map. Can be None.
            timestep: The timestep when this experience occurred.
        """
        try:
            embedding_text = f"Goal: {navigation_goal}. State: {current_state}"
            embedding = self.embedding_provider.get_multimodal_embedding(embedding_text, map_snapshot)
            # 🔧 修复：按照文档要求统一字段名和值格式
            metadata = {
                "navigation_goal": navigation_goal or "",
                "refined_action": refined_action or "{}",
                "is_success": "success" if is_success else "failure",  # 🔧 修复：使用 'success'/'failure' 而不是 'True'/'False'
                "reflect_reason": reflect_reason or "",
                "current_state": current_state or "",  # 🔧 修复：存储完整的 current_state 而不是简单的 scene_caption
                "map_snapshot_b64": encode_image_to_base64(map_snapshot) or "",
                "semantic_enhanced_image_b64": encode_image_to_base64(semantic_enhanced_image) or "",
                "timestep": timestep
            }
            
            # 🔧 修复：使用UUID生成唯一ID，避免重复写入
            doc_id = str(uuid.uuid4())
            
            # 修复数组歧义错误：确保embedding是列表格式
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            elif hasattr(embedding, '__array__'):
                embedding_list = np.array(embedding).tolist()
            else:
                embedding_list = embedding
                
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding_list],
                metadatas=[metadata],
                documents=[embedding_text]
            )
        except Exception as e:
            self.logger.error(f"Failed to add navigation experience: {e}")
            raise

    def retrieve(self, query_text: str, n_results: int = 1, where_filter: Optional[Dict[str, Any]] = None) -> list:
        """Retrieve navigation experiences similar to the given query text.
        
        Args:
            query_text: The text query, typically a description of the current scene.
            n_results: Number of results to return
            where_filter: Optional ChromaDB where filter
            
        Returns:
            List of matching navigation experience entries
        """
        try:
            embedding = self.embedding_provider.get_text_embedding(query_text)
            # Only include where clause if filter is not empty
            query_params = {
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["metadatas"]
            }
            if where_filter:  # Only add where clause if filter is not empty
                query_params["where"] = where_filter
                
            results = self.collection.query(**query_params)
            return results['metadatas'][0] if results['metadatas'] else []
        except Exception as e:
            self.logger.error(f"Failed to retrieve navigation experience memory entries: {str(e)}")
            return []

class SearchAndLocateMemory(ExperienceMemory):
    """Stores and retrieves experiences related to 'Search' and 'Locate' strategies."""
    def __init__(self, client, embedding_provider):
        super().__init__(client, "search_and_locate_memory", embedding_provider)

    def add(self, search_goal: str, map_snapshot: Optional[Image.Image], scene_graph_summary: str, refined_action: str, is_success: bool, reflect_reason: str = "", trajectory: str = "", current_state: str = "", memory_type: str = "experience", semantic_enhanced_image: Optional[Image.Image] = None, timestep: int = 0):
        """Add a search and locate experience entry to memory.
        
        Args:
            search_goal: The search/locate goal.
            map_snapshot: PIL Image of the map. Can be None.
            scene_graph_summary: Text summary of the scene graph.
            refined_action: For failures, the corrected action; for successes, the original effective action.
            is_success: Boolean indicating if the outcome was successful.
            reflect_reason: Core lesson learned from the experience.
            trajectory: JSON string of the action sequence.
            current_state: Detailed composite string describing the current state snapshot.
            memory_type: The type of memory, e.g., 'experience' or 'reflection_note'.
            semantic_enhanced_image: PIL Image of the semantic enhanced map. Can be None.
            timestep: The timestep when this experience occurred.
        """
        try:
            text_for_embedding = f"Goal: {search_goal}. State: {current_state}\nContext: {scene_graph_summary}"
            embedding = self.embedding_provider.get_multimodal_embedding(text_for_embedding, map_snapshot)

            # 🔧 修复：按照文档要求统一字段名和值格式
            metadata = {
                "search_goal": search_goal or "",
                "refined_action": refined_action or "{}",
                "is_success": "success" if is_success else "failure",  # 🔧 修复：使用 'success'/'failure' 而不是 'True'/'False'
                "reflect_reason": reflect_reason or "",
                "current_state": current_state or "",  # 🔧 修复：存储完整的 current_state 而不是简单的 scene_caption
                "map_snapshot_b64": encode_image_to_base64(map_snapshot) or "",
                "semantic_enhanced_image_b64": encode_image_to_base64(semantic_enhanced_image) or "",
                "timestep": timestep
            }
            
            # 🔧 修复：使用UUID生成唯一ID，避免重复写入
            doc_id = str(uuid.uuid4())

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else embedding], # 确保 embedding 是列表
                metadatas=[metadata],
                documents=[text_for_embedding]
            )
        except Exception as e:
            self.logger.error(f"Failed to add search/locate experience: {e}")
            raise

    def retrieve(self, query_text: str, n_results: int = 1, where_filter: Optional[Dict[str, Any]] = None) -> list:
        """Retrieve search and locate experiences similar to the given query text.
        
        Args:
            query_text: The text query, typically a description of the current scene.
            n_results: Number of results to return
            where_filter: Optional ChromaDB where filter
            
        Returns:
            List of matching search and locate experience entries
        """
        try:
            embedding = self.embedding_provider.get_text_embedding(query_text)
            # Only include where clause if filter is not empty
            query_params = {
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["metadatas"]
            }
            if where_filter:  # Only add where clause if filter is not empty
                query_params["where"] = where_filter
                
            results = self.collection.query(**query_params)
            return results['metadatas'][0] if results['metadatas'] else []
        except Exception as e:
            self.logger.error(f"Failed to retrieve search and locate memory entries: {str(e)}")
            return []


# ================== Main Memory System ==================

class MultiModalMemory:
    """Main memory system that integrates different memory types.
    
    Manages strategic plan memory, navigation experience memory, and search experience memory.
    """
    def __init__(self, embedding_provider: OpenAIEmbeddingProvider, db_path: Optional[str] = None):
        # 🔧 修复：使用配置文件管理数据库路径
        if db_path is None:
            from reflectionnav_agentV3.config_loader import get_config
            config = get_config()
            db_path = config.memory.experience_db_path
        
        # 🔧 新增：确保数据库路径存在
        import os
        if not os.path.exists(db_path):
            try:
                os.makedirs(db_path, exist_ok=True)
                self.logger = logging.getLogger(self.__class__.__name__)
                self.logger.info(f"Created database directory: {db_path}")
            except Exception as e:
                self.logger.error(f"Failed to create database directory {db_path}: {e}")
                # 回退到默认路径
                db_path = "./chroma"
                self.logger.warning(f"Falling back to default path: {db_path}")
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_provider = embedding_provider
        
        self.mainplan_experience_memory = MainplanExperienceMemory(self.client, self.embedding_provider)
        self.navigate_experience_memory = NavigateExperienceMemory(self.client, self.embedding_provider)
        self.search_and_locate_memory = SearchAndLocateMemory(self.client, self.embedding_provider)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 🔧 新增：显示实际使用的数据库路径
        actual_path = self.client.get_settings().persist_directory
        self.logger.info(f"MultiModalMemory initialized with {len(self.client.list_collections())} collections at {actual_path}")
        
        # 🔧 新增：验证数据库连接
        try:
            collections = self.client.list_collections()
            self.logger.info(f"Database contains {len(collections)} collections: {[col.name for col in collections]}")
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")

    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage.
        
        Returns:
            Dictionary with collection names as keys and counts as values.
        """
        try:
            stats = {}
            # 🔧 修复：使用正确的collection.count()方法
            for collection in self.client.list_collections():
                stats[collection.name] = collection.count()
            
            # 🔧 新增：添加数据库路径信息
            actual_path = self.client.get_settings().persist_directory
            stats['database_path'] = actual_path
            
            return stats
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory statistics: {str(e)}")
            return {}
