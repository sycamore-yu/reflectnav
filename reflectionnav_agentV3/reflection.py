import base64
import io
from openai import OpenAI
import os
from PIL import Image
import json
from typing import Any, Dict, List, Optional

# 简化导入，移除重复的导入
import json
import os
from openai import OpenAI
from typing import Any, Dict, List

# 🔧 新增：导入LangChain解析器和Pydantic模型
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_community.llms import OpenAI as LangChainOpenAI

# 🔧 修复：添加safe_json_dumps导入
try:
    from reflectionnav_agentV3.utils import safe_json_dumps
except ImportError:
    from utils import safe_json_dumps

try:
    from reflectionnav_agentV3.prompt_manager import prompt_manager
except ImportError:
    from prompt_manager import prompt_manager

# 🔧 新增：导入Pydantic输出模型
try:
    from reflectionnav_agentV3.llm_manager import (
        SuccessAnalysisOutput, 
        FailureAnalysisOutput, 
        MainPlanReflectionOutput
    )
except ImportError:
    from llm_manager import (
        SuccessAnalysisOutput, 
        FailureAnalysisOutput, 
        MainPlanReflectionOutput
    )

class ReflectionAgent:
    def __init__(self, vl_api_key=None, base_url=None):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided either via vl_api_key argument or DASHSCOPE_API_KEY environment variable")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_API_BASE")
        )
        
        # 🔧 新增：初始化LangChain解析器
        self._init_parsers()
    
    def _init_parsers(self):
        """初始化LangChain输出解析器"""
        try:
            # 基础解析器
            self.success_parser = PydanticOutputParser(pydantic_object=SuccessAnalysisOutput)
            self.failure_parser = PydanticOutputParser(pydantic_object=FailureAnalysisOutput)
            self.main_plan_parser = PydanticOutputParser(pydantic_object=MainPlanReflectionOutput)
            
            # 🔧 进阶：使用OutputFixingParser提供自动修复功能
            # 创建LangChain OpenAI客户端用于修复
            llm = LangChainOpenAI(
                openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
                openai_api_base=os.getenv("DASHSCOPE_API_BASE"),
                model_name="qwen-max",
                temperature=0
            )
            
            # 包装解析器，提供自动修复功能
            self.success_parser = OutputFixingParser.from_llm(
                parser=self.success_parser, 
                llm=llm
            )
            self.failure_parser = OutputFixingParser.from_llm(
                parser=self.failure_parser, 
                llm=llm
            )
            self.main_plan_parser = OutputFixingParser.from_llm(
                parser=self.main_plan_parser, 
                llm=llm
            )
            
            print("✅ LangChain解析器初始化成功")
            
        except Exception as e:
            print(f"⚠️  LangChain解析器初始化失败，回退到基础解析: {e}")
            # 回退到基础解析器
            self.success_parser = PydanticOutputParser(pydantic_object=SuccessAnalysisOutput)
            self.failure_parser = PydanticOutputParser(pydantic_object=FailureAnalysisOutput)
            self.main_plan_parser = PydanticOutputParser(pydantic_object=MainPlanReflectionOutput)

    def analyze_failure(self, trajectory_log: list, failure_reason: str, master_plan: Optional[dict] = None):
        """
        Analyzes a failed trajectory to determine the cause and suggest a correction.
        
        Args:
            trajectory_log (list): List of trajectory entries with goal, action, and scene_graph_summary
            failure_reason (str): Description of why the episode failed
            master_plan (dict): The original master plan for the mission
            
        Returns:
            dict: Analysis with 'critical_timestep', 'root_cause', 'corrected_action', and 'reflection_note'
        """
        # 🔧 修复：对trajectory_log进行净化处理，移除不可序列化对象
        def sanitize_trajectory_log(trajectory_data):
            """净化轨迹日志，移除不可序列化的对象，保留LLM分析所需的文本信息"""
            sanitized_log = []
            
            for step in trajectory_data:
                if isinstance(step, dict):
                    # 创建净化后的步骤数据，只保留文本信息
                    sanitized_step = {
                        "goal": step.get("goal", "N/A"),
                        "action": step.get("action", "N/A"),
                        "scene_graph_summary": step.get("scene_graph_summary", "N/A"),
                        "current_state": step.get("current_state", ""),
                        "strategy": step.get("strategy", "Unknown"),
                        "timestep": step.get("timestep", 0),
                        # 移除不可序列化的对象
                        # "map_snapshot": 移除 - 包含numpy数组
                    }
                    sanitized_log.append(sanitized_step)
                elif isinstance(step, str):
                    sanitized_log.append(step)
                else:
                    sanitized_log.append(str(step))
            
            return sanitized_log
        
        # 净化轨迹日志
        sanitized_trajectory = sanitize_trajectory_log(trajectory_log)
        
        # 格式化轨迹文本
        trajectory_text = ""
        for i, step in enumerate(sanitized_trajectory):
            trajectory_text += f"Step {i+1}:\n"
            if isinstance(step, dict):
                trajectory_text += f"  Goal: {step.get('goal', 'N/A')}\n"
                trajectory_text += f"  Action: {step.get('action', 'N/A')}\n"
                trajectory_text += f"  Context: {step.get('scene_graph_summary', 'N/A')}\n\n"
            elif isinstance(step, str):
                trajectory_text += f"  Data: {step}\n\n"
            else:
                trajectory_text += f"  Data: {str(step)}\n\n"
        
        # 🔧 新增：使用LangChain解析器获取格式指令
        format_instructions = self.failure_parser.get_format_instructions()
        
        # 使用安全的JSON序列化（已在文件顶部导入）
        user_prompt = prompt_manager.format_prompt(
            "failure_analysis",
            failure_reason=failure_reason,
            trajectory_log=trajectory_text,
            master_plan=safe_json_dumps(master_plan) if master_plan else "{}",
            format_instructions=format_instructions  # 🔧 新增：注入格式指令
        )
        
        # 🔧 修复：安全获取system_prompt
        system_template = prompt_manager.get_template("reflection_system")
        system_prompt = system_template.template if system_template else ""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-max",  # 使用文本模型，不需要视觉功能
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content is not None:
                # 🔧 新增：使用LangChain解析器替代json.loads
                try:
                    parsed_output = self.failure_parser.parse(content)
                    # 转换为字典格式以保持兼容性
                    return parsed_output.model_dump()
                except Exception as parse_error:
                    print(f"⚠️  解析失败，尝试手动解析: {parse_error}")
                    # 回退到手动解析
                    return json.loads(content)
            else:
                raise ValueError("API response content is empty.")
                
        except Exception as e:
            print(f"Error during reflection analysis: {e}")
            return {
                "critical_timestep": 0,
                "reflect_reason": "Analysis failed due to API error.",
                "corrected_action": {"movement": "north", "reason": "Default fallback action"},
                "reflection_note": "Could not determine a corrective action due to technical issues."
            }

    def analyze_success(self, goal_instruction: str, master_plan: dict, trajectory_log: list):
        """Analyzes a successful trajectory to extract key experiences."""
        log_str = "\n".join([f"Step {i+1}: {log}" for i, log in enumerate(trajectory_log)])
        
        # 🔧 新增：使用LangChain解析器获取格式指令
        format_instructions = self.success_parser.get_format_instructions()
        
        prompt = prompt_manager.format_prompt(
            "success_analysis",
            goal_instruction=goal_instruction,
            master_plan=safe_json_dumps(master_plan),
            trajectory_log=log_str,
            format_instructions=format_instructions  # 🔧 新增：注入格式指令
        )

        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is not None:
                # 🔧 新增：使用LangChain解析器替代json.loads
                try:
                    parsed_output = self.success_parser.parse(content)
                    return parsed_output.model_dump()
                except Exception as parse_error:
                    print(f"⚠️  解析失败，尝试手动解析: {parse_error}")
                    return json.loads(content)
            else:
                raise ValueError("API response content is empty.")
        except Exception as e:
            print(f"Error during success analysis: {e}")
            return {
                "critical_timestep": 0,
                "reflect_reason": "Analysis failed due to API error.",
                "successful_action": {"movement": "north", "reason": "Default fallback action"},
                "reflection_note": "Could not determine success factors due to technical issues."
            }

    def analyze_main_plan(self, task_instruction: str, master_plan: dict, trajectory_history: list, final_reason: str):
        """Analyzes the main plan effectiveness and provides optimization insights."""
        # 🔧 修复：对trajectory_history进行净化处理，移除numpy数组等不可序列化对象
        def sanitize_trajectory_for_llm(trajectory_data):
            """净化轨迹数据，移除不可序列化的对象，保留LLM分析所需的文本信息"""
            sanitized_trajectory = []
            
            for frame in trajectory_data:
                if isinstance(frame, dict):
                    # 创建净化后的帧数据，只保留文本信息
                    sanitized_frame = {
                        "timestep": frame.get("timestep", 0),
                        "goal": frame.get("goal", "Unknown"),
                        "strategy": frame.get("strategy", "Unknown"),
                        "current_state": frame.get("current_state", ""),
                        "action": frame.get("action", ""),
                        "scene_caption": frame.get("scene_caption", ""),
                        "scene_graph_summary": frame.get("scene_graph_summary", ""),
                        "distance_to_target": frame.get("distance_to_target", 0.0),
                        "pose": frame.get("pose", (0, 0, 0, 0)),
                        # 移除不可序列化的对象
                        # "map_snapshot": 移除 - 包含numpy数组
                        # "llm_system_prompt": 移除 - 可能包含敏感信息
                        # "llm_prompt": 移除 - 可能包含敏感信息
                    }
                    sanitized_trajectory.append(sanitized_frame)
                else:
                    # 如果不是字典，转换为字符串
                    sanitized_trajectory.append(str(frame))
            
            return sanitized_trajectory
        
        # 净化轨迹数据
        sanitized_trajectory = sanitize_trajectory_for_llm(trajectory_history)
        
        # 使用安全的JSON序列化（已在文件顶部导入）
        trajectory_str = safe_json_dumps(sanitized_trajectory)
        
        # 🔧 新增：使用LangChain解析器获取格式指令
        format_instructions = self.main_plan_parser.get_format_instructions()
        
        prompt = prompt_manager.format_prompt(
            "main_plan_reflection_prompt",
            task_instruction=task_instruction,
            master_plan=safe_json_dumps(master_plan),
            trajectory_history=trajectory_str,
            final_reason=final_reason,
            format_instructions=format_instructions  # 🔧 新增：注入格式指令
        )

        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is not None:
                # 🔧 新增：使用LangChain解析器替代json.loads
                try:
                    parsed_output = self.main_plan_parser.parse(content)
                    return parsed_output.model_dump()
                except Exception as parse_error:
                    print(f"⚠️  解析失败，尝试手动解析: {parse_error}")
                    return json.loads(content)
            else:
                raise ValueError("API response content is empty.")
        except Exception as e:
            print(f"Error during main plan analysis: {e}")
            return {
                "plan_refined": json.dumps(master_plan),
                "reflection_note": "Analysis failed due to technical issues."
            }
