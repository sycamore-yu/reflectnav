"""
LLM Manager for ReflectionNav_AgentV3 using LangChain.
Manages prompt templates, model calls, and response parsing.
"""

import json
import logging
import re
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from PIL import Image
from reflectionnav_agentV3.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)

# ===============================================
# Pydantic Models for Structured Output
# ===============================================

class NavigationOutput(BaseModel):
    """Navigation strategy output structure."""
    movement: str = Field(description="Direction to move: northwest|northeast|southwest|southeast|north|south|east|west")
    reason: str = Field(description="Explanation of why this direction was chosen")

class NavigationDecisionOutput(BaseModel):
    """Navigation strategy decision output structure."""
    movement: str = Field(description="Direction to move: northwest|northeast|southwest|southeast|north|south|east|west")
    reason: str = Field(description="Explanation of navigation decision")
    next_strategy: str = Field(description="Next strategy decision: 'continue_navigate' or 'next_subtask'")
    confidence: float = Field(description="Confidence in strategy decision 0-1", ge=0, le=1)

class SearchOutput(BaseModel):
    """Search strategy output structure with integrated decision making."""
    movement: str = Field(description="Direction to search: northwest|northeast|southwest|southeast|north|south|east|west")
    reason: str = Field(description="Explanation of search strategy")
    decision: str = Field(description="Next action decision: 'continue_search' or 'next_subtask'")
    confidence: float = Field(description="Confidence in decision 0-1", ge=0, le=1)

class LocateOutput(BaseModel):
    """Locate strategy output structure."""
    status: str = Field(description="Status: TARGET_LOCKED or SEARCHING_VICINITY")
    selected_pos: Optional[List[float]] = Field(description="Target coordinates [x, y] if found")
    reason: str = Field(description="Explanation of locate decision")
    confidence: float = Field(description="Confidence level 0-1", ge=0, le=1)

class PlanOutput(BaseModel):
    """Planning output structure."""
    sub_goals: List[Dict[str, Any]] = Field(description="List of sub-goals with strategy and desired state")
    reason: str = Field(description="Explanation of the plan")

# 🔧 新增：反思分析相关的Pydantic输出模型
class ActionDetail(BaseModel):
    """Represents the detailed action taken by the agent."""
    movement: str = Field(description="The direction of movement: northwest|northeast|southwest|southeast|north|south|east|west")
    reason: str = Field(description="The reason for choosing this movement")

class ReflectionAnalysisOutput(BaseModel):
    """The structured output for reflection analysis."""
    critical_timestep: int = Field(description="The index of the critical timestep that was a turning point", ge=0)
    reflect_reason: str = Field(description="The core lesson learned from this experience")
    reflection_note: str = Field(description="A concise, memorable note for future situations")

class SuccessAnalysisOutput(ReflectionAnalysisOutput):
    """Output schema for a successful mission analysis."""
    successful_action: ActionDetail = Field(description="The effective action taken at the critical timestep")

class FailureAnalysisOutput(ReflectionAnalysisOutput):
    """Output schema for a failed mission analysis."""
    corrected_action: ActionDetail = Field(description="The corrected action that should have been taken")

class MainPlanReflectionOutput(BaseModel):
    """Output schema for main plan reflection analysis."""
    plan_refined: str = Field(description="The refined or corrected plan in JSON format")
    reflection_note: str = Field(description="Detailed analysis and reflection notes about the original plan")

# ===============================================
# LLM Manager Class
# ===============================================

class LLMManager:
    """Manages all LLM interactions with structured output parsing."""

    def __init__(self, openai_client, prompts: Dict[str, str] = None):
        self.client = openai_client
        # Use new PromptManager if no prompts provided (backward compatibility)
        if prompts is None:
            self.prompts = prompt_manager.get_prompt_templates()
            self.use_prompt_manager = True
        else:
            self.prompts = prompts
            self.use_prompt_manager = False
        
        # Store reference to prompt manager
        self.prompt_manager = prompt_manager
        
        # Initialize parsing objects
        self.nav_parser = PydanticOutputParser(pydantic_object=NavigationOutput)
        self.nav_decision_parser = PydanticOutputParser(pydantic_object=NavigationDecisionOutput)
        self.search_parser = PydanticOutputParser(pydantic_object=SearchOutput)
        self.locate_parser = PydanticOutputParser(pydantic_object=LocateOutput)
        self.plan_parser = PydanticOutputParser(pydantic_object=PlanOutput)
        
        # Initialize fixing parsers for error correction
        self.nav_fixing_parser = self._get_fixing_parser(self.nav_parser)
        self.nav_decision_fixing_parser = self._get_fixing_parser(self.nav_decision_parser)
        self.search_fixing_parser = self._get_fixing_parser(self.search_parser)
        self.locate_fixing_parser = self._get_fixing_parser(self.locate_parser)
        self.plan_fixing_parser = self._get_fixing_parser(self.plan_parser)
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt using either PromptManager or legacy prompts."""
        if self.use_prompt_manager:
            return self.prompt_manager.format_prompt(template_name, **kwargs)
        else:
            # Legacy method - use .format() on prompt strings
            template = self.prompts.get(template_name, "")
            try:
                return template.format(**kwargs)
            except KeyError as e:
                logger.error(f"Missing variable {e} for template '{template_name}'")
                return ""
            except Exception as e:
                logger.error(f"Error formatting template '{template_name}': {e}")
                return ""

    def _get_fixing_parser(self, base_parser):
        """Create a fixing parser for error correction."""
        try:
            return OutputFixingParser.from_llm(parser=base_parser, llm=ChatOpenAI(temperature=0))
        except Exception:
            return base_parser

    def call_llm_with_parsing(self,
                            system_prompt: Optional[str],
                            user_prompt: str,
                             images: Optional[List[str]] = None,
                             parser_type: str = "json",
                             max_retries: int = 3) -> Union[Dict, BaseModel, str]:
        """
        Enhanced LLM call with robust parsing support.

        Args:
            system_prompt: System message
            user_prompt: User message
            images: List of base64 encoded images
            parser_type: Type of parser - "json", "pydantic", or specific model names
            max_retries: Maximum retry attempts
        """
        # Map parser types to actual parsers
        parser_map = {
            "NavigationOutput": PydanticOutputParser(pydantic_object=NavigationOutput),
            "SearchOutput": PydanticOutputParser(pydantic_object=SearchOutput),
            "LocateOutput": PydanticOutputParser(pydantic_object=LocateOutput),
            "PlanOutput": PydanticOutputParser(pydantic_object=PlanOutput),
            "json": SimpleJsonOutputParser(),
            "pydantic": SimpleJsonOutputParser()  # Will be determined by context
        }

        # Build content for user message
        content = [{"type": "text", "text": user_prompt}]
        # Add images to content if provided
        if images:
            for img_b64 in images:
                if img_b64:
                    # Insert image content with proper typing
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                    content.insert(0, image_content)

        # Build messages
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            messages = [{"role": "user", "content": content}]

        last_error = None
        for attempt in range(max_retries):
            try:
                # Call LLM
                response = self.client.chat.completions.create(
                    model="qwen-vl-max",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1
                )
                raw_text = response.choices[0].message.content.strip()

                # Parse based on type
                if parser_type == "json" or parser_type == "pydantic":
                    # Try JSON parsing first
                    try:
                        json_result = self._extract_json_fallback(raw_text)
                        return self._normalize_output_fields(json_result)
                    except Exception:
                        return raw_text

                elif parser_type in parser_map:
                    # Use specific pydantic parser
                    parser = parser_map[parser_type]
                    if isinstance(parser, PydanticOutputParser):
                        try:
                            return parser.parse(raw_text)
                        except Exception:
                            # Try with JSON extraction first
                            json_result = self._extract_json_fallback(raw_text)
                            normalized = self._normalize_output_fields(json_result)

                            # Create appropriate model instance
                            if parser_type == "NavigationOutput":
                                return NavigationOutput(**normalized)
                            elif parser_type == "SearchOutput":
                                return SearchOutput(**normalized)
                            elif parser_type == "LocateOutput":
                                return LocateOutput(**normalized)
                            elif parser_type == "PlanOutput":
                                return PlanOutput(**normalized)
                    else:
                        return parser.parse(raw_text)
                else:
                    return raw_text

            except Exception as e:
                last_error = e
                logger.warning(f"Parsing attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue

        # Final fallback
        logger.error(f"All parsing attempts failed. Last error: {last_error}")
        if parser_type in ["NavigationOutput", "SearchOutput", "LocateOutput", "PlanOutput"]:
            return self._create_fallback_response(parser_type)
        return {"error": "parsing_failed"}

    def _extract_json_fallback(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text with multiple fallback strategies."""
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find any JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Manual extraction for common patterns
        try:
            # Extract key-value pairs manually
            result = {}

            # Common patterns
            patterns = [
                (r'"reasoning":\s*"([^"]*)"', 'reason'),
                (r'"reason":\s*"([^"]*)"', 'reason'),
                (r'"movement":\s*"([^"]*)"', 'movement'),
                (r'"decision":\s*"([^"]*)"', 'decision'),
                (r'"status":\s*"([^"]*)"', 'status'),
                (r'"confidence":\s*([0-9.]+)', 'confidence'),
                (r'"information_gain":\s*([0-9.]+)', 'information_gain')
            ]

            for pattern, key in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if key in ['confidence', 'information_gain']:
                        result[key] = float(match.group(1))
                    else:
                        result[key] = match.group(1)

            if result:
                # Apply field normalization
                return self._normalize_output_fields(result)

        except Exception as e:
            logger.error(f"Manual extraction failed: {e}")

        # Final fallback
        logger.warning(f"Could not extract valid JSON from: {text[:100]}...")
        return {}

    def _normalize_output_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names to match expected model fields."""
        field_mapping = {
            # Common field mappings - mapping TO the pydantic model field names
            "reasoning": "reason",  # LLM outputs "reasoning", model expects "reason"
            "explanation": "reason",
            "direction": "movement",
            "move": "movement",
            "action": "movement",
            "selected_position": "selected_pos",
            "position": "selected_pos",
            "target_position": "selected_pos"
        }

        normalized = {}
        for key, value in data.items():
            # Use mapping if exists, otherwise keep original key
            new_key = field_mapping.get(key, key)
            normalized[new_key] = value

        # Add default values for missing required fields
        if "information_gain" not in normalized:
            normalized["information_gain"] = 0.5
        if "confidence" not in normalized:
            normalized["confidence"] = 0.7
        if "reason" not in normalized:  # Use "reason" not "reasoning"
            normalized["reason"] = "Default reasoning"

        # Ensure numeric fields are properly typed
        for field in ["confidence", "information_gain"]:
            if field in normalized and isinstance(normalized[field], str):
                try:
                    normalized[field] = float(normalized[field])
                except ValueError:
                    if field == "confidence":
                        normalized[field] = 0.7
                    else:
                        normalized[field] = 0.5

        return normalized

    def navigate_decision(self, system_prompt: str, user_prompt: str, images: Optional[List[str]] = None) -> NavigationOutput:
        """Generate navigation decision."""
        try:
            result = self.call_llm_with_parsing(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                parser_type="json",
                max_retries=3
            )
            
            if isinstance(result, dict):
                return NavigationOutput(**result)
            else:
                return result
        except Exception as e:
            logger.error(f"Error in navigation decision: {e}")
            return NavigationOutput(movement="north", reason="Default navigation")

    def navigation_strategy_decision(self, system_prompt: str, user_prompt: str, images: Optional[List[str]] = None) -> NavigationDecisionOutput:
        """Generate navigation strategy decision with next strategy choice."""
        try:
            result = self.call_llm_with_parsing(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                parser_type="json",
                max_retries=3
            )
            
            # Handle dict return from call_llm_with_parsing
            if isinstance(result, dict):
                return NavigationDecisionOutput(
                    movement=result.get('movement', 'north'),
                    reason=result.get('reason', result.get('reasoning', 'Default navigation')),
                    next_strategy=result.get('next_strategy', 'continue_navigate'),
                    confidence=result.get('confidence', 0.5)
                )
            else:
                return result
        except Exception as e:
            logger.error(f"Error in navigation strategy decision: {e}")
            return NavigationDecisionOutput(
                movement="north", 
                reason="Default navigation", 
                next_strategy="continue_navigate",
                confidence=0.5
            )

    def search_decision(self, system_prompt: str, user_prompt: str, images: Optional[List[str]] = None) -> SearchOutput:
        """Generate search decision with integrated continue/locate logic."""
        try:
            result = self.call_llm_with_parsing(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                parser_type="json",
                max_retries=3
            )

            # Handle dict return from call_llm_with_parsing
            if isinstance(result, dict):
                return SearchOutput(
                    movement=result.get('movement', 'north'),
                    reason=result.get('reason', result.get('reasoning', 'Default search')),
                    decision=result.get('decision', 'continue_search'),
                    confidence=result.get('confidence', 0.5)
                )
            elif isinstance(result, SearchOutput):
                return result
            else:
                logger.warning(f"Unexpected result type for search decision: {type(result)}")
                return SearchOutput(movement="north", reason="Default search", decision="continue_search", confidence=0.5)

        except Exception as e:
            logger.error(f"Error in search decision: {e}")
            return SearchOutput(movement="north", reason="Error fallback", decision="continue_search", confidence=0.1)

    def locate_decision(self, system_prompt: str, user_prompt: str, images: Optional[List[str]] = None) -> LocateOutput:
        """Get locate decision with structured output."""
        result = self.call_llm_with_parsing(system_prompt, user_prompt, images, "json")

        if isinstance(result, dict):
            return LocateOutput(
                status=result.get('status', 'SEARCHING_VICINITY'),
                selected_pos=result.get('selected_pos'),
                reason=result.get('reason', result.get('reasoning', 'Default reasoning')),
                confidence=result.get('confidence', 0.5)
            )
        return LocateOutput(status='SEARCHING_VICINITY', selected_pos=None, reason='Default reasoning', confidence=0.5)

    def generate_plan(self, system_prompt: str, user_prompt: str, images: Optional[List[str]] = None) -> PlanOutput:
        """Generate plan with structured output."""
        result = self.call_llm_with_parsing(system_prompt, user_prompt, images, "json")

        if isinstance(result, dict):
            return PlanOutput(
                sub_goals=result.get('sub_goals', []),
                reason=result.get('reason', result.get('reasoning', 'Default plan reasoning'))
            )
        return PlanOutput(sub_goals=[], reason='Default plan reasoning')

    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate scene caption from PIL Image using PromptManager."""
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # 🔧 Fixed: scene_caption prompt doesn't need task_instruction parameter
            prompt_text = self.format_prompt("scene_caption")

            # Generate caption using the OpenAI client
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }]
            )
            return response.choices[0].message.content or "Unable to generate scene description"
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return "Unable to generate scene description"

    def _create_fallback_response(self, parser_type: str):
        """Create fallback responses for different parser types."""
        if parser_type == "NavigationOutput":
            return NavigationOutput(movement="north", reason="Fallback response")
        elif parser_type == "SearchOutput":
            return SearchOutput(movement="north", reason="Fallback response", decision="continue_search", confidence=0.5)
        elif parser_type == "LocateOutput":
            return LocateOutput(status="SEARCHING_VICINITY", selected_pos=None, reason="Fallback response", confidence=0.5)
        elif parser_type == "PlanOutput":
            return PlanOutput(sub_goals=[], reason="Fallback response")
        return {"error": "unknown_parser_type"}