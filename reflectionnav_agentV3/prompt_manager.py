"""
Unified Prompt Management System for ReflectionNav Agent V3
Uses LangChain PromptTemplate for better prompt management and consistency.
"""

from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Unified prompt management system using LangChain PromptTemplate.
    Combines and organizes all prompts from prompt_cot.py and prompts.py.
    """
    
    def __init__(self):
        """Initialize all prompt templates."""
        self.templates = self._init_prompt_templates()
        
    def _init_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates using LangChain PromptTemplate."""
        
        templates = {}
        
        # ===== Goal Description Prompts =====
        
        templates["goal_description_nav"] = PromptTemplate(
            input_variables=[],
            template="""You are controlling a UAV to navigate in a city environment. Your task is to navigate to a specific landmark using a top-down sketch.

**Map Symbol Interpretation Guide:**

- **Orientation**: The map is always oriented with North at the top.
    - The top of the image is **North**.
    - The bottom of the image is **South**.
    - The right side of the image is **East**.
    - The left side of the image is **West**.
    - Therefore, the **top-right corner represents Northeast**, and the **bottom-right corner represents Southeast**, and so on.
- **You (The UAV)**: You are represented by the **green arrow (->)**. The arrow's direction shows your UAV's current heading.
- **Your Trajectory**: The **black line** is a trail showing your recent flight path. It shows where you have been.
- **Your Goal**: The target is a semantic landmark, clearly labeled with its name 

**Your Core Logic:**
Your fundamental task is to continuously compare the landmark's position to your own (the green arrow). You must always choose a direction of movement that closes the distance to the landmark. Use your trajectory (the black line) as a tool to verify your progress: if the black line points away from the landmark, you are heading in the wrong direction and must correct your course."""
        )
        
        templates["goal_description_search"] = PromptTemplate(
            input_variables=[],
            template="""You are controlling a UAV to navigate in a city environment. Your task is to search a specific object nearby the landmark using a top-down map. 
The map provides a simplified representation of the urban district, including:
1. Objects: Represented by colored polygons, such as a red square for a car, Brown for a building, or a green circle for a tree and so on.
2. Explored Area: The lavender shading represents the explored area, and the white area is unexplored.
3. The path is represented by a line with the arrow.
Important: Use your prior knowledge and the map to plan your steps systematically."""
        )
        
        templates["goal_description_locate"] = PromptTemplate(
            input_variables=[],
            template="""Your task is to locate the described target based on top-down RGB images. You will be provided two types of information:
1. Target Instruction: A description of the target object that you need to locate.
2. Observation: An aerial view image that may contain the target and nearby surrounding objects.
Use your reasoning and observation skills to identify the target systematically."""
        )
        
        # ===== Navigation Prompts =====
        
        templates["landmark_navigation"] = PromptTemplate(
            input_variables=["geoinstruct", "goal", "state", "history_actions", "experience"],
            template="""Answer the following question based on the provided map.

<Map Legend>
- Your current position: **Green Arrow (->)**
- Your recent path is the: **Black Line**
- Your target is the: **Labeled Landmark** 
</Map Legend>

<Task>
You are currently {geoinstruct}. Your assigned goal is: {goal}.

**First, in the `thought` block, you MUST follow these 3 reasoning steps:**
1.  **Analyze Position**: Describe where the target landmark is relative to your position (the Green Arrow). Use cardinal directions (e.g., "Leslie Road is to my southeast").
2.  **Verify Trajectory**: Look at your recent trajectory (the Black Line). Does its direction point towards the target landmark? State "Yes" or "No".
3.  **Propose Action**: Based on your analysis, decide the correct direction to move. If you answered "No" in step 2, explicitly state that you are correcting a previous error.

**Then, provide your final decision in the `answer` block using the required JSON format.**
</Task>

<Contextual Clues>
- **Previous Actions**: {history_actions}
- **Important**: The 'Previous Actions' list shows your recent movements and reasoning. Analyze this history to avoid oscillating actions (e.g., moving north and then immediately moving south). Strive for consistent progress towards your goal.
- **Reflection Note**: {experience}
</Contextual Clues>

Your reply includes two components: **thought** and **answer**.

<answer>
Do not put thought here. Your answer only includes the JSON object.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your final decision concisely. This should be a summary of your thought process.",
        "movement": "Move [northwest|northeast|southwest|southeast|north|south|east|west]"
    }}```
</answer>
**Note**: The "reason" field should be a concise summary of your thought process (CoT). JSON does not support annotations! Now, output:"""
        )
        
        templates["navigation_strategy_decision"] = PromptTemplate(
            input_variables=["task_instruction", "navigation_goal", "current_position", "target_position", 
                           "distance", "scene_caption", "geoinstruct", "history_actions", "experience"],
            template="""You are a tactical navigation strategist for a UAV. Your response MUST be a single, valid JSON object and nothing else.

<Task>
Main Mission: {task_instruction}
Current Goal: {navigation_goal}
Current position: {current_position}
Target position: {target_position}
Distance to target: {distance:.1f} meters

Current scene observation: {scene_caption}
Geographical context: {geoinstruct}

Recent Action History: {history_actions}
Relevant Experience: {experience}

Analyze if you should continue with the current navigation subtask or move to the next subtask in your plan.
</Task>

<Visual Input Instructions>
You will be provided with TWO images:
1. **Real-time Map**: The current top-down view showing your position (green arrow), trajectory (black line), and target landmark
2. **Historical Reference Image**: A similar scene from past successful navigation experience for comparison

Compare these images to make an informed decision. The real-time map shows your current situation, while the reference image demonstrates what a successful navigation state looks like in similar circumstances.

<Decision Criteria>
- CONTINUE_NAVIGATE: If you need to continue navigating towards landmarks to complete current subtask
- NEXT_SUBTASK: If current navigation subtask is complete and ready to move to next phase

<thought>
Provide your reasoning and step-by-step thought process here. Consider the visual comparison between current and historical images. Do not put into json-format answer.
</thought>

<OutputFormat>
{{
    "reason": "A concise explanation for your strategy decision, under 30 words.",
    "movement": "Direction to move: 'north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'",
    "next_strategy": "Either 'continue_navigate' or 'next_subtask'",
    "confidence": 0.8
}}
</OutputFormat>"""
        )
        
        # ===== Search Prompts =====
        
        templates["object_search"] = PromptTemplate(
            input_variables=["geoinstruct", "goal", "state", "experience"],
            template="""Answer the following question:
<question>
You are currently at/near {geoinstruct}. Your assigned goal is: {goal}. Your desired state is: {state}.

Based on the overall information and top-down semantic map, determine the most logical direction to search. You should prioritize exploring nearby, unvisited areas (white regions) that are contextually appropriate for finding the target.
you better reduce the distance from target

<Contextual Clues>
- **Reflection Note**: {experience}
- **Note**: You may be provided with a second image below the main one. This second image is a visual snapshot from the successful past experience mentioned above. Use it as a reference for what a successful search state looks like.
</Contextual Clues>
</question>

Your reply includes two components: **thought** and **answer**. Your answer should follow the json format and only includes two components: **reason** and **movement**.
<thought>
Provide your reasoning and step-by-step thought process here. Do not put into json-format answer.
</thought>

<answer>
Do not put thought here, your answer only include two components: **reason** and **movement**.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here, no longer than 30 words. This should be a summary of your thought process.",
        "movement": "Move [northwest|northeast|southwest|southeast|north|south|east|west]"
    }}```
</answer>
**Note**: The "reason" field should be a concise summary of your thought process (CoT). JSON does not support annotations! Now, output:"""
        )
        
        templates["search_decision"] = PromptTemplate(
            input_variables=["task_instruction", "current_goal", "desired_state", "current_position", "view_area", 
                           "scene_caption", "geoinstruct", "history_actions", "entropy_info", "experience"],
            template="""You are a tactical search aerial agent. Your response MUST be a single, valid JSON object and nothing else.

<Task>
Task instruction: {task_instruction}
Current Goal: {current_goal}
Desired State to Achieve: {desired_state}
Current Position: {current_position}
View Area: {view_area}
Current Observation: {scene_caption}
You are currently at/near {geoinstruct}
Recent action history: {history_actions}

{entropy_info}

Experience Context: {experience}
---
**<Guiding Principles - You MUST Follow These Rules>**

1.  **Map-First Directive**: Your decision **MUST** be primarily based on the visual information from the top-down semantic map and the RGB image. The map is your ground truth.
2.  **Explore & Approach**: Your chosen direction should aim to achieve one of two objectives:
    - **Explore**: Move towards unexplored areas (white regions on the map) that are most likely to contain the target, based on your `Current Goal`.
    - **Approach**: If you see a potential target on the map that matches the goal description, move directly towards it.
3.  **Avoid Oscillation**: Use your `{history_actions}` as a reference. Do not immediately reverse your previous action (e.g., moving North right after moving South) unless the map clearly shows you have overshot the target. Your goal is consistent progress.

---
Based on the overall information and top-down semantic map, determine the most logical direction to search.
Provide:
1. Movement direction for continued search
2. Reasoning for the search strategy
3. Decision on next action (continue_search or move to next subtask)
4. Confidence in your decision

Return in JSON format with fields: movement, reason, decision, confidence
</Task>

<Visual Input Instructions>
You will be provided with TWO images:
1. **Real-time Map**: The current top-down semantic map showing explored areas (lavender), unexplored areas (white), and your current position
2. **Historical Reference Image**: A similar scene from past successful search experience for comparison

Compare these images to make an informed decision. The real-time map shows your current search situation, while the reference image demonstrates what a successful search pattern looks like in similar circumstances.

<thought>
Provide your reasoning and step-by-step thought process here. Consider the visual comparison between current and historical images. If you have no idea where to search, you better reduce the distance from target. Do not put into json-format answer.
</thought>

<OutputFormat>
{{
    "movement": "Direction to search: 'north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'",
    "reason": "A concise explanation for your chosen search direction, under 30 words.",
    "decision": "Either 'continue_search' or 'next_subtask'",
    "confidence": 0.8
}}
</OutputFormat>"""
        )
        
        # ===== Locate Prompts =====
        
        templates["target_locate"] = PromptTemplate(
            input_variables=["pos", "area", "goal", "clue", "history"],
            template="""Your current position is {pos}, which is at the center of your view. The area of your view are {area}.
<question>
Your assigned goal is: {goal}. 
You have failed to find the target using the scene graph and are now relying on direct visual analysis.
Use all available information to make the best decision.

<Contextual_Clues>
- **Past Experience**: {clue}
- **Previous Actions**: {history}
</Contextual_Clues>

Analyze the image and determine the next best coordinate to move to.
- If you can positively identify the target, provide its precise coordinates.
- If not, suggest a promising nearby coordinate for a better view (e.g., closer to a likely hiding spot).
</question>

Your reply includes two components: **thought** and **answer**. Your answer should follow the json format.
<thought>
Provide your reasoning and step-by-step thought process here. Do not put into json-format answer.
<thought>

<answer>
Your answer should follow the json format and include "status" and "selected_pos". [x, y] Coordinates must be within {area}.
**Required Output Format**:
    ```json
    {{
        "reason": "A concise explanation for your decision. If searching, explain why the new viewpoint is promising. If locked, explain why you are certain. This should be a summary of your thought process.",
        "status": "Either 'TARGET_LOCKED' or 'SEARCHING_VICINITY'.",
        "selected_pos": "[x, y]"
    }}```
</answer>
**Note**: The "reason" field should be a concise summary of your thought process (CoT). JSON does not support annotations! Now, output:"""
        )
        
        # ===== Planning Prompts =====
        
        templates["planner_prompt"] = PromptTemplate(
            input_variables=["instruction", "geoinstruct", "landmarks", "example", "reflection_note"],
            template="""You are a aerial navigation planning expert. Given the following mission and context, create a detailed step-by-step plan, make sure the plan is feasible and logical, and ensure the continuity of the plan.

Mission: {instruction}

Geographic Context:
{geoinstruct}

Available Landmarks:
{landmarks}

Example Successful Plan (for reference):
{example}

Reflection Note from Past Experience:
{reflection_note}

Your task is to create a plan to accomplish the mission. You
1.  **First sub-goal MUST use the "Navigate" strategy.** The `goal` for this step should be to move towards one or two key *landmarks* or the general area of the final target described in the mission.
2.  **"Search" strategy.** After arriving, the `goal` is to explore the immediate area to find visual confirmation of the final target.
3.  "Locate" strategy.** Once the target is visually identified, the `goal` is to move to its exact position for final confirmation.

For each sub-goal, you must define:
- goal: A clear, specific objective based on the rules above.
- strategy: The required strategy for that step ("Navigate", "Search", or "Locate").
- desired_state: A clear condition for completing the sub-goal (e.g., "Within 30 meters of Landmark X", "Visual confirmation of the target").

- **DO NOT** use cardinal directions (e.g., north, south, east, west).
- **DO NOT** use egocentric directions (e.g., left, right, forward, backward).

Output format:
```json
{{
  "sub_goals": [
    {{
            "goal": "Move to landmark X",
            "strategy": "Navigate", 
            "desired_state": "Within 30 meters of landmark X"
    }},
    {{
            "goal": "Search for target Y near landmark X",
            "strategy": "Search",
            "desired_state": "Have visual confirmation or strong clues about target Y location"
    }},
    {{
            "goal": "Navigate to precise location of target Y",
            "strategy": "Locate",
            "desired_state": "Reached target Y successfully"
    }}
  ],
  "reason": "Brief explanation of the overall strategy"
}}
```"""
        )
        
        # ===== Utility Prompts =====
        
        # 🔧 Fixed: Remove task_instruction from scene_caption prompt
        templates["scene_caption"] = PromptTemplate(
            input_variables=[],
            template="""Describe the scene in this aerial/top-down image in 1-2 sentences. Focus on:
1. The main objects or structures visible (buildings, vehicles, roads, etc.)
2. The general layout and spatial relationships
3. Any notable features or landmarks

Keep the description concise and factual."""
        )
        
        templates["landmark_description"] = PromptTemplate(
            input_variables=["landmark", "recent_objects", "surroundings"],
            template="""Geographic Context:
{landmark}

Recent Objects:
{recent_objects}

Surroundings:
{surroundings}

"""
        )
        
        # ===== Scene Graph Prompts =====
        
        templates["local_graph"] = PromptTemplate(
            input_variables=["objects"],
            template="""You are a geospatial scene graph extractor analyzing a north-aligned satellite image. 
Your task is to recognize {objects} and their relationships into a structured JSON graph following these strict rules:

**Node Requirements**
1. Each node must have a unique `id`.
2. Mandatory attributes for every node:
- `object_type`: one of ["vehicle", "road", "building", "parking_lot", "green_space", etc]
- `bbox`: bounding box coordinates [xmin, ymin, xmax, ymax]
3. Optional attribute (only if clearly observable):
- `color`: one of ["white", "black", "red", "gray", "blue", "green", "brown", "silver"]

**Edge Requirements**
1. Only use the following relationship labels, with these meanings:
- **Topological**: 
    - "contains" (one object is completely within another)
    - "adjacent_to" (objects are immediately beside one another)
    - "near_corner" (object is close to a corner of another object)
- **Directional** (absolute, from aerial perspective):
    - Primary: "north_of", "south_of", "east_of", "west_of"
    - Diagonal: "northeast_of", "northwest_of", "southeast_of", "southwest_of"

2. Absolute directional relationships take priority over relative terms:
    - "behind" → convert to "north_of"
    - "next to" → convert to "adjacent_to"
    - "in" → convert to "contains"

**Special Cases & Handling of Ambiguities**
1. **Building and Other Object Relationships**:
- Map ambiguous relative terms:
    - "behind" → convert to "north_of"
    - "next to" → convert to "adjacent_to"
    - "across from" remains as "across_from" (typically with a connecting road node)
2. **Preset Landmarks**:
- Names like "Leslie Road", "Bridgelands Way", "Livingstone Road", etc., are considered preset. Do not extract these from the image; focus solely on dynamic objects and visible spatial relations.
3. **Ambiguity Reduction**:
- Limit your relationship predicate set to the ones provided. This finite vocabulary helps eliminate ambiguity and ensures consistent mapping from natural language descriptions to spatial relationships.
4. **Hierarchical and Iterative Extraction**:
- First, build an initial graph based on absolute spatial cues (from the north-aligned image).
- Then, refine relationships using explicit ordering and structural cues from the description.

<Example Output>
For "white car parked 1st from bottom in right column":
{{
"nodes": [
    {{
    "id": "White01", 
    "object_type": "vehicle",
    "color": "white",
    "bbox": [320, 580, 360, 620]
    }},
    {{
    "id": "ParkingLot07",
    "object_type": "parking_lot",
    "bbox": [300, 500, 700, 800]
    }}
],
"edges": [
    {{
    "source": "White01",
    "target": "ParkingLot07",
    "relationship": "contains"
    }}
]
}}
</Example>
JSON does not support annotations! Please output the json in legal format. Now analyze: {objects}"""
        )
        
        templates["query_operation_chain"] = PromptTemplate(
            input_variables=["instruction"],
            template="""Converts navigation commands into a chain of query operations. Available operations:
- get_geonode_by_name(name_pattern): finds geonodes based on a name pattern. If no name is provided, all geonodes are returned.
- get_child_nodes(parent, relation_type): gets child nodes with the specified relation to the parent.
  Available relation types are: "contains", "adjacent_to", "near_corner", "north_of", "south_of", "east_of", "west_of", "northeast_of", "northwest_of", "southeast_of", "southwest_of"

Transform the instruction: {instruction}

Output as a JSON list of operations:
```json
[
    {{"operation": "get_geonode_by_name", "name_pattern": "parking"}},
    {{"operation": "get_child_nodes", "parent": "parking_lot", "relation_type": "contains"}}
]
```"""
        )
        
        # ===== Reflection Prompts =====
        
        templates["reflection_system"] = PromptTemplate(
            input_variables=[],
            template="""You are an expert in autonomous navigation and a learning coach for a UAV agent. You will be given a detailed description of the failed mission trajectory along with the failure reason.

Your response should use the following format:
<reasoning>
First, analyze the trajectory to understand what happened step by step.
Consider the agent's decisions, the context of each action, and how they led to failure.
Look for patterns like oscillation, wrong directions, or missing information.
</reasoning>

<analysis>
Identify the root cause of the failure - was it a perception error, planning mistake, or execution problem?
</analysis>

<solution>
Determine what the correct action should have been at the critical failure point.
</solution>

<lesson>
Formulate a clear, actionable lesson that the agent can apply in similar future situations.
</lesson>

Your final response must be a valid JSON object with the specified keys."""
        )

        templates["failure_analysis"] = PromptTemplate(
            input_variables=["failure_reason", "trajectory_log", "master_plan", "format_instructions"],
            template="""The agent failed a navigation mission. Analyze the trajectory and identify the critical decision turning point.

**Failure Reason:** {failure_reason}

**Master Plan:** {master_plan}

**Trajectory Log:**
{trajectory_log}

**Critical Task:** Analyze each step's decision (movement reason) and its subsequent results to locate the most critical navigation or search decision frame.

**Required Analysis:**
1. **Identify Critical Turning Point**: Review the trajectory and identify the exact timestep index where the agent made a critical error that led to failure. Return this as an integer.
2. **Reflect Reason**: Analyze why this specific decision was wrong based on the context at that timestep. This is the core lesson learned from the failure.
3. **Corrected Action**: What JSON action should have been taken at that exact timestep instead?
4. **Reflection Note**: A memorable lesson for similar future situations.

{format_instructions}"""
        )
        
        templates["success_analysis"] = PromptTemplate(
            input_variables=["goal_instruction", "master_plan", "trajectory_log", "format_instructions"],
            template="""Analyze a successful navigation mission to extract key learnable experiences.

**Mission Goal:** {goal_instruction}
**Master Plan:** {master_plan}
**Trajectory:** {trajectory_log}

**Critical Task:** Analyze each step's decision (movement reason) and its subsequent results to locate the most critical navigation or search decision frame that led to success.

**Required Analysis:**
1. **Identify Critical Turning Point**: Review the trajectory and identify the exact timestep index where the agent made a critical successful decision. Return this as an integer.
2. **Reflect Reason**: Analyze why this specific decision was effective based on the context at that timestep. This is the core lesson learned from the success.
3. **Successful Action**: The JSON action that was taken at that exact timestep.
4. **Reflection Note**: A memorable lesson for similar future situations.

{format_instructions}"""
        )

        templates["main_plan_reflection_prompt"] = PromptTemplate(
            input_variables=["task_instruction", "master_plan", "trajectory_history", "final_reason", "format_instructions"],
            template="""You are an expert in autonomous navigation planning. Analyze the effectiveness of a strategic plan and provide optimization insights.

**Task Instruction:** {task_instruction}

**Original Master Plan:** {master_plan}

**Complete Trajectory History:** {trajectory_history}

**Final Mission Status:** {final_reason}

**Required Analysis:**
1. **Effectiveness Analysis**: Deeply analyze the original plan's effectiveness, efficiency, and potential defects throughout the mission execution.
2. **Plan Optimization**: 
   - If the mission failed: Generate a corrected plan that is more logically rigorous and more likely to succeed.
   - If the mission succeeded: Consider if there exists a more optimal, more efficient path or strategy, and generate an optimized plan.
3. **Reflection Note**: Extract the complete analysis and thinking process as a reflection note.

{format_instructions}"""
        )

        return templates
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a specific prompt template by name."""
        return self.templates.get(template_name)
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with given variables."""
        template = self.get_template(template_name)
        if template is None:
            logger.error(f"Template '{template_name}' not found")
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable {e} for template '{template_name}'")
            return ""
        except Exception as e:
            logger.error(f"Error formatting template '{template_name}': {e}")
            return ""
    
    def get_all_template_names(self) -> list:
        """Get list of all available template names."""
        return list(self.templates.keys())
    
    def validate_template_variables(self, template_name: str, variables: Dict[str, Any]) -> bool:
        """Validate that all required variables are provided for a template."""
        template = self.get_template(template_name)
        if template is None:
            return False
        
        required_vars = set(template.input_variables)
        provided_vars = set(variables.keys())
        
        missing_vars = required_vars - provided_vars
        if missing_vars:
            logger.warning(f"Missing variables for template '{template_name}': {missing_vars}")
            return False
        
        return True

# ===== Global Instance =====
# Create a global instance for easy import
prompt_manager = PromptManager()

# ===== Legacy Compatibility Functions =====
# These functions provide backward compatibility with existing code

def get_prompt_templates() -> Dict[str, str]:
    """
    Legacy function to get all prompts as simple strings.
    Used for backward compatibility with existing LLMManager.
    """
    legacy_prompts = {}
    
    # Map new template names to legacy names
    name_mapping = {
        "goal_description_nav": "goal_description_nav",
        "goal_description_search": "goal_description_sea", 
        "goal_description_locate": "goal_description_loc",
        "landmark_navigation": "landmark_navigate",
        "object_search": "object_search",
        "target_locate": "object_locate",
        "planner_prompt": "planner_prompt",
        "scene_caption": "scene_caption",
        "landmark_description": "landmark_prompt",
        "navigation_strategy_decision": "navigation_strategy_decision",  # 🔧 Fixed: Add missing mapping
        "search_decision": "search_decision",  # 🔧 Fixed: Add missing mapping
        "local_graph": "local_graph",  # 🔧 Fixed: Add missing mapping
        "query_operation_chain": "query_operation_chain",  # 🔧 Fixed: Add missing mapping
        "reflection_system": "reflection_system",
        "failure_analysis": "failure_analysis",
        "success_analysis": "success_analysis",
        "main_plan_reflection_prompt": "main_plan_reflection_prompt"
    }
    
    for new_name, legacy_name in name_mapping.items():
        template = prompt_manager.get_template(new_name)
        if template:
            legacy_prompts[legacy_name] = template.template
    
    return legacy_prompts

# ===== Export for easy import =====
__all__ = ['PromptManager', 'prompt_manager', 'get_prompt_templates'] 