

### **This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.**

### Overview

ReflectionNavV3 is a sophisticated autonomous navigation agent system that uses multi-modal RAG (Retrieval Augmented Generation) for UAV navigation in city environments. It combines visual understanding, experience-based learning, and reflection mechanisms to improve navigation performance over time.

### Architecture

#### ReflectionNav Agent: 详细开发文档 (V2.1 修订版)

**1. 核心理念 💡**
智能体的核心是一个全局状态 (AgentState)，它作为所有操作的中央数据枢纽。整个任务流程被构建为一个有向图 (Graph)，其中每个节点代表一个具体的操作（如感知、规划），而边则由路由器 (Router) 根据当前状态动态决定。智能体首先进行环境感知和规划，然后进入一个由导航 (Navigate)、搜索 (Search)、定位 (Locate) 三个专家工作流组成的子任务循环。任务结束后，系统会进入反思 (Reflection) 阶段，从成功或失败的经验中学习，并将知识存入记忆模块以供未来使用。

**2. Agent 全局状态 (AgentState) 📚**
`AgentState` 是一个贯穿任务始终的全局数据字典，包含了智能体运行所需的所有静态和动态信息。

**2.1. 静态组件 (Static Components)**
这些组件在任务开始时一次性初始化，并在整个过程中保持不变。

  * **controller**: `LLMController`: 智能体的底层控制器，负责与环境交互（如移动、感知）和执行基于场景图的查询。
  * **landmark\_nav\_map**: `LandmarkNavMap`: 管理地标、语义和环境地图的模块。它会根据感知到的新信息进行实时更新。
  * **episode**: `Episode`: 包含当前任务的所有静态信息，如目标描述、地图名称、起始位置等。
  * **args**: `ExperimentArgs`: 实验配置参数。
  * **task\_instruction**: `str`: 来自 `episode` 的顶层任务目标描述。
  * **memory\_module**: `MultiModalMemory`: 长期记忆中心。这是一个向量索引的记忆数据库，包含了多个集合（collections），用于存储和检索。
      * **集合 1: `strategic_plan_memory`** - 历史任务的主任务规划的经验教训。
          * 通过 `task_instruction` 检索。
          * `task_instruction`: `str` - 任务的原始指令，也是向量化的来源。
          * `master_plan`: `str` - 完整的、以 JSON 字符串形式存储的原始规划。
          * `reflection_note`: `str` - (Reflection)产生的对原始规划的分析和思考过程。
          * `plan_refined`: `str` - (Reflection)而产生的修正或优化的规划。JSON 字符串形式的完整修正后规划/如果原本规划成功完成了任务，包含JSON 字符串形式的完整优化后规划。
          * `status`: `str` - 任务的最终状态，如 'success' 或 'failure'。
      * **集合 2: `navigate_experience_memory`** - 导航相关的经验教训和对应语义图片。
          * **检索方式**: **使用完整的 `current_state` 字符串作为查询向量进行检索**。
          * `navigation_goal`: `str` - 导航子任务的具体目标。
          * `is_success`: `str` - 该经验是否来自一个成功的决策 (**值为 'success' 或 'failure'**)。
          * `reflect_reason`: `str` - **核心经验教训**。对于**失败经验**，这是对失败根本原因的深入分析；对于**成功经验**，这是对决策为何有效以及如何优化的总结。
          * `refined_action`: `str` - **修正或确认的动作**。对于**失败经验**，这是反思后生成的、本应执行的**修正动作**；对于**成功经验**，这就是当时执行并被验证为有效的**原始动作**。两者都以JSON字符串形式存储。
          * `current_state`: `str` - 关键时刻的智能体状态快照。
          * `semantic_enhanced_image_b64`: `str` - 对应的语义地图的 **Base64 编码字符串**。
      * **集合 3: `search_and_locate_memory`** - 搜索相关的经验教训和对应语义图片。
          * **检索方式**: **使用完整的 `current_state` 字符串作为查询向量进行检索**。
          * `search_goal`: `str` - 搜索子任务的具体目标。
          * `is_success`: `str` - 该经验是否来自一个成功的决策 (**值为 'success' 或 'failure'**)。
          * `reflect_reason`: `str` - **核心经验教训**。对于**失败经验**，这是对失败根本原因的深入分析；对于**成功经验**，这是对决策为何有效以及如何优化的总结。
          * `refined_action`: `str` - **修正或确认的动作**。对于**失败经验**，这是反思后生成的、本应执行的**修正动作**；对于**成功经验**，这就是当时执行并被验证为有效的**原始动作**。两者都以JSON字符串形式存储。
          * `current_state`: `str` - 关键时刻的智能体状态快照。
          * `semantic_enhanced_image_b64`: `str` - 对应的语义地图的 **Base64 编码字符串**。

**2.2. 动态状态 (Dynamic State)**
这些状态在任务执行期间会不断被更新。

  * `plan`: `Optional[dict]`: 由 `generate_plan_with_rag` 节点生成的高级分步计划，包含 `sub_goals` 列表和 `reason`推理过程。
  * `current_task_index`: `int`: 指向当前正在执行的子任务在 `plan` 中的索引。
  * `timestep`: `int`: 模拟环境中的当前时间步。
  * `current_observation`: `dict`: 当前时间步的直接感知结果。这是一个字典，包含：
      * `'image'`: `np.ndarray` - 原始的 RGB 图像。
      * `'image_b64'`: `str` - Base64 编码的图像字符串。
      * `'scene_description'`: `str` - 由 VLM 生成的场景文字描述。
      * `'geoinstruct'`: `str` - 结合了地标、周围物体的地理空间上下文描述。
  * `subtask_context`: `dict`: 专家交接备忘录。用于在子任务步骤之间传递上下文，包含上一步的 `{ goal, CoT_summary, movement }`。
  * `current_state`: `str`: 为 RAG 和反思设计的状态快照。这是一个复合字符串，格式为：`"Current aerial perspective observation: {'scene_description': str+'geoinstruct': str}. Task: {顶层任务指令}. Current Goal: {子任务目标} (Strategy: {当前策略})"`。
  * `trajectory_history`: `List[FrameLog]`: 完整的轨迹历史记录。用于为最终的反思模块记录每个时间帧的详细日志。`FrameLog` 包含：
      * `current_state`
      * `movement reason` (专家输出的JSON)
      * `map_snapshot` (该步骤的地图图像)
      * `timestep`, `pose` (x,y,z,yaw), `distance_to_target`

**2.3. 最终结果 (Final Result)**
在任务结束时设置，用于记录最终的执行结果。

  * `mission_success`: `bool`: 任务是否成功完成的标志。
  * `final_reason`: `str`: 任务结束的原因，例如 'success', 'timeout', 'goal\_unreachable'。

**3. 主工作流图 (Main Workflow Graph) 🗺️**
主工作流是智能体的高层控制循环，负责协调感知、规划和子任务执行。

**3.1. 感知决策 (should\_perceive? Router)**

  * **决策问题**: 是否需要执行感知操作？
  * **分支**:
      * **是 (Yes)**: 如果是任务的第一步，或达到了预设的感知时间/距离间隔。流向 `perceive_and_model` 节点。
      * **否 (No)**: 如果未达到感知间隔。跳过感知，直接流向规划决策。

**3.2. 节点 A: perceive\_and\_model (感知与建模)**

  * **职责**: 感知外部世界，并更新智能体的内部世界模型和短期记忆。
  * **输入**: `controller.pose` (当前位姿)。
  * **核心逻辑与信息存入**:
    1.  **获取观测**: 调用 `controller.perceive` 获得原始 RGB 图像。
    2.  **更新地图**: 调用 `landmark_nav_map.update_observations`，将新的 RGB 图像信息融合进内部的导航地图中。
    3.  **生成描述**: 调用 VLM (`llm_manager.generate_image_caption`) 为当前 RGB 图像生成一段文字描述 (`scene_description`)。
    4.  **更新场景图**: **调用 `controller.understand` 和 `controller.build_scene_graph` 来感知并更新场景图数据库，确保持续构建对环境的全面认知。**
    5.  **构建地理指令**: 综合地标、场景图物体等信息，构建 `geoinstruct` 字符串。
    6.  **更新 AgentState (短期记忆)**:
          * 将上述结果存入 `AgentState.current_observation` 字典中。
  * **流向**: `should_generate_plan?` 路由器。

**3.3. 规划决策 (should\_generate\_plan? Router)**

  * **决策问题**: 是否需要生成新的高级计划？
  * **分支**:
      * **是 (Yes)**: 如果 `AgentState.plan` 为 `None`。流向 `generate_plan_with_rag` 节点。
      * **否 (No)**: 如果计划已存在。流向 `execute_subtask_router` 节点，开始执行子任务。

**3.4. 节点 C: generate\_plan\_with\_rag (生成规划)**

  * **职责**: 结合从记忆 `mainplan_experience_memory` 中检索到的相似经验，为当前任务创建一个高层次、分步骤的计划。
  * **核心逻辑**:
    1.  **RAG**:
          * 以 `task_instruction` 为查询，从 `mainplan_experience_memory` 中检索最相似的规划作为 `refined_plan` 和对应的思考 `reflect_reason`。
    2.  **Prompt**: 将 RAG 检索到的 `refined_plan` 和 `reflect_reason`，连同当前观测 `geoinstruct` 和任务目标 `task_instruction`，一起填入 `planner_prompt` 模板。
    3.  **LLM 调用**: 请求大语言模型生成一个结构化的计划。其输出被 `PlanOutput` 模型解析，包含 `sub_goals` 列表（每个sub-goal含`goal`, `strategy`, `desired_state`）和 `reason`。
  * **输出**: 更新 `plan`, `current_task_index` (设为0), `subtask_context`。
  * **流向**: `execute_subtask_router` 节点。

**3.5. 节点 D: execute\_subtask\_router (子任务路由)**

  * **职责**: 根据 `plan` 中当前子任务定义的策略 (`strategy`)，将工作流分派给相应的专家。同时，它会构建统一的 `current_state` 快照，供后续的 RAG 和反思模块使用。
  * **输入**: `plan`, `current_task_index`, `current_observation`, `task_instruction`。
  * **输出**: `current_state` (格式: `"Current aerial perspective observation: {观测描述}. Task: {顶层任务指令}. Current Goal: {子任务目标} (Strategy: {当前策略})"`).
  * **分支**:
      * 如果 `strategy` 是 'navigate'，流向 Navigate (导航) 专家工作流。
      * 如果 `strategy` 是 'search'，流向 Search (搜索) 专家工作流。
      * 如果 `strategy` 是 'locate'，流向 Locate (定位) 专家工作流。
  * **注意**：`Locate` 专家不再被强制为最后一个节点，可以根据计划在流程中被多次调用。

**3.6. 节点 finalize\_step (最终化步骤)**

  * **职责**: 在 Search (搜索) 和 Navigate (导航) 专家工作流执行动作后，此节点负责更新智能体的物理状态（位姿、时间步）并记录轨迹历史。
  * **输入**: `movement` (专家输出), `reason` (专家输出)。
  * **输出**:
      * `controller.pose` (移动到新位姿)。
      * `timestep` (加1)。
      * `trajectory_history`: 添加一条包含当前所有详细信息的 `FrameLog` 记录。
  * **流向**: `should_continue?` 路由器。

**3.7. 循环/结束决策 (should\_continue? Router)**

  * **决策问题**: 当前任务是否应该继续？
  * **分支**:
      * **继续循环 (Continue Loop)**: 如果任务未完成（例如，未超时且未到达最终目标）。流回到主循环的起点 `should_perceive?` 路由器。
      * **结束任务 (End Mission)**: 如果任务已完成或失败。流向 `check_mission_success` 路由器。

**4. 子任务专家工作流 (Sub-Task Expert Workflows) 🧭**
每个专家工作流负责执行一种特定类型的子任务。

**4.1. Navigate (导航) 专家**

  * **目标**: 导航至一个指定的地标。
  * **RAG**: **使用在子任务路由阶段生成的完整 `current_state` 字符串作为查询向量**，从 `navigation_experience_memory` 中检索相关的 `refined_action` `reflect_reason` 和 `retrieved_image_b64` (对应语义图片)。
  * **执行 (execute\_navigate)**:
      * **Prompt**: 结合 `subtask_context`, `current_state` 和 RAG 经验 (`refined_action`, `reflect_reason`)。**VLM 的输入会按顺序包含两张图片：第一张是被标记为“当前实时地图”的实时地标地图 (`gen_map(query_type='landmark')`)，第二张是被标记为“历史经验参考地图”的 `retrieved_image_b64`。Prompt中会明确指示模型两张图各自的身份，引导其进行对比分析。**
      * **输出**: LLM 的输出被 `NavigationDecisionOutput` 模型解析为一个包含 `movement`, `reason`, `next_strategy`, `confidence` 的结构化对象，存入 `action_result`。
  * **完成检查 (is\_landmark\_reached?)**:
      * **是**: 如果与地标的物理距离小于阈值，或 `next_strategy` 决策为 'next\_subtask'。流向 Finalize & Advance。
      * **否**: 否则流向 Finalize & Loop。

**4.2. Search (搜索) 专家**

  * **目标**: 在一个区域内搜索特定目标或信息。
  * **RAG**: **使用在子任务路由阶段生成的完整 `current_state` 字符串作为查询向量**，从 `search_experience_memory` 检索相关的 `refined_action` `reflect_reason` 和 `retrieved_image_b64` (对应语义图片)。
  * **执行 (execute\_search)**:
      * **Prompt**: 结合 `subtask_context`, `current_state` 和 RAG 经验 (`refined_action`, `reflect_reason`)。**VLM 的输入会按顺序包含两张图片：第一张是被标记为“当前实时地图”的实时语义地图 (`gen_map(query_type='semantic')`)，第二张是被标记为“历史经验参考地图”的 `retrieved_image_b64`。Prompt中会明确指示模型两张图各自的身份，引导其进行对比分析。**
      * **输出**: LLM 的输出被 `SearchOutput` 模型解析为一个包含 `movement`, `reason`, `decision`, `confidence` 的结构化对象，存入 `action_result`。
  * **完成检查 (is\_target\_info\_sufficient?)**:
      * **是**: 如果 `decision` 为 'next\_subtask' 或搜索超时。流向 Finalize & Advance。
      * **否**: 如果 `decision` 为 'continue\_search'，流向 Finalize & Loop。

**4.3. Locate (定位) 专家**

  * **目标**: 精确定位一个已经大致发现的目标。
  * **执行 (execute\_locate)**:
      * **核心动作**: 优先使用 `controller.query_engine.robust_subgraph_query` 对场景图进行结构化查询。如果失败，则回退到 VLM。
      * **Prompt (VLM回退时)**: 结合 `subtask_context`, `current_state` 和当前观测图像，直接请求 VLM 给出目标坐标。
      * **输出**: 输出被 `LocateOutput` 模型解析，包含 `status` ('TARGET\_LOCKED' 或 'SEARCHING\_VICINITY'), `selected_pos` 等。
  * **完成检查**: 定位是一个原子操作，执行后直接流向 Finalize & Advance。

**5. 任务结束、反思与学习 (End-of-Mission, Reflection & Learning) 🧠**
当主循环结束后，系统进入反思阶段，对整个任务过程进行复盘和学习。此阶段分为两个层面：对高层主规划的反思，以及对底层子任务执行的经验总结。

**5.1. 成功子任务经验总结 (summarize\_success\_experience)**
当 `check_mission_success` 路由器判断任务成功时，触发此流程以总结子任务的关键决策。

  * **读取轨迹**: 读取 `AgentState.trajectory_history`。
  * **LLM分析**: 将完整的 `trajectory_history` (作为JSON) 和 `master_plan` 发送给 LLM，使用 `success_analysis` 提示词，**要求其识别出导致任务成功的决策转折点，并返回该转折点发生时的 `timestep` 索引。模型会分析每一步的决策（`movement reason`）及其后续结果，定位到最关键的导航和搜索决策帧**，并对该帧形成：
      * `reflect_reason`: `str` - 核心经验教训，这是对成功路径的优化思考。
      * `refined_action`: `str` - 当时执行的成功的原始动作。
  * **写入记忆**: 调用 `memory_module.add`，根据大模型提取的关键帧的`timestep`,得到对应的语义图片，`current_state`，`navigation_goal`/`search_goal` 和反思笔记 (`reflect_reason`) 及 `refined_action` 一同存入相应的经验数据库中 (`navigation_experience_memory` 或 `search_experience_memory`)。

**5.2. 失败子任务经验反思 (reflect\_on\_failure)**
当 `check_mission_success` 路由器判断任务失败时，触发此流程以反思子任务的失败原因。

  * **准备输入**: 将 `failure_reason` 和完整的 `trajectory_history` (作为JSON) 填入 `failure_analysis` 提示词模板。
  * **LLM反思**: 将完整的 `trajectory_history` (作为JSON) 和 `master_plan` 发送给 LLM，使用 `failure_analysis` 提示词，**要求其识别出导致任务失败的决策转折点，并返回该转折点发生时的 `timestep` 索引。模型会分析每一步的决策（`movement reason`）及其后续结果，定位到最关键的导航和搜索决策帧**，并对该帧形成：
      * `reflect_reason`: `str` - 核心经验教训，这是对失败原因的思考和反思。
      * `refined_action`: `str` - 修正后应该执行的动作。
  * **写入记忆**: 调用 `memory_module.add`，根据大模型提取的关键帧的`timestep`,得到对应的语义图片，`current_state`，`navigation_goal`/`search_goal` 和反思笔记 (`reflect_reason`) 及 `refined_action` 一同存入相应的经验数据库中 (`navigation_experience_memory` 或 `search_experience_memory`)。

**5.3. 主规划反思与优化 (reflect\_on\_main\_plan)**
此流程在任务结束后触发，独立于子任务经验总结，专注于优化高层的主规划，为 `mainplan_experience_memory` 提供学习内容。

  * **LLM分析与重写**: 将原始主规划 `master_plan`、完整的 `trajectory_history` 以及任务最终状态 `final_reason` 提供给一个强大的LLM。使用专用的 `main_plan_reflection_prompt` 提示词，要求模型：
      * **分析**：深入分析原始规划在整个任务过程中的有效性、效率和潜在缺陷。
      * **重写**：
          * 如果任务失败，生成一个修正后的、逻辑更严谨、更可能成功的规划 (`plan_refined`)。
          * 如果任务成功，思考是否存在更优、更高效的路径或策略，并生成一个优化后的规划 (`plan_refined`)。
      * **生成反思笔记**: LLM 对规划的完整分析和思考过程被提炼为 `reflection_note`。
  * **写入记忆**: 调用 `memory_module.add`，将原始的 `task_instruction`、`master_plan`、新生成的 `plan_refined` (JSON 字符串)、`reflection_note` 以及任务状态 `status` 一同写入 `mainplan_experience_memory` 集合中。

**6. 其他**

  * 大模型输出的 `reason` 是其 `thought` 部分(CoT)的总结，请你在prompt中明确。


#### Data Organization

`results/ReflectionNavV3_[episode_id]/`
├── `trajectory.json`          \# Mission execution log
├── `episode_summary.json`       \# Mission outcome (success/failure)
├── `rgb_images/`
│   └── `rgb_timestep_XXXX.png`   \# RGB visual snapshots
└── `semantic_images/`
├── `landmark_[id]_step_XXXX.png`  \# Semantic maps for navigation
└── `semantic_[id]_step_XXXX.png`  \# Semantic maps for search



### **Configuration**

#### Key Settings (config.yaml)

  * **Memory**: `experience_db_path`, `max_retrieval_results`, `similarity_threshold`
  * **Navigation**: `max_search_radius`, `search_threshold`, `action_scale`
  * **Strategy**: `Maps_timeout`, `search_timeout`, `landmark_proximity`
  * **RAG**: `retrieval_interval`, `context_window_size`
  * **Reflection**: `enable_reflection`, `max_key_experiences`
"
```

### **Key Dependencies**

  * LangGraph: Workflow orchestration
  * ChromaDB: Vector database for experience storage
  * PIL/Pillow: Image processing and context loading
  * LangChain: Prompt management and LLM integration
  * OpenAI: LLM services and embeddings

