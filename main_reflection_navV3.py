#!/usr/bin/env python3
"""
Main Entry Point for ReflectionNav_AgentV3

This script serves as the main executable for running the ReflectionNav_AgentV3 agent.
It handles argument parsing, model initialization, and episode execution following
the project's non-destructive development principle.
"""

import os
import json
import torch
from openai import OpenAI
import numpy as np
from tqdm import trange
import pickle

from scenegraphnav.parser import parse_args
from gsamllavanav.evaluate import eval_planning_metrics
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from scenegraphnav.evaluate import run_episodes_batch
from gsamllavanav.observation import cropclient

DEVICE = 'cuda'
test_data = 'hard'

def initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key):
    vlmodel = None
    llmodel = None

    if VLM_backbone == 'Qwen-online':
        vlmodel = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif VLM_backbone == 'GPT':
        vlmodel = OpenAI(
            api_key=vl_api_key,
            base_url="OPENAI_BASE_URL",
        )

    if LLM_backbone == 'Qwen-online':
        llmodel = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif LLM_backbone == 'GPT':
        llmodel = OpenAI(
            api_key=ll_api_key,
            base_url="OPENAI_BASE_URL",
        )
    return vlmodel, llmodel

def get_cached_episodes(split, test_data, altitude, objects, max_episodes=None):
    cache_dir = "data/episodes_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    max_episodes_str = 'all' if max_episodes is None else str(max_episodes)
    cache_file = os.path.join(cache_dir, f"{split}_{test_data}_{altitude}_{max_episodes_str}_episodes.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached episodes from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Generating new episodes...")
        episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(split, test_data, altitude),
            max_episodes=max_episodes
        )
        print(f"Saving generated episodes to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(episodes, f)
        return episodes

def main():
    args = parse_args()
    # Load image cache before running any episodes
    cropclient.load_image_cache()

    # Load CityRefer objects
    objects = get_city_refer_objects()

    try:
        # Try to import the agent (this will fail until the modules are implemented)
        from reflectionnav_agentV3 import ReflectionNav_AgentV3, OpenAIEmbeddingProvider
        
        # Create a single embedding provider instance
        embedding_provider = OpenAIEmbeddingProvider()

        # Use the episode caching function
        if args.test_one_example:
            test_episodes = get_cached_episodes(args.split, 'all', args.altitude, objects, max_episodes=1)
        else:
            test_episodes = get_cached_episodes(args.split, test_data, args.altitude, objects, max_episodes=None)
        
        trajectory_logs = {}
        
        VLM_backbone = 'Qwen-online'
        LLM_backbone = 'Qwen-online'
        vl_api_key = os.getenv("DASHSCOPE_API_KEY")
        ll_api_key = os.getenv("DASHSCOPE_API_KEY")
        vlmodel, llmodel = initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key)

        # Validate vlmodel before proceeding
        if vlmodel is None:
            print("Error: Could not initialize VLM model. Please check your API key and configuration.")
            return

        for episode in test_episodes:
            print(f"--- Running Episode: {episode.id} ---")
            print(f"Instruction: {episode.target_description}")
            
            try:
                agent = ReflectionNav_AgentV3(
                    args,
                    episode.start_pose,
                    episode,
                    vlmodel,           
                    set_height=args.set_height,
                    embedding_provider=embedding_provider,
                )
                
                # The 'naive' flag can now be controlled by the reflection switch
                result, trajectory_log = agent.run()
                trajectory_logs[episode.id] = trajectory_log
                
            except Exception as e:
                print(f"Error running episode {episode.id}: {e}")
                continue

        # Note: You might need to adjust the evaluation call if it depends on args
        # that are now parsed differently. This example assumes it's compatible.
        # metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)
        # print(f"{args.split} -- NE {metrics.mean_final_pos_to_goal_dist: .1f}, SR {metrics.success_rate_final_pos_to_goal*100: .2f}, OSR {metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {metrics.success_rate_weighted_by_path_length*100: .2f}metrics")

    except ImportError as e:
        print("Error: ReflectionNav_AgentV3 modules not found!")
        print(f"Import error details: {e}")
        print("Please ensure all required modules are implemented in the reflectionnav_agentV3 directory:")
        print("- reflectionnav_agent_v3.py")
        print("- multimodal_memory.py") 
        print("- llm_manager.py")
        print("- graph_nodes.py")
        print("- agent_state.py")
        print("- reflection.py")
        print("- utils.py")
        print("- prompt_cot.py")
        print("- __init__.py")
        return
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    main()
