"""
Simple PPO Training and Agent Comparison Script
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from environment.core import DynamicOvenBatchingEnv
from agents.ppo_agent import BasicPPOAgent
from agents.random_agent import RandomAgent
from agents.fifo_agent import FIFOAgent


def main():
    """Main function to train PPO and compare with other agents"""
    print("Starting PPO Agent Training and Comparison")
    print("=" * 50)
    
    # Create environment with improved reward system
    env_config = {
        'num_ovens': 2,
        'oven_capacity': 9,
        'batch_time': 10.0,
        'batch_energy_cost': 9.0,
        'heating_time': 5.0,
        'cooling_rate': 0.1,
        'horizon': 1440.0,
        'arrival_rate': 10,
        'due_date_offset_mean': 60.0,
        'due_date_offset_std': 30.0,
        'energy_alpha': 2.0,  # Energy penalty multiplier
        'lateness_beta': 1.0,  # Lateness penalty multiplier
        'use_dynamic_arrivals': True,
        'time_step': 1.0
    }
    env = DynamicOvenBatchingEnv(**env_config)
    print(f"Environment created with {env.unwrapped.num_ovens} ovens")
    
    # Training parameters - increased for better learning
    total_timesteps = 1.440 * 10**6  # Increased significantly
    num_evaluation_episodes = 10  # Reduced for faster evaluation
    
    # Directories
    save_dir = "./saved_models"
    results_dir = "./results"
    
    # Train PPO agent with better hyperparameters
    print("Training PPO agent...")
    ppo_agent = BasicPPOAgent(env, learning_rate=3e-4,  # Reduced learning rate
                              hyperparameters={'n_steps': 2048,  
                                                'batch_size': 64,  
                                                'n_epochs': 4,  # Fewer epochs
                                                'gamma': 0.99, 
                                                'gae_lambda': 0.95, 
                                                'clip_range': 0.2,
                                                'ent_coef': 0.1,  # Encourage exploration
                                                'vf_coef': 0.5,
                                                'max_grad_norm': 0.5,
                                                'verbose': 0}) 
    
    ppo_agent.train(total_timesteps=total_timesteps, log_dir=save_dir)
    
    # Create other agents
    random_agent = RandomAgent(env)
    fifo_agent = FIFOAgent(env)
    
    # Evaluate all agents
    agents = {
        'PPO': ppo_agent,
        'Random': random_agent,
        'FIFO': fifo_agent
    }
    
    all_results = {}
    
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name} agent...")
        
        results = {
            'episode': [],
            'total_reward': [],
            'total_energy_cost': [],
            'total_lateness_penalty': [],
            'jobs_completed': [],
            'jobs_late': [],
            'throughput': []
        }
        
        for episode in range(num_evaluation_episodes):
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            episode_energy_cost = 0
            episode_lateness_penalty = 0
            jobs_completed = 0
            jobs_late = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                # Extract metrics from info if available
                if 'total_energy_cost' in info:
                    episode_energy_cost = info['total_energy_cost']
                if 'total_lateness_penalty' in info:
                    episode_lateness_penalty = info['total_lateness_penalty']
                if 'completed_jobs' in info:
                    jobs_completed = info['completed_jobs']
                if 'late_jobs_count' in info:
                    jobs_late = info['late_jobs_count']
                if 'total_reward' in info:
                    episode_reward = info['total_reward']
            
            # Calculate throughput
            throughput = jobs_completed / env.unwrapped.horizon if env.unwrapped.horizon > 0 else 0
            
            results['episode'].append(episode)
            results['total_reward'].append(episode_reward)
            results['total_energy_cost'].append(episode_energy_cost)
            results['total_lateness_penalty'].append(episode_lateness_penalty)
            results['jobs_completed'].append(jobs_completed)
            results['jobs_late'].append(jobs_late)
            results['throughput'].append(throughput)
        
        all_results[agent_name] = pd.DataFrame(results)
    
    # Create comparison summary
    comparison_summary = []
    
    for agent_name, results in all_results.items():
        summary = {
            'Agent': agent_name,
            'Avg_Total_Reward': results['total_reward'].mean(),
            'Std_Total_Reward': results['total_reward'].std(),
            'Avg_Energy_Cost': results['total_energy_cost'].mean(),
            'Avg_Lateness_Penalty': results['total_lateness_penalty'].mean(),
            'Avg_Jobs_Completed': results['jobs_completed'].mean(),
            'Avg_Jobs_Late': results['jobs_late'].mean(),
            'Avg_Throughput': results['throughput'].mean()
        }
        comparison_summary.append(summary)
    
    comparison_df = pd.DataFrame(comparison_summary)
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results for each agent
    for agent_name, results in all_results.items():
        results.to_csv(f"{results_dir}/{agent_name.lower()}_results.csv", index=False)
    
    # Save comparison summary
    comparison_df.to_csv(f"{results_dir}/agent_comparison_summary.csv", index=False)
    
    # Print comparison summary
    print("\n" + "=" * 50)
    print("AGENT COMPARISON SUMMARY")
    print("=" * 50)
    print(comparison_df.to_string(index=False))
    
    print(f"\nResults saved to: {results_dir}")
    print(f"PPO model saved to: {save_dir}")
    print("Training and comparison completed!")


if __name__ == "__main__":
    main()
