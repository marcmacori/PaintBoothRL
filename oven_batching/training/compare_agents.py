"""
Agent Comparison Script

This script trains a PPO agent and compares its performance with
the random and FIFO baseline agents on the oven batching environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os
import time
from tqdm import tqdm

# Import environment and agents
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core import DynamicOvenBatchingEnv
from agents.random_agent import RandomAgent
from agents.fifo_agent import FIFOAgent
from agents.ppo_agent import BasicPPOAgent


class AgentEvaluator:
    """Class to evaluate and compare different agents"""
    
    def __init__(self, env_config: Dict = None):
        """
        Initialize the evaluator
        
        Args:
            env_config: Environment configuration parameters
        """
        self.env_config = env_config or {}
        self.results = {}
        
    def create_environment(self) -> DynamicOvenBatchingEnv:
        """Create a new environment instance"""
        config = {
            'num_ovens': 2,
            'oven_capacity': 9,
            'batch_time': 10.0,
            'batch_energy_cost': 1.0,
            'heating_time': 15.0,
            'cooling_rate': 0.1,
            'horizon': 1440.0,  # 24 hours in minutes
            'arrival_rate': 0.5,
            'due_date_offset_mean': 60.0,
            'due_date_offset_std': 20.0,
            'energy_alpha': 1.0,
            'lateness_beta': 2.0,
            'idle_penalty': 0.01,
            'underfill_penalty': 0.1,
            'use_dynamic_arrivals': True
        }
        config.update(self.env_config)
        return DynamicOvenBatchingEnv(**config)
    
    def evaluate_agent(self, agent, env: DynamicOvenBatchingEnv, 
                      num_episodes: int = 10, render: bool = False) -> Dict:
        """
        Evaluate an agent over multiple episodes
        
        Args:
            agent: The agent to evaluate
            env: The environment to use
            num_episodes: Number of episodes to run
            render: Whether to render the environment
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_throughputs = []
        episode_energy_costs = []
        episode_lateness_penalties = []
        episode_completion_rates = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            agent.reset()
            
            total_reward = 0.0
            total_energy_cost = 0.0
            total_lateness_penalty = 0.0
            completed_jobs = 0
            
            while True:
                if render:
                    env.render()
                
                # Get action from agent
                action, _ = agent.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                total_energy_cost = info.get('total_energy_cost', 0.0)
                total_lateness_penalty = info.get('total_lateness_penalty', 0.0)
                completed_jobs = info.get('completed_jobs', 0)
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_throughputs.append(completed_jobs)
            episode_energy_costs.append(total_energy_cost)
            episode_lateness_penalties.append(total_lateness_penalty)
            
            # Calculate completion rate (completed vs total jobs that arrived)
            total_jobs = completed_jobs + len(env.queue)
            completion_rate = completed_jobs / max(total_jobs, 1)
            episode_completion_rates.append(completion_rate)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_throughput': np.mean(episode_throughputs),
            'std_throughput': np.std(episode_throughputs),
            'mean_energy_cost': np.mean(episode_energy_costs),
            'std_energy_cost': np.std(episode_energy_costs),
            'mean_lateness_penalty': np.mean(episode_lateness_penalties),
            'std_lateness_penalty': np.std(episode_lateness_penalties),
            'mean_completion_rate': np.mean(episode_completion_rates),
            'std_completion_rate': np.std(episode_completion_rates),
            'episode_rewards': episode_rewards,
            'episode_throughputs': episode_throughputs,
            'episode_energy_costs': episode_energy_costs,
            'episode_lateness_penalties': episode_lateness_penalties,
            'episode_completion_rates': episode_completion_rates
        }
    
    def train_ppo_agent(self, training_timesteps: int = 50000, 
                       eval_freq: int = 10000) -> BasicPPOAgent:
        """
        Train a PPO agent
        
        Args:
            training_timesteps: Number of timesteps to train for
            eval_freq: Evaluation frequency during training
            
        Returns:
            Trained PPO agent
        """
        print("Creating training environment...")
        train_env = self.create_environment()
        
        print("Initializing PPO agent...")
        ppo_agent = BasicPPOAgent(
            train_env,
            learning_rate=3e-4
        )
        
        print(f"Training PPO agent for {training_timesteps} timesteps...")
        start_time = time.time()
        ppo_agent.train(
            total_timesteps=training_timesteps,
            log_dir="./ppo_training_logs"
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return ppo_agent
    
    def compare_agents(self, num_episodes: int = 20, 
                      training_timesteps: int = 100000,
                      save_results: bool = True) -> Dict:
        """
        Compare all agents
        
        Args:
            num_episodes: Number of episodes for evaluation
            training_timesteps: Number of timesteps to train PPO agent
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with comparison results
        """
        print("=" * 60)
        print("AGENT COMPARISON EXPERIMENT")
        print("=" * 60)
        
        # Train PPO agent
        print("\n1. Training PPO Agent...")
        ppo_agent = self.train_ppo_agent(training_timesteps=training_timesteps)
        
        # Evaluate all agents
        agents = {
            'Random': RandomAgent,
            'FIFO': FIFOAgent,
            'PPO': lambda env: ppo_agent
        }
        
        results = {}
        
        for agent_name, agent_class in agents.items():
            print(f"\n2. Evaluating {agent_name} Agent...")
            
            # Create evaluation environment
            eval_env = self.create_environment()
            
            # Create agent instance
            if agent_name == 'PPO':
                # For PPO, use the same environment it was trained on
                agent = ppo_agent
                # Update the environment reference
                agent.env = eval_env
            else:
                agent = agent_class(eval_env)
            
            # Evaluate agent
            agent_results = self.evaluate_agent(agent, eval_env, num_episodes=num_episodes)
            results[agent_name] = agent_results
            
            print(f"   Mean Reward: {agent_results['mean_reward']:.2f} ± {agent_results['std_reward']:.2f}")
            print(f"   Mean Throughput: {agent_results['mean_throughput']:.1f} ± {agent_results['std_throughput']:.1f}")
            print(f"   Mean Energy Cost: {agent_results['mean_energy_cost']:.2f} ± {agent_results['std_energy_cost']:.2f}")
            print(f"   Mean Lateness Penalty: {agent_results['mean_lateness_penalty']:.2f} ± {agent_results['std_lateness_penalty']:.2f}")
            print(f"   Mean Completion Rate: {agent_results['mean_completion_rate']:.3f} ± {agent_results['std_completion_rate']:.3f}")
        
        # Save results
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save results to CSV and create plots"""
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results to CSV
        detailed_data = []
        for agent_name, agent_results in results.items():
            for episode in range(len(agent_results['episode_rewards'])):
                detailed_data.append({
                    'Agent': agent_name,
                    'Episode': episode + 1,
                    'Reward': agent_results['episode_rewards'][episode],
                    'Throughput': agent_results['episode_throughputs'][episode],
                    'Energy_Cost': agent_results['episode_energy_costs'][episode],
                    'Lateness_Penalty': agent_results['episode_lateness_penalties'][episode],
                    'Completion_Rate': agent_results['episode_completion_rates'][episode]
                })
        
        df = pd.DataFrame(detailed_data)
        df.to_csv("results/agent_comparison_detailed.csv", index=False)
        
        # Save summary results
        summary_data = []
        for agent_name, agent_results in results.items():
            summary_data.append({
                'Agent': agent_name,
                'Mean_Reward': agent_results['mean_reward'],
                'Std_Reward': agent_results['std_reward'],
                'Mean_Throughput': agent_results['mean_throughput'],
                'Std_Throughput': agent_results['std_throughput'],
                'Mean_Energy_Cost': agent_results['mean_energy_cost'],
                'Std_Energy_Cost': agent_results['std_energy_cost'],
                'Mean_Lateness_Penalty': agent_results['mean_lateness_penalty'],
                'Std_Lateness_Penalty': agent_results['std_lateness_penalty'],
                'Mean_Completion_Rate': agent_results['mean_completion_rate'],
                'Std_Completion_Rate': agent_results['std_completion_rate']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results/agent_comparison_summary.csv", index=False)
        
        # Create comparison plots
        self.create_comparison_plots(results)
        
        print(f"\nResults saved to 'results/' directory")
    
    def create_comparison_plots(self, results: Dict):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agent Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        agents = list(results.keys())
        metrics = ['mean_reward', 'mean_throughput', 'mean_energy_cost', 'mean_completion_rate']
        metric_names = ['Mean Reward', 'Mean Throughput', 'Mean Energy Cost', 'Mean Completion Rate']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            
            values = [results[agent][metric] for agent in agents]
            errors = [results[agent][f'std_{metric.split("_", 1)[1]}'] for agent in agents]
            
            bars = ax.bar(agents, values, yerr=errors, capsize=5, color=colors[:len(agents)])
            ax.set_title(metric_name, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(errors) * 0.1,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/agent_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create learning curve plot for PPO
        if 'PPO' in results:
            self.plot_ppo_learning_curve()
    
    def plot_ppo_learning_curve(self):
        """Plot PPO learning curve if training logs exist"""
        try:
            # This would require reading tensorboard logs or training history
            # For now, we'll create a placeholder
            print("Learning curve plotting would require tensorboard logs")
        except Exception as e:
            print(f"Could not plot learning curve: {e}")


def main():
    """Main function to run the agent comparison"""
    # Create evaluator
    evaluator = AgentEvaluator()
    
    # Run comparison
    results = evaluator.compare_agents(
        num_episodes=20,
        training_timesteps=100000,  # Increased for better learning
        save_results=True
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)
    
    for agent_name, agent_results in results.items():
        print(f"\n{agent_name} Agent:")
        print(f"  Average Reward: {agent_results['mean_reward']:.2f} ± {agent_results['std_reward']:.2f}")
        print(f"  Average Throughput: {agent_results['mean_throughput']:.1f} jobs")
        print(f"  Average Energy Cost: {agent_results['mean_energy_cost']:.2f}")
        print(f"  Average Lateness Penalty: {agent_results['mean_lateness_penalty']:.2f}")
        print(f"  Average Completion Rate: {agent_results['mean_completion_rate']:.1%}")


if __name__ == "__main__":
    main()
