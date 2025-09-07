"""
Simple script to run episode analysis with different agents
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episode_analysis import run_episode_with_tracking, create_comprehensive_plots, create_detailed_plots
from environment.core import DynamicOvenBatchingEnv
from agents.random_agent import RandomAgent
from agents.fifo_agent import FIFOAgent
from agents.ppo_agent import BasicPPOAgent


def run_analysis_with_agent(agent_name: str, agent, env_params: dict = None):
    """Run episode analysis with a specific agent"""
    
    # Default environment parameters
    if env_params is None:
        env_params = {
            'num_ovens': 2,
            'oven_capacity': 9,
            'batch_time': 10.0,
            'horizon': 1440.0,  # 24 hours
            'arrival_rate': 0.5,
            'use_dynamic_arrivals': True,
            'seed': 42
        }
    
    # Create environment
    env = DynamicOvenBatchingEnv(**env_params)
    
    print(f"\n{'='*60}")
    print(f"Running episode analysis with {agent_name} agent")
    print(f"{'='*60}")
    
    # Run episode with tracking
    tracker = run_episode_with_tracking(env, agent, max_steps=2000)
    
    print(f"Episode completed! Total steps: {tracker.episode_length}")
    
    # Create plots with agent-specific filenames
    base_filename = f"results/{agent_name.lower()}_episode_analysis"
    
    print("Creating comprehensive plots...")
    create_comprehensive_plots(tracker, save_path=f"{base_filename}.png")
    
    print("Creating detailed plots...")
    create_detailed_plots(tracker, save_path=f"{base_filename}.png")
    
    # Save data to CSV
    df = tracker.get_dataframe()
    csv_path = f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Episode data saved to: {csv_path}")
    
    # Print summary statistics
    print(f"\n{agent_name.upper()} AGENT SUMMARY")
    print("-" * 40)
    print(f"Total Steps: {len(df)}")
    print(f"Final Time: {df['current_time'].iloc[-1]:.1f} minutes")
    print(f"Final Queue Length: {df['queue_length'].iloc[-1]}")
    print(f"Total Completed Jobs: {df['completed_jobs'].iloc[-1]}")
    print(f"Total Late Jobs: {df['late_jobs_count'].iloc[-1]}")
    print(f"Final Total Reward: {df['total_reward'].iloc[-1]:.2f}")
    print(f"Final Energy Cost: {df['total_energy_cost'].iloc[-1]:.2f}")
    print(f"Final Lateness Penalty: {df['total_lateness_penalty'].iloc[-1]:.2f}")
    
    # Action distribution
    action_counts = df['action_type'].value_counts()
    action_labels = ['Wait', 'Launch', 'Heat']
    print(f"\nAction Distribution:")
    for i, label in enumerate(action_labels):
        count = action_counts.get(i, 0)
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return tracker


def main():
    """Run analysis with all available agents"""
    # Create environment
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
    
    # Create environment for agent initialization
    env = DynamicOvenBatchingEnv(**env_config)
    
    # Define agents
    agents = {
        'Random': RandomAgent(env),
        'FIFO': FIFOAgent(env),
    }
    
    # Try to add PPO agent if model exists
    try:
        ppo_agent = BasicPPOAgent(env)
        ppo_agent.load_model('saved_models/ppo_model')
        agents['PPO'] = ppo_agent
        print("PPO model found and loaded!")
    except Exception as e:
        print(f"PPO model not found or failed to load: {e}")
        print("Skipping PPO agent analysis.")
    
    # Run analysis with each agent
    trackers = {}
    for agent_name, agent in agents.items():
        try:
            tracker = run_analysis_with_agent(agent_name, agent, env_config)
            trackers[agent_name] = tracker
        except Exception as e:
            print(f"Error running analysis with {agent_name} agent: {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in the 'results/' directory")
    print("Files created:")
    for agent_name in trackers.keys():
        print(f"  - {agent_name.lower()}_episode_analysis.png")
        print(f"  - {agent_name.lower()}_episode_analysis_detailed.png")
        print(f"  - {agent_name.lower()}_episode_analysis.csv")


if __name__ == "__main__":
    main()
