"""
Episode Analysis Script for Oven Batching Environment

This script runs a complete episode of the oven batching environment and tracks
all observations, statistics, rewards, info, and actions for each step.
Then it creates comprehensive plots showing the evolution of all metrics over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core import DynamicOvenBatchingEnv
from agents.random_agent import RandomAgent
from agents.fifo_agent import FIFOAgent
from agents.ppo_agent import BasicPPOAgent


class EpisodeTracker:
    """Tracks all data from an episode for analysis and plotting"""
    
    def __init__(self, env: DynamicOvenBatchingEnv):
        self.env = env
        self.reset()
    
    def reset(self):
        """Reset the tracker for a new episode"""
        self.step_data = []
        self.episode_length = 0
        
    def record_step(self, step: int, observation: np.ndarray, action: np.ndarray, 
                   reward: float, info: Dict[str, Any], terminated: bool, truncated: bool):
        """Record data from a single step"""
        
        # Parse action components
        action_type, oven_id, num_panels = action
        
        # Parse observation components (based on environment structure)
        obs_dim = len(observation)
        num_ovens = self.env.num_ovens
        
        # Global observations (first 6 elements)
        current_time_norm = observation[0]
        queue_length_norm = observation[1]
        avg_waiting_time_norm = observation[2]
        max_lateness_risk_norm = observation[3]
        urgent_jobs_norm = observation[4]
        energy_tariff_norm = observation[5]
        
        # Oven observations (6 per oven)
        oven_data = []
        for i in range(num_ovens):
            start_idx = 6 + i * 6
            oven_data.append({
                'busy': observation[start_idx],
                'heating': observation[start_idx + 1],
                'cooling': observation[start_idx + 2],
                'time_to_completion': observation[start_idx + 3],
                'time_to_ready': observation[start_idx + 4],
                'temperature': observation[start_idx + 5]
            })
        
        # Denormalize values where possible
        current_time = current_time_norm * self.env.horizon
        energy_tariff = energy_tariff_norm * 1.5
        
        # Record step data
        step_record = {
            'step': step,
            'current_time': current_time,
            'current_time_norm': current_time_norm,
            
            # Observations
            'queue_length_norm': queue_length_norm,
            'avg_waiting_time_norm': avg_waiting_time_norm,
            'max_lateness_risk_norm': max_lateness_risk_norm,
            'urgent_jobs_norm': urgent_jobs_norm,
            'energy_tariff_norm': energy_tariff_norm,
            'energy_tariff': energy_tariff,
            
            # Actions
            'action_type': action_type,
            'oven_id': oven_id,
            'num_panels': num_panels,
            
            # Rewards
            'reward': reward,
            'action_validation_reward': info.get('action_validation_reward', 0.0),
            'launch_reward': info.get('launch_reward', 0.0),
            'heating_reward': info.get('heating_reward', 0.0),
            'wait_reward': info.get('wait_reward', 0.0),
            'energy_cost_penalty': info.get('energy_cost_penalty', 0.0),
            'lateness_penalty': info.get('lateness_penalty', 0.0),
            'completion_reward': info.get('completion_reward', 0.0),
            
            # Statistics
            'queue_length': info.get('queue_length', 0),
            'completed_jobs': info.get('completed_jobs', 0),
            'total_reward': info.get('total_reward', 0.0),
            'total_energy_cost': info.get('total_energy_cost', 0.0),
            'total_lateness_penalty': info.get('total_lateness_penalty', 0.0),
            'late_jobs_count': info.get('late_jobs_count', 0),
            
            # Episode status
            'terminated': terminated,
            'truncated': truncated
        }
        
        # Add oven-specific data
        for i, oven_info in enumerate(oven_data):
            for key, value in oven_info.items():
                step_record[f'oven_{i}_{key}'] = value
        
        self.step_data.append(step_record)
        self.episode_length += 1
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert recorded data to pandas DataFrame"""
        return pd.DataFrame(self.step_data)


def run_episode_with_tracking(env: DynamicOvenBatchingEnv, agent, max_steps: int = 1000) -> EpisodeTracker:
    """Run a complete episode while tracking all data"""
    
    tracker = EpisodeTracker(env)
    observation, info = env.reset()
    
    step = 0
    while step < max_steps:
        # Get action from agent
        action, _ = agent.predict(observation)
        
        # Take step in environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Record step data
        tracker.record_step(step, observation, action, reward, info, terminated, truncated)
        
        # Update observation
        observation = next_observation
        step += 1
        
        # Check if episode is done
        if terminated or truncated:
            break
    
    return tracker


def create_comprehensive_plots(tracker: EpisodeTracker, save_path: str = None):
    """Create comprehensive plots of all tracked data"""
    
    df = tracker.get_dataframe()
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle('Oven Batching Environment - Episode Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. Time and Queue Metrics
    ax = axes[0]
    ax.plot(df['current_time'], df['queue_length'], 'b-', label='Queue Length', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Queue Length')
    ax.set_title('Queue Length Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Rewards Breakdown
    ax = axes[1]
    reward_components = ['action_validation_reward', 'launch_reward', 'heating_reward', 
                        'wait_reward', 'energy_cost_penalty', 'lateness_penalty', 'completion_reward']
    for component in reward_components:
        if component in df.columns:
            ax.plot(df['current_time'], df[component], label=component.replace('_', ' ').title())
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Reward Component')
    ax.set_title('Reward Components Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Total Rewards and Penalties
    ax = axes[2]
    ax.plot(df['current_time'], df['total_reward'], 'g-', label='Total Reward', linewidth=2)
    ax.plot(df['current_time'], df['total_energy_cost'], 'r-', label='Total Energy Cost', linewidth=2)
    ax.plot(df['current_time'], df['total_lateness_penalty'], 'orange', label='Total Lateness Penalty', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Cumulative Value')
    ax.set_title('Cumulative Rewards and Penalties')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Completed Jobs and Late Jobs
    ax = axes[3]
    ax.plot(df['current_time'], df['completed_jobs'], 'g-', label='Completed Jobs', linewidth=2)
    ax.plot(df['current_time'], df['late_jobs_count'], 'r-', label='Late Jobs Count', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Count')
    ax.set_title('Job Completion Statistics')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 5. Energy Tariff
    ax = axes[4]
    ax.plot(df['current_time'], df['energy_tariff'], 'purple', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Energy Tariff')
    ax.set_title('Energy Tariff Over Time')
    ax.grid(True, alpha=0.3)
    
    # 6. Action Distribution
    ax = axes[5]
    action_counts = df['action_type'].value_counts()
    action_labels = ['Wait', 'Launch', 'Heat']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    ax.bar(action_labels, [action_counts.get(i, 0) for i in range(3)], color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Action Type Distribution')
    
    # 7. Oven Temperatures (for first oven)
    ax = axes[6]
    if 'oven_0_temperature' in df.columns:
        ax.plot(df['current_time'], df['oven_0_temperature'], 'b-', linewidth=2)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Temperature')
        ax.set_title('Oven 0 Temperature')
        ax.grid(True, alpha=0.3)
    
    # 8. Oven Status (for first oven)
    ax = axes[7]
    if 'oven_0_busy' in df.columns:
        ax.plot(df['current_time'], df['oven_0_busy'], 'g-', label='Busy', linewidth=2)
        ax.plot(df['current_time'], df['oven_0_heating'], 'orange', label='Heating', linewidth=2)
        ax.plot(df['current_time'], df['oven_0_cooling'], 'red', label='Cooling', linewidth=2)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Status')
        ax.set_title('Oven 0 Status')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 9. Normalized Observations
    ax = axes[8]
    obs_metrics = ['queue_length_norm', 'avg_waiting_time_norm', 'max_lateness_risk_norm', 'urgent_jobs_norm']
    for metric in obs_metrics:
        if metric in df.columns:
            ax.plot(df['current_time'], df[metric], label=metric.replace('_norm', '').replace('_', ' ').title())
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized Observations')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 10. Number of Panels per Action
    ax = axes[9]
    ax.plot(df['current_time'], df['num_panels'], 'b-', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Number of Panels')
    ax.set_title('Panels per Action')
    ax.grid(True, alpha=0.3)
    
    # 11. Oven Utilization (for first oven)
    ax = axes[10]
    if 'oven_0_time_to_completion' in df.columns:
        ax.plot(df['current_time'], df['oven_0_time_to_completion'], 'b-', linewidth=2)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Time to Completion')
        ax.set_title('Oven 0 Time to Completion')
        ax.grid(True, alpha=0.3)
    
    # 12. Episode Summary Statistics
    ax = axes[11]
    ax.axis('off')
    summary_text = f"""
Episode Summary:
- Total Steps: {len(df)}
- Final Time: {df['current_time'].iloc[-1]:.1f} minutes
- Final Queue Length: {df['queue_length'].iloc[-1]}
- Total Completed Jobs: {df['completed_jobs'].iloc[-1]}
- Total Late Jobs: {df['late_jobs_count'].iloc[-1]}
- Final Total Reward: {df['total_reward'].iloc[-1]:.2f}
- Final Energy Cost: {df['total_energy_cost'].iloc[-1]:.2f}
- Final Lateness Penalty: {df['total_lateness_penalty'].iloc[-1]:.2f}
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
    
    plt.show()


def create_detailed_plots(tracker: EpisodeTracker, save_path: str = None):
    """Create additional detailed plots for deeper analysis"""
    
    df = tracker.get_dataframe()
    
    # Create a second figure with more detailed plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Detailed Episode Analysis', fontsize=16, fontweight='bold')
    
    # 1. Reward Components Stacked
    ax = axes[0, 0]
    reward_components = ['launch_reward', 'heating_reward', 'completion_reward']
    negative_components = ['action_validation_reward', 'wait_reward', 'energy_cost_penalty', 'lateness_penalty']
    
    # Stack positive rewards
    positive_data = df[reward_components].clip(lower=0)
    ax.stackplot(df['current_time'], [positive_data[col] for col in reward_components], 
                labels=reward_components, alpha=0.7)
    
    # Stack negative rewards
    negative_data = df[negative_components].clip(upper=0)
    ax.stackplot(df['current_time'], [negative_data[col] for col in negative_components], 
                labels=negative_components, alpha=0.7)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Components (Stacked)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 2. Action Type Timeline
    ax = axes[0, 1]
    action_types = df['action_type'].values
    action_labels = ['Wait', 'Launch', 'Heat']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, action_type in enumerate(range(3)):
        action_times = df['current_time'][action_types == action_type]
        if len(action_times) > 0:
            ax.scatter(action_times, [i] * len(action_times), 
                      c=colors[i], s=50, alpha=0.7, label=action_labels[i])
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Action Type')
    ax.set_title('Action Timeline')
    ax.set_yticks(range(3))
    ax.set_yticklabels(action_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Oven Status Timeline (for all ovens)
    ax = axes[1, 0]
    num_ovens = len([col for col in df.columns if col.startswith('oven_') and col.endswith('_busy')])
    
    for oven_id in range(num_ovens):
        busy_times = df['current_time'][df[f'oven_{oven_id}_busy'] > 0.5]
        heating_times = df['current_time'][df[f'oven_{oven_id}_heating'] > 0.5]
        cooling_times = df['current_time'][df[f'oven_{oven_id}_cooling'] > 0.5]
        
        if len(busy_times) > 0:
            ax.scatter(busy_times, [oven_id] * len(busy_times), 
                      c='green', s=30, alpha=0.7, label=f'Oven {oven_id} Busy' if oven_id == 0 else "")
        if len(heating_times) > 0:
            ax.scatter(heating_times, [oven_id + 0.3] * len(heating_times), 
                      c='orange', s=30, alpha=0.7, label=f'Oven {oven_id} Heating' if oven_id == 0 else "")
        if len(cooling_times) > 0:
            ax.scatter(cooling_times, [oven_id - 0.3] * len(cooling_times), 
                      c='red', s=30, alpha=0.7, label=f'Oven {oven_id} Cooling' if oven_id == 0 else "")
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Oven ID')
    ax.set_title('Oven Status Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Queue and Completion Rate
    ax = axes[1, 1]
    ax_twin = ax.twinx()
    
    ax.plot(df['current_time'], df['queue_length'], 'b-', label='Queue Length', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Queue Length', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # Calculate completion rate (jobs per minute)
    completion_rate = df['completed_jobs'].diff() / df['current_time'].diff()
    ax_twin.plot(df['current_time'], completion_rate, 'r-', label='Completion Rate', linewidth=2)
    ax_twin.set_ylabel('Completion Rate (jobs/min)', color='r')
    ax_twin.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('Queue Length and Completion Rate')
    ax.grid(True, alpha=0.3)
    
    # 5. Energy Efficiency
    ax = axes[2, 0]
    # Calculate energy efficiency (completed jobs per energy cost)
    energy_efficiency = df['completed_jobs'] / (df['total_energy_cost'].abs() + 1e-6)
    ax.plot(df['current_time'], energy_efficiency, 'purple', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Energy Efficiency (jobs/energy)')
    ax.set_title('Energy Efficiency Over Time')
    ax.grid(True, alpha=0.3)
    
    # 6. Performance Metrics
    ax = axes[2, 1]
    # Calculate on-time delivery rate
    on_time_rate = (df['completed_jobs'] - df['late_jobs_count']) / (df['completed_jobs'] + 1e-6)
    ax.plot(df['current_time'], on_time_rate, 'g-', label='On-time Rate', linewidth=2)
    
    # Calculate throughput (jobs per hour)
    throughput = df['completed_jobs'] / (df['current_time'] / 60 + 1e-6)
    ax_twin2 = ax.twinx()
    ax_twin2.plot(df['current_time'], throughput, 'orange', label='Throughput', linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('On-time Rate', color='g')
    ax_twin2.set_ylabel('Throughput (jobs/hour)', color='orange')
    ax.tick_params(axis='y', labelcolor='g')
    ax_twin2.tick_params(axis='y', labelcolor='orange')
    ax.set_title('Performance Metrics')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        detailed_save_path = save_path.replace('.png', '_detailed.png')
        plt.savefig(detailed_save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plots saved to: {detailed_save_path}")
    
    plt.show()


def main():
    """Main function to run episode analysis"""
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
    env = DynamicOvenBatchingEnv(**env_config)

    # Create agent (you can change this to any available agent)
    agent = RandomAgent(env.action_space)
    
    print("Running episode with tracking...")
    tracker = run_episode_with_tracking(env, agent, max_steps=2000)
    
    print(f"Episode completed! Total steps: {tracker.episode_length}")
    
    # Create plots
    print("Creating comprehensive plots...")
    create_comprehensive_plots(tracker, save_path='results/episode_analysis.png')
    
    print("Creating detailed plots...")
    create_detailed_plots(tracker, save_path='results/episode_analysis.png')
    
    # Save data to CSV
    df = tracker.get_dataframe()
    csv_path = 'results/episode_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Episode data saved to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EPISODE SUMMARY")
    print("="*50)
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


if __name__ == "__main__":
    main()
