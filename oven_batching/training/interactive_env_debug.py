"""
Interactive Debug Script for Oven Batching Environment

This script allows you to manually control the environment step by step
and see detailed information about what's happening at each time step.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from environment.core import DynamicOvenBatchingEnv


def print_action_help():
    """Print help for action format"""
    print("\n" + "="*60)
    print("ACTION FORMAT: [action_type] [oven_id] [num_panels]")
    print("="*60)
    print("Action Types:")
    print("  0 = WAIT (do nothing)")
    print("  1 = LAUNCH (start a batch)")
    print("  2 = HEAT (heat an oven)")
    print("\nExamples:")
    print("  '0 0 0' = Wait")
    print("  '1 0 5' = Launch batch with 5 panels in oven 0")
    print("  '2 1 0' = Heat oven 1")
    print("  'q' = Quit")
    print("  'h' = Show this help")
    print("  's' = Show current state")
    print("  'a' = Show available actions")
    print("\nNote: Invalid actions are allowed for debugging!")
    print("Invalid actions will result in action_validation_reward = -0.1")
    print("="*60)


def print_state(env, step_num, cumulative_rewards=None):
    """Print detailed current state"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num} - Time: {env.current_time:.1f}/{env.horizon:.1f}")
    print(f"{'='*80}")
    
    # Queue information
    print(f"\nQUEUE ({len(env.queue)} jobs):")
    if env.queue:
        for i, job in enumerate(env.queue[:10]):  # Show first 10 jobs
            time_to_due = job.due_date - env.current_time
            lateness = max(0, env.current_time - job.due_date)
            status = "LATE" if lateness > 0 else "ON_TIME"
            print(f"  Job {job.id}: Arrived {job.arrival_time:.1f}, Due {job.due_date:.1f}, "
                  f"Time to due: {time_to_due:.1f}, {status}")
        if len(env.queue) > 10:
            print(f"  ... and {len(env.queue) - 10} more jobs")
    else:
        print("  Empty")
    
    # Oven information
    print(f"\nOVENS:")
    for i, oven in enumerate(env.ovens):
        status_str = {
            0: "IDLE",
            1: "BUSY", 
            2: "HEATING"
        }[oven.status.value]
        
        temp_str = f"Temp: {oven.temperature:.2f}"
        if oven.status.value == 1:  # BUSY
            time_left = oven.completion_time - env.current_time
            jobs_in_batch = len(oven.current_batch_jobs)
            print(f"  Oven {i}: {status_str} | {temp_str} | "
                  f"Batch completes in {time_left:.1f} min | {jobs_in_batch} jobs")
        elif oven.status.value == 2:  # HEATING
            time_to_ready = oven.heating_start_time + oven.heating_time - env.current_time
            print(f"  Oven {i}: {status_str} | {temp_str} | "
                  f"Ready in {time_to_ready:.1f} min")
        else:  # IDLE
            ready_str = "READY" if oven.temperature >= 1.0 else "COLD"
            print(f"  Oven {i}: {status_str} | {temp_str} | {ready_str}")
    
    # Statistics
    print(f"\nSTATISTICS:")
    print(f"  Completed jobs: {len(env.completed_jobs)}")
    print(f"  Late jobs: {env.late_jobs_count}")
    print(f"  Total energy cost: {env.total_energy_cost:.2f}")
    print(f"  Total lateness penalty: {env.total_lateness_penalty:.2f}")
    print(f"  Total reward: {env.total_reward:.2f}")
    
    # Reward breakdown if available
    if cumulative_rewards:
        print(f"\nREWARD BREAKDOWN:")
        print(f"  Action Validation: {cumulative_rewards['action_validation']:.2f}")
        print(f"  Launch: {cumulative_rewards['launch']:.2f}")
        print(f"  Heating: {cumulative_rewards['heating']:.2f}")
        print(f"  Wait: {cumulative_rewards['wait']:.2f}")
        print(f"  Energy Cost Penalty: {cumulative_rewards['energy_cost']:.2f}")
        print(f"  Lateness Penalty: {cumulative_rewards['lateness']:.2f}")
        print(f"  Completion: {cumulative_rewards['completion']:.2f}")


def print_available_actions(env):
    """Print available actions based on current state"""
    print(f"\n{'='*60}")
    print("AVAILABLE ACTIONS:")
    print(f"{'='*60}")
    
    queue_length = len(env.queue)
    
    # Check each oven
    for i, oven in enumerate(env.ovens):
        print(f"\nOven {i}:")
        
        if oven.is_ready_to_start():
            print(f"  READY for LAUNCH")
            max_panels = min(queue_length, env.oven_capacity)
            if max_panels > 0:
                print(f"  Can launch: 1 to {max_panels} panels")
                for panels in range(1, max_panels + 1):
                    print(f"    Action: '1 {i} {panels}'")
            else:
                print(f"  No jobs in queue to launch")
        elif oven.status.value == 0 and oven.temperature < 1.0:  # IDLE and cold
            print(f"  Can HEAT (currently cold)")
            print(f"    Action: '2 {i} 0'")
        elif oven.status.value == 1:  # BUSY
            time_left = oven.completion_time - env.current_time
            print(f"  BUSY - completes in {time_left:.1f} min")
        elif oven.status.value == 2:  # HEATING
            time_to_ready = oven.heating_start_time + oven.heating_time - env.current_time
            print(f"  HEATING - ready in {time_to_ready:.1f} min")
    
    print(f"\nAlways available:")
    print(f"  WAIT: '0 0 0'")


def get_valid_action(env):
    """Get user action input (allows invalid actions for debugging)"""
    while True:
        try:
            action_input = input("\nEnter action (or 'h' for help, 'q' to quit): ").strip().lower()
            
            if action_input == 'q':
                return None
            elif action_input == 'h':
                print_action_help()
                continue
            elif action_input == 's':
                return 'show_state'
            elif action_input == 'a':
                return 'show_actions'
            
            # Parse action
            parts = action_input.split()
            if len(parts) != 3:
                print("Error: Action must have 3 parts: [action_type] [oven_id] [num_panels]")
                continue
            
            action_type = int(parts[0])
            oven_id = int(parts[1])
            num_panels = int(parts[2])
            
            # Check if action is valid but don't prevent it
            is_valid = env._is_action_valid(action_type, oven_id, num_panels)
            if not is_valid:
                print("Warning: This action is invalid for the current state, but proceeding anyway...")
                print("Expected behavior: action_validation_reward should be -0.1")
            
            return np.array([action_type, oven_id, num_panels])
            
        except ValueError:
            print("Error: Please enter valid integers")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def main():
    """Main interactive debug function"""
    print("Interactive Oven Batching Environment Debug")
    print("="*60)
    
    # Create environment with same config as training
    env_config = {
        'num_ovens': 1,
        'oven_capacity': 9,
        'batch_time': 10.0,
        'batch_energy_cost': 9.0,
        'heating_time': 5.0,
        'cooling_rate': 0.1,
        'horizon': 1440.0,
        'arrival_rate': 0.8,
        'due_date_offset_mean': 60.0,
        'due_date_offset_std': 20.0,
        'energy_alpha': 10.0,  # Energy penalty multiplier
        'lateness_beta': 1.0,  # Lateness penalty multiplier
        'use_dynamic_arrivals': True,
        'time_step': 1.0
    }
    
    env = DynamicOvenBatchingEnv(**env_config)
    print(f"Environment created with {env.num_ovens} ovens")
    print(f"Horizon: {env.horizon} minutes ({env.horizon/60:.1f} hours)")
    print(f"Arrival rate: {env.arrival_rate} jobs/minute")
    print(f"Due date offset: {env.due_date_offset_mean} ± {env.due_date_offset_std} minutes")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nEnvironment reset. Initial queue length: {info['queue_length']}")
    
    print_action_help()
    
    step_num = 0
    action_history = []
    
    # Track cumulative rewards
    cumulative_rewards = {
        'action_validation': 0.0,
        'launch': 0.0,
        'heating': 0.0,
        'wait': 0.0,
        'energy_cost': 0.0,
        'lateness': 0.0,
        'completion': 0.0
    }
    
    while step_num < env.horizon:
        print_state(env, step_num, cumulative_rewards)
        
        action = get_valid_action(env)
        if action is None:
            print("Exiting...")
            break
        elif isinstance(action, str):
            if action == 'show_state':
                continue
            elif action == 'show_actions':
                print_available_actions(env)
                continue
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        # Record action
        action_str = f"Step {step_num}: {action[0]} {action[1]} {action[2]}"
        action_history.append(action_str)
        
        print(f"\nAction taken: {action_str}")
        print(f"Total Reward: {reward:.4f}")
        print(f"\nReward Breakdown:")
        print(f"  Action Validation: {info['action_validation_reward']:.4f}")
        print(f"  Launch: {info['launch_reward']:.4f}")
        print(f"  Heating: {info['heating_reward']:.4f}")
        print(f"  Wait: {info['wait_reward']:.4f}")
        print(f"  Energy Cost Penalty: {info['energy_cost_penalty']:.4f}")
        print(f"  Lateness Penalty: {info['lateness_penalty']:.4f}")
        print(f"  Completion: {info['completion_reward']:.4f}")
        print(f"\nState Update:")
        print(f"  Queue length: {info['queue_length']}")
        print(f"  Completed jobs: {info['completed_jobs']}")
        print(f"  Late jobs: {info['late_jobs_count']}")
        print(f"  Total lateness penalty: {info['total_lateness_penalty']:.2f}")
        
        # Update cumulative rewards
        cumulative_rewards['action_validation'] += info['action_validation_reward']
        cumulative_rewards['launch'] += info['launch_reward']
        cumulative_rewards['heating'] += info['heating_reward']
        cumulative_rewards['wait'] += info['wait_reward']
        cumulative_rewards['energy_cost'] += info['energy_cost_penalty']
        cumulative_rewards['lateness'] += info['lateness_penalty']
        cumulative_rewards['completion'] += info['completion_reward']
        
        step_num += 1
        
        if done or truncated:
            print(f"\nEpisode ended at step {step_num}")
            break
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total steps: {step_num}")
    print(f"Final queue length: {len(env.queue)}")
    print(f"Total completed jobs: {len(env.completed_jobs)}")
    print(f"Total late jobs: {env.late_jobs_count}")
    print(f"Total energy cost: {env.total_energy_cost:.2f}")
    print(f"Total lateness penalty: {env.total_lateness_penalty:.2f}")
    print(f"Total reward: {env.total_reward:.2f}")
    
    # Show cumulative reward breakdown
    print(f"\nCumulative Reward Breakdown:")
    print(f"  Action Validation: {cumulative_rewards['action_validation']:.2f}")
    print(f"  Launch: {cumulative_rewards['launch']:.2f}")
    print(f"  Heating: {cumulative_rewards['heating']:.2f}")
    print(f"  Wait: {cumulative_rewards['wait']:.2f}")
    print(f"  Energy Cost Penalty: {cumulative_rewards['energy_cost']:.2f}")
    print(f"  Lateness Penalty: {cumulative_rewards['lateness']:.2f}")
    print(f"  Completion: {cumulative_rewards['completion']:.2f}")
    print(f"  Total: {sum(cumulative_rewards.values()):.2f}")
    
    # Show action history
    print(f"\nAction History (last 20 actions):")
    for action in action_history[-20:]:
        print(f"  {action}")


if __name__ == "__main__":
    main()
