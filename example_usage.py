#!/usr/bin/env python3
"""
Example usage of the Paint Booth Environment

This script demonstrates how to use the PaintBoothEnv for training
reinforcement learning agents or running simulations.
"""

import numpy as np
from env import PaintBoothEnv

def random_agent_example():
    """Example using a random agent"""
    print("=== Random Agent Example ===")
    
    # Create environment
    env = PaintBoothEnv(shift_duration=8.0, time_step=1.0)  # 8-hour shift, 1-minute steps
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 100:  # Limit steps for demo
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render every 30 steps (30 minutes)
        if step_count % 30 == 0:
            env.render()
            print(f"Step {step_count}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            print(f"Info: {info}")
            print("-" * 50)
    
    print(f"\nFinal Results:")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Info: {info}")
    
    env.close()

def simple_heuristic_agent():
    """Example using a simple heuristic agent"""
    print("\n=== Simple Heuristic Agent Example ===")
    
    env = PaintBoothEnv(shift_duration=2.0, time_step=1.0)  # 2-hour shift for quick demo
    obs = env.reset()
    
    total_reward = 0
    done = False
    step_count = 0
    
    while not done:
        # Simple heuristic: always try to process the first available order
        if len(env.pending_orders) > 0:
            action = [1, 0]  # Process first order, use default equipment choice
        else:
            action = [0, 0]  # No action
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render every 15 steps
        if step_count % 15 == 0:
            env.render()
            print(f"Step {step_count}, Action: {action}, Reward: {reward:.2f}")
            print(f"Info: {info}")
            print("-" * 50)
    
    print(f"\nHeuristic Agent Results:")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Info: {info}")
    
    # Calculate some performance metrics
    if info['completed_panels'] + info['defective_panels'] > 0:
        quality_rate = info['completed_panels'] / (info['completed_panels'] + info['defective_panels'])
        print(f"Quality Rate: {quality_rate:.2%}")
        print(f"Throughput: {info['completed_panels']} panels in {step_count} minutes")
        print(f"Average Quality Score: {info['quality_score']:.2f}")
    
    env.close()

def analyze_order_patterns():
    """Analyze order generation patterns"""
    print("\n=== Order Generation Pattern Analysis ===")
    
    env = PaintBoothEnv(shift_duration=8.0, time_step=5.0)  # 5-minute steps for faster analysis
    obs = env.reset()
    
    order_times = []
    order_counts_per_hour = [0] * 8
    paint_type_counts = {'water': 0, 'solvent': 0}
    panel_count_distribution = {1: 0, 2: 0, 3: 0}
    
    done = False
    step_count = 0
    
    while not done:
        orders_before = env.total_orders_generated
        
        # Take no action, just observe order generation
        obs, reward, done, info = env.step([0, 0])
        step_count += 1
        
        orders_after = env.total_orders_generated
        new_orders = orders_after - orders_before
        
        if new_orders > 0:
            current_hour = int(env.current_time // 60)
            if current_hour < 8:
                order_counts_per_hour[current_hour] += new_orders
            
            # Analyze current pending orders
            for order in list(env.pending_orders):
                paint_type_counts[order.paint_type.value] += 1
                panel_count_distribution[len(order.panels)] += 1
                order_times.append(env.current_time)
    
    print(f"Total orders generated: {env.total_orders_generated}")
    print(f"Orders per hour: {order_counts_per_hour}")
    print(f"Paint type distribution: {paint_type_counts}")
    print(f"Panel count distribution: {panel_count_distribution}")
    
    # Identify peak times
    peak_hours = [i for i, count in enumerate(order_counts_per_hour) if count > np.mean(order_counts_per_hour)]
    print(f"Peak hours (above average): {peak_hours}")
    
    env.close()

def test_equipment_utilization():
    """Test equipment utilization and bottlenecks"""
    print("\n=== Equipment Utilization Test ===")
    
    env = PaintBoothEnv(shift_duration=4.0, time_step=1.0)  # 4-hour test
    obs = env.reset()
    
    equipment_busy_time = {}
    equipment_names = [
        "water_paint_robot", "solvent_paint_robot", "water_flash_off",
        "solvent_flash_off_1", "solvent_flash_off_2", "water_oven",
        "solvent_oven_1", "solvent_oven_2", "varnish_robot"
    ]
    
    for name in equipment_names:
        equipment_busy_time[name] = 0
    
    done = False
    step_count = 0
    
    while not done:
        # Simple strategy: always process first available order
        if len(env.pending_orders) > 0:
            action = [1, 0]
        else:
            action = [0, 0]
        
        # Record equipment busy times before step
        for name in equipment_names:
            equipment = getattr(env, name)
            if equipment.busy_until > env.current_time:
                equipment_busy_time[name] += env.time_step
        
        obs, reward, done, info = env.step(action)
        step_count += 1
    
    print(f"Equipment Utilization (over {step_count} minutes):")
    for name, busy_time in equipment_busy_time.items():
        utilization = (busy_time / step_count) * 100
        print(f"  {name.replace('_', ' ').title()}: {utilization:.1f}%")
    
    print(f"\nFinal Performance:")
    print(f"Completed Panels: {info['completed_panels']}")
    print(f"Defective Panels: {info['defective_panels']}")
    print(f"Pending Orders: {info['pending_orders']}")
    
    env.close()

if __name__ == "__main__":
    print("Paint Booth Environment Demo")
    print("=" * 50)
    
    # Run examples
    random_agent_example()
    simple_heuristic_agent()
    analyze_order_patterns()
    test_equipment_utilization()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Train a reinforcement learning agent (e.g., PPO, DQN)")
    print("2. Implement more sophisticated scheduling heuristics")
    print("3. Analyze bottlenecks and optimize equipment configuration")
    print("4. Experiment with different shift patterns and order volumes")
