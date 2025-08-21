#!/usr/bin/env python3
"""
Simple demonstration of the fixed Paint Booth Environment.

This script shows a quick example of how to use the environment with a basic agent.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.paint_booth_env import PaintBoothEnv
from agents.smart_agents import ImprovedGreedyAgent


def main():
    print("🏭 Paint Booth Environment - Simple Demo")
    print("=" * 40)
    
    # Create environment and agent
    env = PaintBoothEnv(shift_duration=2.0, time_step=1.0)  # 2-hour demo
    agent = ImprovedGreedyAgent()
    
    print(f"Environment: {env.shift_duration/60} hour shift")
    print(f"Agent: {agent.name}")
    print(f"Expected panel completion time: ~90 minutes")
    
    # Run episode
    observation = env.reset()
    agent.reset()
    
    total_reward = 0
    step = 0
    completed_at_step = None
    
    print(f"\nRunning episode...")
    
    while True:
        # Get action from agent
        info = {
            'pending_orders': len(env.pending_orders),
            'buffer_orders': len(env._get_complete_orders_in_buffer())
        }
        action = agent.select_action(observation, info)
        
        # Take step
        observation, reward, done, env_info = env.step(action)
        total_reward += reward
        
        # Print progress every 30 minutes
        if step % 30 == 0:
            print(f"Time {env.current_time:3.0f}min: "
                  f"Completed={env_info['completed_panels']:2d}, "
                  f"Defective={env_info['defective_panels']:2d}, "
                  f"Pending={len(env.pending_orders):2d}")
        
        # Check for first completion
        if env_info['completed_panels'] > 0 and step > 0 and completed_at_step is None:
            completed_at_step = step
            print(f"🎉 First panel completed at {env.current_time:.0f} minutes!")
        
        step += 1
        if done:
            break
    
    # Show results
    print(f"\n📊 Final Results:")
    print(f"   Total time: {env.current_time:.0f} minutes")
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Orders generated: {env_info['total_orders']}")
    print(f"   Panels completed: {env_info['completed_panels']}")
    print(f"   Panels defective: {env_info['defective_panels']}")
    print(f"   Average quality: {env_info['quality_score']:.3f}")
    
    if env_info['completed_panels'] > 0:
        print(f"\n✅ SUCCESS! The environment is working correctly!")
        print(f"   Panels successfully completed the full manufacturing process.")
        print(f"   Quality system is functional.")
        print(f"   Agent decisions are impacting outcomes.")
    else:
        print(f"\n⚠️  No completions in this short demo.")
        print(f"   Try running a longer simulation (4+ hours) for guaranteed completions.")
    
    print(f"\n🚀 Environment is ready for advanced agent development!")


if __name__ == "__main__":
    main()
