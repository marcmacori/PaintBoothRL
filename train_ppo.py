"""
Basic PPO Training Script for Paint Booth Environment
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.paint_booth_env import PaintBoothEnv
from agents.ppo_agent import PPOAgent, TrainingCallback


def main():
    """Main training function"""
    print("Initializing Paint Booth Environment...")
    
    # Create environment
    env = PaintBoothEnv()
    
    # Create PPO agent with basic hyperparameters
    print("Creating PPO Agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=1024,      # Smaller for faster training
        batch_size=64,
        n_epochs=5,        # Fewer epochs for faster training
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )
    
    # Create callback for monitoring
    callback = TrainingCallback(verbose=1)
    
    # Train the agent
    print("Starting training...")
    total_timesteps = 50000  # Basic training run
    
    try:
        agent.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save the trained model
        model_path = "models/ppo_paint_booth"
        os.makedirs("models", exist_ok=True)
        agent.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training progress
        if callback.episode_rewards:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(callback.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(1, 2, 2)
            plt.plot(callback.episode_lengths)
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            print("Training plots saved to training_progress.png")
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_agent(agent, env, num_episodes=5)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save("models/ppo_paint_booth_interrupted")
        print("Model saved before exit")


def test_agent(agent, env, num_episodes=5):
    """Test the trained agent"""
    rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = done or truncated
        
        rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    print(f"\nTest Results:")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")


if __name__ == "__main__":
    main()
