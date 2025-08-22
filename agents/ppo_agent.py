"""
Basic PPO Agent for Paint Booth Environment
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch


class PPOAgent:
    """Simple PPO agent wrapper for paint booth environment"""
    
    def __init__(self, env, learning_rate=3e-4, n_steps=2048, batch_size=64, 
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2):
        """
        Initialize PPO agent
        
        Args:
            env: Paint booth environment
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to collect per rollout
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
        """
        self.env = env
        
        # Create vectorized environment (required by stable-baselines3)
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Initialize PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=1,
            device="auto"  # Use GPU if available
        )
    
    def train(self, total_timesteps):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, observation, deterministic=True):
        """Make a prediction"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path):
        """Save the trained model"""
        self.model.save(path)
    
    def load(self, path):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.vec_env)


class TrainingCallback(BaseCallback):
    """Simple callback to log training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    if len(self.episode_rewards) % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        mean_length = np.mean(self.episode_lengths[-10:])
                        print(f"Episode {len(self.episode_rewards)}: "
                              f"Mean Reward: {mean_reward:.2f}, "
                              f"Mean Length: {mean_length:.1f}")
        return True
