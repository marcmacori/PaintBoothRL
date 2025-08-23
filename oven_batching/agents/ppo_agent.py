"""
Basic PPO Agent for Dynamic Oven Batching Environment

A simple reinforcement learning agent using PPO from stable-baselines3.
"""

import numpy as np
from typing import Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os


class BasicPPOAgent:
    """Basic PPO agent using stable-baselines3"""
    
    def __init__(self, env, learning_rate: float = 3e-4, seed: Optional[int] = None):
        """
        Initialize basic PPO agent
        
        Args:
            env: The environment to interact with
            learning_rate: Learning rate for the optimizer
            seed: Random seed
        """
        self.env = env
        self.seed = seed
        self.is_trained = False
        
        # Create wrapped environment for training
        def make_env():
            return Monitor(env)
        
        self.training_env = DummyVecEnv([make_env])
        
        # Initialize PPO model with better settings for this environment
        self.model = PPO(
            "MlpPolicy",
            self.training_env,
            learning_rate=learning_rate,
            n_steps=2048,  # Increased for better exploration
            batch_size=64,
            n_epochs=10,   # More epochs for better learning
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed
        )
    
    def train(self, total_timesteps: int = 50000, log_dir: str = "./ppo_logs"):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Number of timesteps to train for
            log_dir: Directory to save logs
        """
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # Save the model
        self.model.save(f"{log_dir}/ppo_model")
        self.is_trained = True
        print("Training completed!")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = PPO.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")
    
    def predict(self, observation, state=None, deterministic=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action based on current observation
        
        Args:
            observation: Current environment observation
            state: Agent state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get action mask
        action_mask = self.env.get_action_mask()
        
        # Extract information from observation
        queue_length = int(observation[1])  # Queue length is at index 1
        current_time = self.env.unwrapped.current_time
        
        # Smart action selection based on environment state
        if queue_length > 0:
            # We have jobs to process
            ready_ovens = []
            cold_ovens = []
            
            for i, oven in enumerate(self.env.unwrapped.ovens):
                if oven.is_ready_to_start(current_time):
                    ready_ovens.append(i)
                elif oven.status.value == 0 and oven.temperature < 0.99:  # IDLE and cold
                    cold_ovens.append(i)
            
            if ready_ovens:
                # Launch action with ready ovens
                oven_id = ready_ovens[0]
                num_panels = min(queue_length, self.env.unwrapped.ovens[oven_id].capacity)
                # Ensure this panel count is valid
                if action_mask[2][num_panels]:
                    return np.array([1, oven_id, num_panels]), state
                else:
                    # Find the maximum valid panel count
                    valid_panels = np.where(action_mask[2])[0]
                    if len(valid_panels) > 0:
                        num_panels = valid_panels[-1]
                        return np.array([1, oven_id, num_panels]), state
            
            elif cold_ovens:
                # Heat action for cold ovens
                oven_id = cold_ovens[0]
                if action_mask[1][oven_id]:
                    return np.array([2, oven_id, 0]), state
        
        # Default to wait action if no other actions are appropriate
        return np.array([0, 0, 0]), state
    
    def reset(self):
        """Reset agent state"""
        return None
