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
    
    def __init__(self, env, learning_rate: float = 3e-4, seed: Optional[int] = None, hyperparameters: Optional[dict] = None):
        """
        Initialize basic PPO agent
        
        Args:
            env: The environment to interact with
            learning_rate: Learning rate for the optimizer
            seed: Random seed
            hyperparameters: Dictionary of PPO hyperparameters
        """
        self.env = env
        self.seed = seed
        self.is_trained = False
        
        # Create wrapped environment for training
        def make_env():
            return Monitor(env)
        
        self.training_env = DummyVecEnv([make_env])
        
        # Default hyperparameters
        default_hyperparams = {
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 0
        }
        
        # Update with provided hyperparameters
        if hyperparameters:
            default_hyperparams.update(hyperparameters)
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.training_env,
            learning_rate=learning_rate,
            seed=seed,
            **default_hyperparams
        )
    
    def train(self, total_timesteps: int = 50000, log_dir: str = "./saved_models"):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Number of timesteps to train for
            log_dir: Directory to save logs
        """
        os.makedirs(log_dir, exist_ok=True)
        
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
        Predict action based on current observation using the trained PPO model
        
        Args:
            observation: Current environment observation
            state: Agent state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get action mask to ensure valid actions
        action_mask = self.env.get_action_mask()
        
        # Use the trained PPO model to predict action
        action, state = self.model.predict(observation, state, deterministic)
        
        # Ensure the predicted action is valid according to action mask
        action_type, oven_id, num_panels = action
        
        # Validate action type (0=wait, 1=launch, 2=heat)
        if action_type not in [0, 1, 2]:
            action_type = 0  # Default to wait if invalid
        
        # Validate oven_id
        if oven_id < 0 or oven_id >= len(self.env.unwrapped.ovens):
            oven_id = 0  # Default to first oven if invalid
        
        # Validate num_panels based on action type
        if action_type == 1:  # Launch action
            # For launch, num_panels should be valid according to action_mask[2]
            if not action_mask[2][num_panels]:
                # Find a valid panel count
                valid_panels = np.where(action_mask[2])[0]
                if len(valid_panels) > 0:
                    num_panels = valid_panels[0]  # Use first valid option
                else:
                    action_type = 0  # No valid launch possible, default to wait
                    num_panels = 0
        elif action_type == 2:  # Heat action
            # For heat, check if the oven can be heated
            if not action_mask[1][oven_id]:
                action_type = 0  # Cannot heat this oven, default to wait
                num_panels = 0
            else:
                num_panels = 0  # Heat actions always have 0 panels
        else:  # Wait action (action_type == 0)
            num_panels = 0
        
        return np.array([action_type, oven_id, num_panels]), state
    
    def reset(self):
        """Reset agent state"""
        return None
