"""
Bayesian Optimization for PPO Hyperparameter Tuning with Matern Kernel
"""

#TODO: Add Latin Hypercube Sampling
#TODO: We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import sys
import time
from datetime import datetime
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oven_batching.environment.core import DynamicOvenBatchingEnv
from oven_batching.agents.ppo_agent import BasicPPOAgent

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class PPOHyperparameterOptimizer:
    """Bayesian optimization for PPO hyperparameters using Matern kernel"""
    
    def __init__(self, n_trials=15, n_random_starts=3, eval_episodes=5, training_timesteps=8000):
        self.n_trials = n_trials
        self.n_random_starts = n_random_starts
        self.eval_episodes = eval_episodes
        self.training_timesteps = training_timesteps
        
        # Results storage
        self.results = []
        self.best_score = float('-inf')
        self.best_params = None
        
        # Hyperparameter search space
        self.param_ranges = {
            'learning_rate': (1e-5, 1e-2),
            'n_steps': (512, 4096),
            'batch_size': (32, 128),
            'n_epochs': (3, 20),
            'gamma': (0.9, 0.999),
            'gae_lambda': (0.8, 0.99),
            'clip_range': (0.1, 0.3),
            'ent_coef': (1e-4, 1e-1),
            'vf_coef': (0.1, 1.0),
            'max_grad_norm': (0.1, 2.0)
        }
        
        self.param_names = list(self.param_ranges.keys())
        
        # Initialize Gaussian Process with Matern kernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        self.scaler = StandardScaler()
        
        self.X_observed = []
        self.y_observed = []
    
    def create_environment(self):
        """Create environment instance"""
        return DynamicOvenBatchingEnv(
            num_ovens=2, oven_capacity=9, batch_time=10.0, horizon=1440.0,
            arrival_rate=0.5, use_dynamic_arrivals=True
        )
    
    def sample_random_params(self):
        """Sample random hyperparameters"""
        params = []
        for param_name in self.param_names:
            min_val, max_val = self.param_ranges[param_name]
            
            if param_name in ['learning_rate', 'ent_coef']:
                log_min, log_max = np.log10(min_val), np.log10(max_val)
                val = 10 ** random.uniform(log_min, log_max)
            else:
                val = random.uniform(min_val, max_val)
            
            if param_name in ['n_steps', 'batch_size', 'n_epochs']:
                val = int(round(val))
            
            params.append(val)
        return params
    
    def suggest_next_params(self):
        """Suggest next hyperparameters using GP"""
        if len(self.X_observed) < 2:
            return self.sample_random_params()
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        X_scaled = self.scaler.fit_transform(X)
        
        self.gp.fit(X_scaled, y)
        
        # Generate candidates and predict
        candidates = np.array([self.sample_random_params() for _ in range(50)])
        candidates_scaled = self.scaler.transform(candidates)
        mean_pred, std_pred = self.gp.predict(candidates_scaled, return_std=True)
        
        # UCB acquisition function
        ucb = mean_pred + 0.1 * std_pred
        best_idx = np.argmax(ucb)
        return candidates[best_idx].tolist()
    
    def evaluate_hyperparameters(self, params):
        """Evaluate hyperparameters by training PPO agent"""
        try:
            env = self.create_environment()
            
            # Import the BasicPPOAgent
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'oven_batching'))
            from agents.ppo_agent import BasicPPOAgent
            
            # Create a custom PPO agent with the given hyperparameters
            class CustomPPOAgent(BasicPPOAgent):
                def __init__(self, env, params):
                    self.env = env
                    self.seed = 42
                    self.is_trained = False
                    
                    # Create wrapped environment for training
                    from stable_baselines3.common.vec_env import DummyVecEnv
                    from stable_baselines3.common.monitor import Monitor
                    
                    def make_env():
                        return Monitor(env)
                    
                    self.training_env = DummyVecEnv([make_env])
                    
                    # Initialize PPO model with custom hyperparameters
                    from stable_baselines3 import PPO
                    self.model = PPO(
                        "MlpPolicy",
                        self.training_env,
                        learning_rate=params[0],
                        n_steps=int(params[1]),
                        batch_size=int(params[2]),
                        n_epochs=int(params[3]),
                        gamma=params[4],
                        gae_lambda=params[5],
                        clip_range=params[6],
                        ent_coef=params[7],
                        vf_coef=params[8],
                        max_grad_norm=params[9],
                        verbose=0,
                        seed=self.seed
                    )
            
            # Create and train the custom PPO agent
            ppo_agent = CustomPPOAgent(env, params)
            
            # Train the agent
            os.makedirs("./temp_ppo_logs", exist_ok=True)
            print(f"Training PPO agent for {self.training_timesteps} timesteps...")
            ppo_agent.train(total_timesteps=self.training_timesteps, log_dir="./temp_ppo_logs")
            
            # Evaluate the trained agent
            episode_rewards = []
            for _ in range(self.eval_episodes):
                obs, info = env.reset()
                ppo_agent.reset()
                total_reward = 0.0
                
                while True:
                    action, _ = ppo_agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                
                episode_rewards.append(total_reward)
            
            avg_reward = np.mean(episode_rewards)
            
            # Store results
            result = {
                'timestamp': datetime.now().isoformat(),
                'avg_reward': avg_reward,
                'std_reward': np.std(episode_rewards),
                **dict(zip(self.param_names, params))
            }
            self.results.append(result)
            
            self.X_observed.append(params)
            self.y_observed.append(avg_reward)
            
            if avg_reward > self.best_score:
                self.best_score = avg_reward
                self.best_params = dict(zip(self.param_names, params))
                print(f"New best: {self.best_score:.3f}")
            
            print(f"Trial - Reward: {avg_reward:.3f}, Std: {np.std(episode_rewards):.3f}")
            return avg_reward
            
        except Exception as e:
            print(f"Error: {e}")
            return 0.0
    
    def optimize(self):
        """Run Bayesian optimization"""
        print(f"Starting optimization: {self.n_trials} trials, {self.training_timesteps} timesteps each")
        
        start_time = time.time()
        
        for trial in range(self.n_trials):
            print(f"\nTrial {trial + 1}/{self.n_trials}")
            
            if trial < self.n_random_starts:
                params = self.sample_random_params()
                print("Random sampling...")
            else:
                params = self.suggest_next_params()
                print("Bayesian suggestion...")
            
            self.evaluate_hyperparameters(params)
        
        print(f"\nOptimization completed in {time.time() - start_time:.2f}s")
        print(f"Best score: {self.best_score:.3f}")
        return self.best_params, self.results
    
    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ppo_hyperopt_results_{timestamp}.csv"
        
        os.makedirs("bopt_results", exist_ok=True)
        filepath = os.path.join("bopt_results", filename)
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return filepath
    
    def plot_results(self):
        """Plot optimization convergence"""
        if not self.results:
            print("No results to plot")
            return
        
        rewards = [r['avg_reward'] for r in self.results]
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, 'b-', alpha=0.6, label='Trial Reward')
        plt.plot(np.maximum.accumulate(rewards), 'r-', linewidth=2, label='Best Reward So Far')
        plt.xlabel('Trial')
        plt.ylabel('Average Reward')
        plt.title('Bayesian Optimization Convergence (Matern Kernel)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    """Main function to run hyperparameter optimization"""
    optimizer = PPOHyperparameterOptimizer(
        n_trials=15,
        n_random_starts=3,
        eval_episodes=5,
        training_timesteps=8000
    )
    
    best_params, all_results = optimizer.optimize()
    results_file = optimizer.save_results()
    optimizer.plot_results()
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.best_score:.3f}")


if __name__ == "__main__":
    main()
