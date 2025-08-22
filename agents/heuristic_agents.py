import numpy as np
import random
from typing import Tuple, List, Dict, Any
from collections import defaultdict


class RandomAgent:
    """
    A basic random agent for the Paint Booth Environment.
    """
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.name = "Random Agent"
        self.episode_rewards = []
        self.episode_stats = []
    
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        if info:
            pending_orders = info.get('pending_orders', 0)
            buffer_orders = info.get('buffer_orders', 0)
        else:
            pending_orders = 10
            buffer_orders = 5
        
        # Random order selection
        order_action = random.randint(0, min(pending_orders, 50)) if pending_orders > 0 else 0
        
        # Random buffer action
        buffer_action = random.randint(0, min(buffer_orders, 20)) if buffer_orders > 0 else 0
        
        return order_action, buffer_action
    
    def reset(self):
        pass
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        if done:
            self.episode_rewards.append(sum(self.episode_rewards) if hasattr(self, '_current_episode_reward') else 0)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0)
            })
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats
        }


class GreedyAgent:
    """
    A greedy agent that always tries to process the first order and buffer item.
    """
    
    def __init__(self):
        self.name = "Greedy Agent"
        self.episode_rewards = []
        self.episode_stats = []
    
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        if info:
            pending_orders = info.get('pending_orders', 0)
            buffer_orders = info.get('buffer_orders', 0)
        else:
            pending_orders = 0
            buffer_orders = 0
        
        # Greedy: always try to process first order and first buffer item
        order_action = 1 if pending_orders > 0 else 0
        buffer_action = 1 if buffer_orders > 0 else 0
        
        return order_action, buffer_action
    
    def reset(self):
        pass
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        if done:
            self.episode_rewards.append(sum(self.episode_rewards) if hasattr(self, '_current_episode_reward') else 0)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0)
            })
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats
        }


class DoNothingAgent:
    """
    An agent that takes no actions - useful for testing environment dynamics.
    """
    
    def __init__(self):
        self.name = "Do Nothing Agent"
        self.episode_rewards = []
        self.episode_stats = []
    
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        return 0, 0
    
    def reset(self):
        pass
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        if done:
            self.episode_rewards.append(sum(self.episode_rewards) if hasattr(self, '_current_episode_reward') else 0)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0)
            })
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats
        }





# Agent factory function for easy instantiation
def create_agent(agent_type: str, **kwargs):
    """
    Factory function to create agents by name.
    
    Args:
        agent_type: Name of the agent type to create
        **kwargs: Additional arguments for agent initialization
    
    Returns:
        Agent instance
    """
    agents = {
        'random': RandomAgent,
        'greedy': GreedyAgent,
        'do_nothing': DoNothingAgent
    }
    
    if agent_type.lower() not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    return agents[agent_type.lower()](**kwargs)


# List of all available agents
ALL_AGENTS = [
    'random',
    'greedy', 
    'do_nothing'
]
