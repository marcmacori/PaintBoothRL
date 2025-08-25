"""
Random Agent for Dynamic Oven Batching Environment
"""

import numpy as np
from typing import Tuple, Optional


class RandomAgent:
    """Simple random agent for comparison and baseline"""
    
    def __init__(self, env):
        """
        Initialize random agent
        
        Args:
            env: The environment to interact with
        """
        self.env = env
    
    def predict(self, state=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action based on current observation
        
        Args:
            observation: Current environment observation
            state: Agent state (not used for random agent)
            deterministic: Whether to use deterministic policy (not used for random agent)
            
        Returns:
            Tuple of (action, state)
        """
        # Get valid actions from environment
        action_mask = self.env.get_action_mask()
        
        # Sample valid action for each dimension
        action_type = np.random.choice(np.where(action_mask[0])[0])
        
        # Handle case where no ovens are available
        valid_ovens = np.where(action_mask[1])[0]
        if len(valid_ovens) == 0:
            # If no ovens available, force wait action
            action_type = 0
            oven_id = 0  # Oven ID doesn't matter for wait action
        else:
            oven_id = np.random.choice(valid_ovens)
        
        # Handle case where no panel counts are valid (e.g., no jobs in queue)
        valid_panels = np.where(action_mask[2])[0]
        if len(valid_panels) == 0:
            # If no valid panel counts, default to 0 (which is ignored for wait/heat actions)
            num_panels = 0
        else:
            num_panels = np.random.choice(valid_panels)
        
        action = np.array([action_type, oven_id, num_panels])
        
        return action, state
    
    def reset(self):
        """Reset agent state (no state for random agent)"""
        pass
