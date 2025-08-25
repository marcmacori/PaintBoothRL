"""
FIFO (First In, First Out) Heuristic Agent for Dynamic Oven Batching Environment
"""

import numpy as np
from typing import Tuple, Optional


class FIFOAgent:
    """FIFO heuristic agent that processes jobs in arrival order when possible"""
    
    def __init__(self, env):
        """
        Initialize FIFO agent
        
        Args:
            env: The environment to interact with
        """
        self.env = env
    
    def predict(self, observation, state=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action based on current observation using FIFO strategy
        
        Args:
            observation: Current environment observation
            state: Agent state (not used for FIFO agent)            
        Returns:
            Tuple of (action, state)
        """
        # Get valid actions from environment
        action_mask = self.env.get_action_mask()
        
        # Extract queue length from observation (index 1)
        queue_length = int(observation[1])
        
        # Find ready ovens (ovens that can start a batch)
        valid_ovens = np.where(action_mask[1])[0]
        ready_ovens = []
        
        for oven_id in valid_ovens:
            oven = self.env.unwrapped.ovens[oven_id]
            if oven.is_ready_to_start():
                ready_ovens.append(oven_id)
        
        # Strategy 1: If we have jobs and ready ovens, launch a batch
        if queue_length > 0 and len(ready_ovens) > 0:
            # Choose the first ready oven
            oven_id = ready_ovens[0]
            
            # Determine how many panels to process
            # Try to fill the oven, but don't exceed queue length
            oven_capacity = self.env.unwrapped.ovens[oven_id].capacity
            num_panels = min(queue_length, oven_capacity)
            
            # Check if this panel count is valid
            valid_panels = np.where(action_mask[2])[0]
            if num_panels in valid_panels:
                return np.array([1, oven_id, num_panels]), state
        
        # Strategy 2: If we have cold ovens and jobs in queue, heat an oven
        if queue_length > 0:
            for oven_id in valid_ovens:
                oven = self.env.unwrapped.ovens[oven_id]
                if oven.status.value == 0 and oven.temperature < 1.0:  # IDLE and cold
                    return np.array([2, oven_id, 0]), state
        
        # Strategy 3: If no immediate actions possible, wait
        return np.array([0, 0, 0]), state
    
    def reset(self):
        """Reset agent state (no state for FIFO agent)"""
        pass
