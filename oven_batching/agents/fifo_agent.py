"""
FIFO (First In, First Out) Agent for Dynamic Oven Batching Environment

A simple heuristic agent that processes jobs in the order they arrived (FIFO logic).
"""

import numpy as np
from typing import Tuple, Optional
from environment.core import OvenStatus


class FIFOAgent:
    """Simple FIFO agent that processes jobs in arrival order"""
    
    def __init__(self, env):
        """
        Initialize FIFO agent
        
        Args:
            env: The environment to interact with
        """
        self.env = env
    
    def predict(self, state=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action based on FIFO logic
        
        Args:
            state: Agent state (not used for FIFO agent)
            
        Returns:
            Tuple of (action, state)
        """
        # Get valid actions from environment
        action_mask = self.env.get_action_mask()
        
        # Get current queue length
        queue_length = len(self.env.unwrapped.queue)
        
        # If no jobs in queue, wait
        if queue_length == 0:
            action_type = 0  # Wait
            oven_id = 0  # Oven ID doesn't matter for wait action
            num_panels = 0
        else:
            # Check if any ovens need heating first
            cold_ovens = []
            ready_ovens = []
            
            for i, oven in enumerate(self.env.unwrapped.ovens):
                if oven.status == OvenStatus.IDLE:
                    if oven.temperature < 1.0:
                        cold_ovens.append(i)
                    else:
                        ready_ovens.append(i)
            
            # If there are cold ovens, heat the first one
            if cold_ovens:
                action_type = 2  # Heat
                oven_id = cold_ovens[0]
                num_panels = 0
            elif ready_ovens:
                # Launch action with FIFO logic
                action_type = 1  # Launch
                
                # Choose the first ready oven
                oven_id = ready_ovens[0]
                
                # Determine how many panels to process
                # Use maximum possible panels up to oven capacity and queue length
                max_panels = min(queue_length, self.env.unwrapped.oven_capacity)
                
                # Find valid panel count in action mask
                valid_panels = np.where(action_mask[2])[0]
                valid_panels = valid_panels[valid_panels <= max_panels]
                
                if len(valid_panels) > 0:
                    # Choose the maximum valid panel count (process as many as possible)
                    num_panels = valid_panels[-1]
                else:
                    # Fallback to wait if no valid panel counts
                    action_type = 0
                    oven_id = 0
                    num_panels = 0
            else:
                # No ovens available, wait
                action_type = 0  # Wait
                oven_id = 0
                num_panels = 0
        
        action = np.array([action_type, oven_id, num_panels])
        
        return action, state
    
    def reset(self):
        """Reset agent state (no state for FIFO agent)"""
        pass
