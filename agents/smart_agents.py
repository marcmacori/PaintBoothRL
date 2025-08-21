import numpy as np
import random
from typing import Tuple, List, Dict, Any
from collections import defaultdict

class ImprovedGreedyAgent:
    """
    An improved greedy agent that makes smarter decisions based on environment state.
    
    This agent prioritizes:
    1. Processing orders quickly to avoid long wait times
    2. Moving complete orders from buffer to varnish efficiently
    3. Balancing order types to avoid bottlenecks
    """
    
    def __init__(self):
        self.name = "Improved Greedy Agent"
        self.episode_rewards = []
        self.episode_stats = []
        self.decision_history = []
    
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        """
        Select action using improved greedy heuristics.
        
        Args:
            observation: Current environment observation
            info: Additional information from environment
            
        Returns:
            Tuple of (order_selection, buffer_action)
        """
        if not info:
            return 0, 0
        
        pending_orders = info.get('pending_orders', 0)
        buffer_orders = info.get('buffer_orders', 0)
        
        # Always prioritize moving from buffer first (to prevent buffer overflow)
        buffer_action = 1 if buffer_orders > 0 else 0
        
        # For order processing, be more selective
        if pending_orders > 0:
            # Process orders more aggressively when buffer is not full
            if buffer_orders < 3:
                order_action = 1  # Process first order
            elif buffer_orders < 5:
                order_action = 1 if random.random() < 0.7 else 0  # 70% chance
            else:
                order_action = 0  # Wait when buffer is getting full
        else:
            order_action = 0
        
        self.decision_history.append((order_action, buffer_action, pending_orders, buffer_orders))
        return order_action, buffer_action
    
    def reset(self):
        """Reset agent state for new episode"""
        self.decision_history = []
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        """Update agent statistics"""
        if done:
            self.episode_rewards.append(sum(self.episode_rewards) if hasattr(self, '_current_episode_reward') else 0)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0),
                'decisions_made': len(self.decision_history)
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats
        }


class BalancedAgent:
    """
    A balanced agent that tries to optimize both throughput and quality.
    
    This agent considers:
    1. Current system load
    2. Wait times
    3. Equipment utilization
    4. Quality preservation
    """
    
    def __init__(self, aggressiveness: float = 0.7):
        self.name = "Balanced Agent"
        self.aggressiveness = aggressiveness  # How aggressive to be in processing
        self.episode_rewards = []
        self.episode_stats = []
        self.step_count = 0
    
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        """
        Select action using balanced optimization approach.
        """
        if not info:
            return 0, 0
        
        pending_orders = info.get('pending_orders', 0)
        buffer_orders = info.get('buffer_orders', 0)
        self.step_count += 1
        
        # Buffer management - prioritize based on buffer fullness and time
        if buffer_orders > 0:
            if buffer_orders >= 3:  # High priority when buffer is getting full
                buffer_action = 1
            elif buffer_orders >= 1 and self.step_count % 3 == 0:  # Periodic processing
                buffer_action = 1
            else:
                buffer_action = 0
        else:
            buffer_action = 0
        
        # Order processing - balance throughput with system capacity
        if pending_orders > 0:
            # More aggressive early in the shift, more conservative later
            time_factor = min(1.0, self.step_count / 200.0)  # Ramp up over ~3 hours
            
            if buffer_orders < 2:  # Plenty of buffer space
                order_action = 1 if random.random() < self.aggressiveness else 0
            elif buffer_orders < 4:  # Moderate buffer usage
                order_action = 1 if random.random() < (self.aggressiveness * 0.6) else 0
            else:  # High buffer usage
                order_action = 1 if random.random() < (self.aggressiveness * 0.3) else 0
        else:
            order_action = 0
        
        return order_action, buffer_action
    
    def reset(self):
        """Reset agent state for new episode"""
        self.step_count = 0
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        """Update agent statistics"""
        if done:
            self.episode_rewards.append(sum(self.episode_rewards) if hasattr(self, '_current_episode_reward') else 0)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0),
                'total_steps': self.step_count
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats
        }


class AdaptiveAgent:
    """
    An adaptive agent that learns from recent performance and adjusts its strategy.
    
    This agent:
    1. Tracks recent success/failure patterns
    2. Adapts processing rates based on outcomes
    3. Learns optimal buffer management
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.name = "Adaptive Agent"
        self.learning_rate = learning_rate
        self.episode_rewards = []
        self.episode_stats = []
        
        # Adaptive parameters
        self.order_processing_rate = 0.7  # Probability of processing an order
        self.buffer_urgency_threshold = 3  # When to prioritize buffer
        self.recent_rewards = []
        self.recent_outcomes = []
        
        self.step_count = 0
        self.last_reward = 0
        self.last_action = (0, 0)
    
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        """
        Select action using adaptive strategy based on recent performance.
        """
        if not info:
            return 0, 0
        
        pending_orders = info.get('pending_orders', 0)
        buffer_orders = info.get('buffer_orders', 0)
        self.step_count += 1
        
        # Adapt based on recent performance
        if len(self.recent_rewards) > 10:
            avg_recent_reward = np.mean(self.recent_rewards[-10:])
            if avg_recent_reward > 0:
                # Good performance, be more aggressive
                self.order_processing_rate = min(0.9, self.order_processing_rate + self.learning_rate * 0.1)
            else:
                # Poor performance, be more conservative
                self.order_processing_rate = max(0.3, self.order_processing_rate - self.learning_rate * 0.1)
        
        # Adaptive buffer management
        if buffer_orders >= self.buffer_urgency_threshold:
            buffer_action = 1
        elif buffer_orders > 0 and random.random() < 0.5:
            buffer_action = 1
        else:
            buffer_action = 0
        
        # Adaptive order processing
        if pending_orders > 0:
            # Consider system state
            system_load = (buffer_orders / 5.0)  # Normalize buffer load
            adjusted_rate = self.order_processing_rate * (1.0 - system_load * 0.5)
            
            order_action = 1 if random.random() < adjusted_rate else 0
        else:
            order_action = 0
        
        self.last_action = (order_action, buffer_action)
        return order_action, buffer_action
    
    def reset(self):
        """Reset agent state for new episode"""
        self.step_count = 0
        self.recent_rewards = []
        self.recent_outcomes = []
        self.last_reward = 0
        self.last_action = (0, 0)
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        """Update agent with learning from recent performance"""
        # Track rewards for adaptation
        step_reward = reward - self.last_reward
        self.recent_rewards.append(step_reward)
        self.last_reward = reward
        
        # Keep only recent history
        if len(self.recent_rewards) > 50:
            self.recent_rewards = self.recent_rewards[-30:]
        
        if done:
            self.episode_rewards.append(reward)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0),
                'total_steps': self.step_count,
                'final_processing_rate': self.order_processing_rate
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats,
            'current_processing_rate': self.order_processing_rate
        }


class OptimalTimingAgent:
    """
    An agent that tries to optimize timing based on manufacturing process knowledge.
    
    This agent understands:
    1. Total processing time is ~90 minutes
    2. Buffer wait time should be minimized
    3. Equipment utilization patterns
    """
    
    def __init__(self, shift_duration: float = 8.0):
        self.name = "Optimal Timing Agent"
        self.episode_rewards = []
        self.episode_stats = []
        self.step_count = 0
        self.shift_duration = shift_duration * 60  # Convert to minutes
        
        # Process knowledge
        self.total_process_time = 90  # minutes
        self.buffer_max_safe_time = 3  # minutes
        
    def select_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Tuple[int, int]:
        """
        Select action using process timing optimization.
        """
        if not info:
            return 0, 0
        
        pending_orders = info.get('pending_orders', 0)
        buffer_orders = info.get('buffer_orders', 0)
        self.step_count += 1
        
        # Calculate shift progress
        max_steps = int(self.shift_duration / 1.0)  # Assuming 1-minute time steps
        shift_progress = self.step_count / max_steps
        
        # Buffer management - urgent when orders are waiting
        if buffer_orders > 0:
            # Always process buffer orders to minimize wait time
            buffer_action = 1
        else:
            buffer_action = 0
        
        # Order processing - consider timing and capacity
        if pending_orders > 0:
            # Early in shift: be aggressive to get orders started
            if shift_progress < 0.3:  # First ~2.4 hours
                order_action = 1
            # Mid shift: balance based on buffer state
            elif shift_progress < 0.7:  # Next ~3.2 hours
                if buffer_orders < 2:
                    order_action = 1
                elif buffer_orders < 4:
                    order_action = 1 if self.step_count % 2 == 0 else 0
                else:
                    order_action = 0
            # Late in shift: only if buffer is empty (orders might not complete)
            else:
                order_action = 1 if buffer_orders == 0 else 0
        else:
            order_action = 0
        
        return order_action, buffer_action
    
    def reset(self):
        """Reset agent state for new episode"""
        self.step_count = 0
    
    def update(self, observation: np.ndarray, action: Tuple[int, int], 
               reward: float, next_observation: np.ndarray, done: bool, info: Dict[str, Any]):
        """Update agent statistics"""
        if done:
            self.episode_rewards.append(reward)
            self.episode_stats.append({
                'total_orders': info.get('total_orders', 0),
                'completed_panels': info.get('completed_panels', 0),
                'defective_panels': info.get('defective_panels', 0),
                'quality_score': info.get('quality_score', 0),
                'final_pending_orders': info.get('pending_orders', 0),
                'final_buffer_orders': info.get('buffer_orders', 0),
                'total_steps': self.step_count
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        if not self.episode_stats:
            return {}
        
        latest_stats = self.episode_stats[-1]
        return {
            'episodes_completed': len(self.episode_stats),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'latest_episode': latest_stats
        }
