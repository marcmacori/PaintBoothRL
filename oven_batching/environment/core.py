"""
Dynamic Oven Batching with Energy Tariffs Environment

A Gymnasium environment for simulating a production system with ovens used for curing painted panels.
The agent must decide when to launch oven batches to balance throughput, energy cost, and lateness penalties.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OvenStatus(Enum):
    """Status of an oven"""
    IDLE = 0
    BUSY = 1
    HEATING = 2
    COOLING = 3


class JobType(Enum):
    """Types of jobs with different characteristics"""
    STANDARD = 0
    PREMIUM = 1
    RUSH = 2
    BULK = 3


@dataclass
class Job:
    """Represents a job (panel) in the system"""
    id: int
    arrival_time: float
    due_date: float
    completion_time: Optional[float] = None
    
    def is_late(self) -> bool:
        """Check if the job is completed after its due date"""
        return self.completion_time is not None and self.completion_time > self.due_date
    
    def calculate_lateness(self) -> float:
        """Calculate lateness penalty (positive if late, 0 if on time)"""
        if self.completion_time is None:
            return 0.0
        return max(0.0, self.completion_time - self.due_date)
    
class Oven:
    """Represents an oven in the production system"""
    
    def __init__(self, oven_id: int, capacity: int, batch_time: float, 
                 heating_time: float = 15.0, cooling_rate: float = 0.1):
        self.oven_id = oven_id
        self.capacity = capacity
        self.batch_time = batch_time
        self.heating_time = heating_time
        self.cooling_rate = cooling_rate
        
        # Temperature dynamics
        self.temperature = 0.0  # 0 = cold, 1.0 = operating temperature
        self.target_temperature = 1.0
        self.heating_start_time = 0.0
        self.cooling_start_time = 0.0
        
        # Status and timing
        self.status = OvenStatus.IDLE
        self.completion_time = 0.0
        self.current_batch_jobs: List[Job] = []
    
    def is_ready_to_start(self, current_time: float) -> bool:
        """Check if oven is ready to start a new batch (at temperature and idle)"""
        return self.status == OvenStatus.IDLE and self.temperature >= 1.0
    
    def needs_heating(self, current_time: float) -> bool:
        """Check if oven needs to be heated before starting a batch"""
        return self.status == OvenStatus.IDLE and self.temperature < 1.0
    
    def update_temperature(self, current_time: float):
        """Update oven temperature based on current status and time"""
        if self.status == OvenStatus.HEATING:
            # Linear heating from current temperature to 1.0 over heating_time
            elapsed = current_time - self.heating_start_time
            # Calculate how much heating is needed (from current temp to 1.0)
            temp_needed = 1.0 - self.temperature
            heating_progress = min(1.0, elapsed / self.heating_time)
            self.temperature = min(1.0, self.temperature + temp_needed * heating_progress)
            
            if self.temperature >= 1.0:
                self.status = OvenStatus.IDLE
                
        elif self.status == OvenStatus.BUSY:
            # Maintain temperature during processing
            self.temperature = 1.0
            
        elif self.status == OvenStatus.IDLE:
            # When idle, oven automatically cools down
            if not hasattr(self, 'idle_start_time'):
                self.idle_start_time = current_time
            
            # Exponential cooling: T = T0 * exp(-cooling_rate * elapsed)
            elapsed = current_time - self.idle_start_time
            self.temperature = max(0.0, self.temperature * np.exp(-self.cooling_rate * elapsed))
            
            if self.temperature <= 0.01:  # Consider cold when below 1%
                self.temperature = 0.0
    
    def start_batch(self, jobs: List[Job], current_time: float) -> bool:
        """Start a new batch with given jobs"""
        if not self.is_ready_to_start(current_time) or len(jobs) > self.capacity:
            return False
        
        self.current_batch_jobs = jobs
        self.status = OvenStatus.BUSY
        self.completion_time = current_time + self.batch_time
        # Reset idle start time since we're no longer idle
        if hasattr(self, 'idle_start_time'):
            delattr(self, 'idle_start_time')
        return True
    
    def start_heating(self, current_time: float) -> bool:
        """Start heating the oven from current temperature to operating temperature"""
        if self.status != OvenStatus.IDLE:
            return False
        
        # Can start heating from any temperature (including if already at operating temp)
        self.status = OvenStatus.HEATING
        self.heating_start_time = current_time
        # Reset idle start time since we're no longer idle
        if hasattr(self, 'idle_start_time'):
            delattr(self, 'idle_start_time')
        return True
    
    def stop_heating(self, current_time: float):
        """Stop heating and start cooling"""
        if self.status == OvenStatus.HEATING:
            self.status = OvenStatus.COOLING
            self.cooling_start_time = current_time
    

    
    def complete_batch(self, current_time: float) -> List[Job]:
        """Complete the current batch and return the jobs"""
        if self.status == OvenStatus.IDLE or current_time < self.completion_time:
            return []
        
        for job in self.current_batch_jobs:
            job.completion_time = current_time
        
        completed_jobs = self.current_batch_jobs.copy()
        self.current_batch_jobs = []
        
        # Set oven to idle - automatic cooling will start immediately
        self.status = OvenStatus.IDLE
        self.idle_start_time = current_time  # Start tracking idle time for cooling
        
        return completed_jobs
    
    def get_time_to_completion(self, current_time: float) -> float:
        """Get time remaining until batch completion"""
        if self.status == OvenStatus.IDLE:
            return 0.0
        elif self.status == OvenStatus.HEATING:
            return max(0.0, self.heating_start_time + self.heating_time - current_time)
        elif self.status == OvenStatus.BUSY:
            return max(0.0, self.completion_time - current_time)
        else:  # Should not reach here
            return 0.0
    
    def get_time_to_ready(self, current_time: float) -> float:
        """Get time until oven is ready for a new batch"""
        if self.is_ready_to_start(current_time):
            return 0.0
        elif self.status == OvenStatus.HEATING:
            return max(0.0, self.heating_start_time + self.heating_time - current_time)
        elif self.status == OvenStatus.BUSY:
            return max(0.0, self.completion_time - current_time)
        else:  # IDLE but cold
            return 0.0  # Can start heating immediately


class DynamicOvenBatchingEnv(gym.Env):
    """
    Dynamic Oven Batching with Energy Tariffs Environment
    
    A production system simulation where the agent must decide when to launch oven batches
    to balance throughput, energy cost, and lateness penalties.
    """
    
    def __init__(self, 
                 num_ovens: int = 1,
                 oven_capacity: int = 9,
                 batch_time: float = 10.0,
                 batch_energy_cost: float = 1.0,
                 heating_time: float = 15.0,
                 cooling_rate: float = 0.1,
                 horizon: float = 1440.0,
                 arrival_rate: float = 0.5,
                 due_date_offset_mean: float = 60.0,
                 due_date_offset_std: float = 20.0,
                 energy_alpha: float = 1.0,
                 lateness_beta: float = 2.0,
                 idle_penalty: float = 0.01,
                 underfill_penalty: float = 0.1,
                 use_dynamic_arrivals: bool = True,
                 seed: Optional[int] = None):
        """Initialize the environment with configurable parameters"""
        super().__init__()
        
        # Store parameters
        self.num_ovens = num_ovens
        self.oven_capacity = oven_capacity
        self.batch_time = batch_time
        self.batch_energy_cost = batch_energy_cost
        self.heating_time = heating_time
        self.cooling_rate = cooling_rate
        self.horizon = horizon
        self.arrival_rate = arrival_rate
        self.due_date_offset_mean = due_date_offset_mean
        self.due_date_offset_std = due_date_offset_std
        self.energy_alpha = energy_alpha
        self.lateness_beta = lateness_beta
        self.idle_penalty = idle_penalty
        self.underfill_penalty = underfill_penalty
        self.use_dynamic_arrivals = use_dynamic_arrivals
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # Initialize ovens
        self.ovens = [Oven(i, oven_capacity, batch_time, heating_time, cooling_rate) 
                     for i in range(num_ovens)]
        
        # Define spaces
        # Multidiscrete action space: [action_type, oven_id, num_panels]
        # action_type: 0=wait, 1=launch, 2=heat
        # oven_id: which oven to use (0 to num_ovens-1)
        # num_panels: number of panels for launch action (0 to oven_capacity)
        self.action_space = spaces.MultiDiscrete([
            3,                    # Action type: 0=wait, 1=launch, 2=heat
            num_ovens,            # Oven ID
            oven_capacity + 1     # Number of panels (0 to oven_capacity)
        ])
        
        obs_dim = 7 + 6 * num_ovens  # 7 global + 6 per oven (busy, heating, cooling, time_to_completion, time_to_ready, temperature)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Initialize state
        self.reset()
    
    def _generate_tariff(self, time: float) -> float:
        """Generate energy tariff based on time of day"""
        hour = (time % 1440) / 60  # Convert to hours
        if 8 <= hour <= 18:  # Peak hours
            return 1.5
        elif 6 <= hour <= 8 or 18 <= hour <= 22:  # Shoulder hours
            return 1.2
        else:  # Off-peak hours
            return 0.8
    
    def _get_dynamic_arrival_rate(self, time: float) -> float:
        """
        Get dynamic arrival rate based on shift patterns
        
        Args:
            time: Current time in minutes
            
        Returns:
            Arrival rate (jobs per minute)
        """
        hour = (time % 1440) / 60  # Convert to hours (24-hour cycle)
        
        # Define shift patterns (3 shifts of 8 hours each)
        # Shift 1: 6:00-14:00 (6-14 hours) - Morning shift
        # Shift 2: 14:00-22:00 (14-22 hours) - Afternoon shift
        # Shift 3: 22:00-6:00 (22-6 hours) - Night shift
        
        # Base arrival rate
        base_rate = self.arrival_rate
        
        # Determine which shift we're in and set multiplier
        if 6 <= hour < 14:  # Shift 1 (morning)
            shift_multiplier = 2.5  # Highest activity
        elif 14 <= hour < 22:  # Shift 2 (afternoon)
            shift_multiplier = 2.0  # Medium activity
        else:  # Shift 3 (night shift)
            shift_multiplier = 1.5  # Lower activity
        
        # Calculate position within shift (0 to 8 hours)
        if 6 <= hour < 14:
            shift_position = hour - 6
        elif 14 <= hour < 22:
            shift_position = hour - 14
        else:
            shift_position = hour - 22 if hour >= 22 else hour + 2  # Handle wrap-around
        
        # Create peak patterns within each shift
        if shift_position <= 2:  # First 2 hours of shift
            # Beginning of shift peak - highest activity
            peak_factor = 1.0 + (2.0 - shift_position) * 0.5
            arrival_rate = base_rate * shift_multiplier * peak_factor
        elif shift_position >= 6 and shift_position <= 8:  # Last 2 hours of shift
            # End of shift peak - rush to finish work
            end_peak_factor = 1.0 + (shift_position - 6) * 0.3
            arrival_rate = base_rate * shift_multiplier * end_peak_factor
        else:  # Middle of shift (hours 2-6)
            # Valley period - lower activity
            arrival_rate = base_rate * shift_multiplier * 0.7
        
        return arrival_rate
    
    def _generate_next_arrival(self) -> float:
        """Generate time until next job arrival using dynamic or constant arrival rate"""
        if self.use_dynamic_arrivals:
            current_arrival_rate = self._get_dynamic_arrival_rate(self.current_time)
        else:
            current_arrival_rate = self.arrival_rate
        return self.rng.exponential(1.0 / current_arrival_rate)
    
    def _generate_due_date_offset(self) -> float:
        """Generate due date offset for a new job"""
        return self.rng.normal(self.due_date_offset_mean, self.due_date_offset_std)
    
    def _get_state_vector(self) -> np.ndarray:
        """Get the current state as a vector"""
        # Calculate queue statistics
        if self.queue:
            waiting_times = [self.current_time - job.arrival_time for job in self.queue]
            avg_waiting_time = np.mean(waiting_times)
            
            lateness_risks = []
            urgent_jobs = 0
            for job in self.queue:
                time_to_due = job.due_date - self.current_time
                lateness_risks.append(max(0, -time_to_due))
                if time_to_due < 30:  # Jobs due within 30 minutes are urgent
                    urgent_jobs += 1
            
            min_lateness_risk = min(lateness_risks) if lateness_risks else 0
            max_lateness_risk = max(lateness_risks) if lateness_risks else 0
        else:
            avg_waiting_time = 0.0
            min_lateness_risk = 0.0
            max_lateness_risk = 0.0
            urgent_jobs = 0
        
        # Build state vector
        state = [
            self.current_time / self.horizon,  # Normalized time
            len(self.queue),  # Queue length
            avg_waiting_time,
            min_lateness_risk,
            max_lateness_risk,
            urgent_jobs,
            self._generate_tariff(self.current_time)
        ]
        
        # Add oven information
        for oven in self.ovens:
            state.append(1.0 if oven.status == OvenStatus.BUSY else 0.0)
            state.append(1.0 if oven.status == OvenStatus.HEATING else 0.0)
            state.append(1.0 if oven.status == OvenStatus.IDLE and oven.temperature < 0.99 else 0.0)  # Cooling indicator
            state.append(oven.get_time_to_completion(self.current_time))
            state.append(oven.get_time_to_ready(self.current_time))
            state.append(oven.temperature)  # Add temperature information
        
        return np.array(state, dtype=np.float32)
    
    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for valid actions in multidiscrete format"""
        # Initialize masks for each dimension
        action_type_mask = np.ones(3, dtype=bool)  # All action types are always valid
        oven_mask = np.zeros(self.num_ovens, dtype=bool)
        num_panels_mask = np.zeros(self.oven_capacity + 1, dtype=bool)
        
        queue_length = len(self.queue)
        
        # Check which ovens are available for different actions
        for i, oven in enumerate(self.ovens):
            if oven.is_ready_to_start(self.current_time):
                # Oven is ready for launch action
                oven_mask[i] = True
                # Allow launch actions with valid number of panels
                for panels in range(1, min(queue_length + 1, self.oven_capacity + 1)):
                    num_panels_mask[panels] = True
            elif oven.status == OvenStatus.IDLE:
                # Oven is idle - can be heated
                oven_mask[i] = True
            # Note: Ovens that are HEATING or BUSY are not available for any actions
        
        # For wait action (action_type=0), num_panels is ignored, so we don't need to mask it
        # For launch action (action_type=1), we only allow valid panel counts (already set above)
        # For heat action (action_type=2), num_panels is ignored, so we don't need to mask it
        
        return [action_type_mask, oven_mask, num_panels_mask]
    
    def _advance_time_to_next_event(self) -> float:
        """Advance time to the next event"""
        next_arrival = self.next_arrival_time
        
        # Check for batch completions
        next_completion = min([oven.completion_time for oven in self.ovens 
                             if oven.status == OvenStatus.BUSY], default=np.inf)
        
        # Check for heating completions - but only if oven is actually still heating
        next_heating = min([oven.heating_start_time + oven.heating_time for oven in self.ovens 
                           if oven.status == OvenStatus.HEATING and 
                           oven.heating_start_time + oven.heating_time > self.current_time], default=np.inf)
        
        next_event_time = min(next_arrival, next_completion, next_heating)
        time_advanced = next_event_time - self.current_time
        self.current_time = next_event_time
        return time_advanced
    
    def _advance_time_small_step(self, max_step: float = 1.0) -> float:
        """Advance time by a small step, not jumping to next events"""
        next_arrival = self.next_arrival_time
        
        # Check for batch completions
        next_completion = min([oven.completion_time for oven in self.ovens 
                             if oven.status == OvenStatus.BUSY], default=np.inf)
        
        # Check for heating completions - but only if oven is actually still heating
        next_heating = min([oven.heating_start_time + oven.heating_time for oven in self.ovens 
                           if oven.status == OvenStatus.HEATING and 
                           oven.heating_start_time + oven.heating_time > self.current_time], default=np.inf)
        
        next_event_time = min(next_arrival, next_completion, next_heating)
        
        # Only advance by a small step, not to the next event
        time_advanced = min(max_step, next_event_time - self.current_time)
        self.current_time += time_advanced
        return time_advanced
    
    def _process_arrivals(self):
        """Process any job arrivals at the current time"""
        while self.current_time >= self.next_arrival_time:
            due_date_offset = self._generate_due_date_offset()
            job = Job(
                id=self.job_counter,
                arrival_time=self.current_time,
                due_date=self.current_time + due_date_offset
            )
            
            self.queue.append(job)
            self.job_counter += 1
            self.next_arrival_time += self._generate_next_arrival()
    
    def _process_oven_completions(self):
        """Process any oven completions at the current time"""
        for oven in self.ovens:
            # Update temperature for all ovens
            oven.update_temperature(self.current_time)
            
            # Process batch completions
            if oven.status == OvenStatus.BUSY and self.current_time >= oven.completion_time:
                completed_jobs = oven.complete_batch(self.current_time)
                self.completed_jobs.extend(completed_jobs)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Reset environment state
        self.current_time = 0.0
        self.job_counter = 0
        self.queue: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.total_reward = 0.0
        self.total_energy_cost = 0.0
        self.total_lateness_penalty = 0.0
        
        # Reset ovens
        for oven in self.ovens:
            oven.status = OvenStatus.IDLE
            oven.completion_time = 0.0
            oven.current_batch_jobs = []
            oven.temperature = 0.0  # Start cold
            oven.heating_start_time = 0.0
            oven.cooling_start_time = 0.0
        
        # Schedule first arrival
        self.next_arrival_time = self._generate_next_arrival()
        self._process_arrivals()
        
        observation = self._get_state_vector()
        info = {
            'queue_length': len(self.queue),
            'completed_jobs': len(self.completed_jobs),
            'total_reward': self.total_reward
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        action_type, oven_id, num_panels = action
        
        # Check if action is valid
        action_mask = self._get_action_mask()
        
        # Basic validation
        if (action_type < 0 or action_type >= 3 or 
            oven_id < 0 or oven_id >= len(self.ovens) or
            num_panels < 0 or num_panels > self.oven_capacity):
            reward = -1.0
            time_advanced = self._advance_time_small_step(max_step=1.0)
        # Check action-specific validity
        elif action_type == 1 and (not action_mask[1][oven_id] or not action_mask[2][num_panels]):
            # Launch action: check oven availability and valid panel count
            reward = -1.0
            time_advanced = self._advance_time_small_step(max_step=1.0)
        elif action_type == 2 and not action_mask[1][oven_id]:
            # Heat action: check oven availability
            reward = -1.0
            time_advanced = self._advance_time_small_step(max_step=1.0)

        else:
            reward = 0.0
            time_advanced = 0.0  # Default value
            
            if action_type == 0:
                # Wait action - advance to next event
                time_advanced = self._advance_time_to_next_event()
                reward -= self.idle_penalty * time_advanced
            elif action_type == 1:
                # Launch action - use small step to allow gradual batch processing observation
                if oven_id >= len(self.ovens):
                    reward = -1.0
                else:
                    oven = self.ovens[oven_id]
                    
                    if not oven.is_ready_to_start(self.current_time):
                        reward = -1.0
                    elif num_panels > len(self.queue):
                        reward = -1.0
                    elif num_panels == 0:
                        reward = -1.0  # Can't launch 0 panels
                    else:
                        # Launch batch
                        jobs_to_process = self.queue[:num_panels]
                        self.queue = self.queue[num_panels:]
                        
                        success = oven.start_batch(jobs_to_process, self.current_time)
                        if success:
                            # Calculate energy cost
                            current_tariff = self._generate_tariff(self.current_time)
                            energy_cost = self.batch_energy_cost * current_tariff
                            self.total_energy_cost += energy_cost
                            reward -= self.energy_alpha * energy_cost
                            
                            # Penalty for underfilled batch
                            if num_panels < self.oven_capacity * 0.5:
                                reward -= self.underfill_penalty
                            
                            # Use small time step to allow gradual batch processing observation
                            time_advanced = self._advance_time_small_step(max_step=3.0)
                        else:
                            reward = -1.0
                            time_advanced = self._advance_time_small_step(max_step=1.0)
            elif action_type == 2:
                # Heating action - use small step to allow gradual heating observation
                if oven_id >= len(self.ovens):
                    reward = -1.0
                else:
                    oven = self.ovens[oven_id]
                    if oven.start_heating(self.current_time):
                        # Heating energy cost
                        current_tariff = self._generate_tariff(self.current_time)
                        heating_cost = self.batch_energy_cost * 0.5 * current_tariff  # Heating costs less than processing
                        self.total_energy_cost += heating_cost
                        reward -= self.energy_alpha * heating_cost
                        
                        # Use small time step to allow gradual heating observation
                        time_advanced = self._advance_time_small_step(max_step=2.0)
                    else:
                        reward = -1.0
                        time_advanced = self._advance_time_small_step(max_step=1.0)
        
        # Process events
        self._process_arrivals()
        self._process_oven_completions()
        
        # Calculate rewards for completed jobs
        for job in self.completed_jobs:
            if job.completion_time == self.current_time:
                reward += 1.0  # Throughput reward
                
                if job.is_late():
                    lateness_penalty = self.lateness_beta * job.calculate_lateness()
                    self.total_lateness_penalty += lateness_penalty
                    reward -= lateness_penalty
        
        # Add intermediate rewards for good decisions
        if action_type == 2:  # Heating action
            # Small positive reward for proactive heating when queue is building up
            if len(self.queue) > self.oven_capacity * 0.5:
                reward += 0.1
        elif action_type == 1 and num_panels > 0:  # Launch action
            # Small positive reward for processing jobs when oven is ready
            reward += 0.05

        
        self.total_reward += reward
        
        # Check termination
        terminated = self.current_time >= self.horizon
        truncated = False
        
        observation = self._get_state_vector()
        info = {
            'queue_length': len(self.queue),
            'completed_jobs': len(self.completed_jobs),
            'total_reward': self.total_reward,
            'total_energy_cost': self.total_energy_cost,
            'total_lateness_penalty': self.total_lateness_penalty,
            'current_time': self.current_time,
            'action_mask': self._get_action_mask()
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the current state (simplified)"""
        if mode == 'human':
            print(f"Time: {self.current_time:.1f}/{self.horizon:.1f} | "
                  f"Queue: {len(self.queue)} | "
                  f"Completed: {len(self.completed_jobs)} | "
                  f"Reward: {self.total_reward:.2f}")
    
    def get_action_mask(self) -> np.ndarray:
        """Get the current action mask"""
        return self._get_action_mask()
