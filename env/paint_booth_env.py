import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque

class PaintType(Enum):
    WATER = "water"
    SOLVENT = "solvent"

class ProcessStage(Enum):
    PAINT_ROBOT = "paint_robot"
    FLASH_OFF = "flash_off"
    OVEN = "oven"
    BUFFER = "buffer"
    VARNISH_ROBOT = "varnish_robot"
    VARNISH_FLASH_OFF = "varnish_flash_off"
    VARNISH_OVEN = "varnish_oven"
    COMPLETE = "complete"

@dataclass
class Panel:
    """Represents a single panel in the system"""
    panel_id: int
    order_id: int
    paint_type: PaintType
    stage: ProcessStage
    stage_start_time: float
    stage_duration: float
    quality_factor: float = 1.0
    defect_probability: float = 0.0
    order_size: int = 1  # Total number of panels in the original order

@dataclass
class Order:
    """Represents an order containing 1-3 panels"""
    order_id: int
    panels: List[Panel]
    paint_type: PaintType
    arrival_time: float
    priority: float = 1.0

class Equipment:
    """Base class for all equipment in the paint booth"""
    def __init__(self, name: str, capacity: int, processing_time: float, is_continuous: bool = False):
        self.name = name
        self.capacity = capacity
        self.processing_time = processing_time
        self.panels: List[Panel] = []
        self.busy_until: float = 0.0
        self.is_continuous = is_continuous  # True for continuous processors, False for batch processors
    
    def is_available(self, current_time: float) -> bool:
        if self.is_continuous:
            # Continuous processors are available as long as they have capacity
            return len(self.panels) < self.capacity
        else:
            # Batch processors must also not be busy
            return len(self.panels) < self.capacity and current_time >= self.busy_until
    
    def can_accept(self, panels: List[Panel], current_time: float) -> bool:
        if self.is_continuous:
            # Only check capacity, not busy time
            return len(self.panels) + len(panels) <= self.capacity
        else:
            # Check both capacity and busy time
            return (len(self.panels) + len(panels) <= self.capacity and 
                    current_time >= self.busy_until)
    
    def add_panels(self, panels: List[Panel], current_time: float):
        if self.can_accept(panels, current_time):
            for panel in panels:
                panel.stage_start_time = current_time
                panel.stage_duration = self.processing_time
                self.panels.append(panel)
            # Only set busy_until for batch processors
            if not self.is_continuous:
                self.busy_until = current_time + self.processing_time
    
    def get_completed_panels(self, current_time: float) -> List[Panel]:
        completed = []
        remaining = []
        
        for panel in self.panels:
            if current_time >= panel.stage_start_time + panel.stage_duration:
                completed.append(panel)
            else:
                remaining.append(panel)
        
        self.panels = remaining
        if not self.panels:
            self.busy_until = 0.0
            
        return completed

class PaintRobot(Equipment):
    """Paint robot that can handle up to 3 panels from one order (batch processor)"""
    def __init__(self, name: str, paint_type: PaintType):
        super().__init__(name, capacity=3, processing_time=15.0, is_continuous=False)  # Batch processor
        self.paint_type = paint_type
    
    def can_accept_order(self, order: Order, current_time: float) -> bool:
        return (order.paint_type == self.paint_type and 
                len(order.panels) <= self.capacity and
                len(self.panels) == 0 and  # Robot must be empty for new order
                current_time >= self.busy_until)

class FlashOffCabinet(Equipment):
    """Flash off cabinet with capacity for 12 panels (continuous processor)"""
    def __init__(self, name: str, paint_type: PaintType):
        super().__init__(name, capacity=12, processing_time=10.0, is_continuous=True)  # Continuous processor
        self.paint_type = paint_type

class Oven(Equipment):
    """Oven with capacity for 9 panels (continuous processor)"""
    def __init__(self, name: str, paint_type: PaintType):
        super().__init__(name, capacity=9, processing_time=20.0, is_continuous=True)  # Continuous processor
        self.paint_type = paint_type

class BufferZone:
    """Buffer zone where panels wait before varnishing (max 5 minutes)"""
    def __init__(self):
        self.panels: List[Panel] = []
        self.max_wait_time = 5.0  # 5 minutes max
    
    def add_panel(self, panel: Panel, current_time: float):
        panel.stage = ProcessStage.BUFFER
        panel.stage_start_time = current_time
        self.panels.append(panel)
    
    def get_panels(self, current_time: float) -> List[Panel]:
        # Update defect probability based on wait time
        for panel in self.panels:
            wait_time = current_time - panel.stage_start_time
            if wait_time > self.max_wait_time:
                # Remove panels that waited too long (defect)
                panel.defect_probability = 1.0
            else:
                # Increase defect probability based on wait time
                panel.defect_probability += (wait_time / self.max_wait_time) * 0.1
        
        return self.panels.copy()
    
    def remove_panels(self, panels_to_remove: List[Panel]):
        for panel in panels_to_remove:
            if panel in self.panels:
                self.panels.remove(panel)

class OrderGenerator:
    """Generates orders with Poisson distribution and shift patterns"""
    def __init__(self, shift_duration: float = 8.0):  # 8 hour shifts
        self.shift_duration = shift_duration * 60  # Convert to minutes
        self.peak_times = [0, 3.5 * 60, 6.5 * 60]  # Peak times in minutes
        self.base_rate = 2.0  # Base Poisson rate
        self.peak_multiplier = 3.0  # Multiplier during peaks
        self.panel_id_counter = 0
        self.order_id_counter = 0
    
    def get_order_rate(self, current_time: float) -> float:
        """Get current order generation rate based on time"""
        shift_time = current_time % self.shift_duration
        
        # Check if we're near a peak time (within 30 minutes)
        for peak_time in self.peak_times:
            if abs(shift_time - peak_time) <= 30:
                return self.base_rate * self.peak_multiplier
        
        return self.base_rate
    
    def generate_orders(self, current_time: float, time_step: float) -> List[Order]:
        """Generate orders for the current time step"""
        rate = self.get_order_rate(current_time)
        num_orders = np.random.poisson(rate * time_step / 60)  # Convert time_step to hours
        
        orders = []
        for _ in range(num_orders):
            # Random paint type
            paint_type = random.choice([PaintType.WATER, PaintType.SOLVENT])
            
            # Random number of panels (1-3)
            num_panels = random.randint(1, 3)
            
            # Create panels
            panels = []
            for _ in range(num_panels):
                panel = Panel(
                    panel_id=self.panel_id_counter,
                    order_id=self.order_id_counter,
                    paint_type=paint_type,
                    stage=ProcessStage.PAINT_ROBOT,
                    stage_start_time=current_time,
                    stage_duration=0.0,
                    order_size=num_panels
                )
                panels.append(panel)
                self.panel_id_counter += 1
            
            # Create order
            order = Order(
                order_id=self.order_id_counter,
                panels=panels,
                paint_type=paint_type,
                arrival_time=current_time
            )
            orders.append(order)
            self.order_id_counter += 1
        
        return orders
    
def calculate_quality_factor(actual_time: float, optimal_time: float) -> Tuple[float, float]:
    """Calculate quality factor and defect probability based on timing"""
    time_ratio = actual_time / optimal_time
    
    if time_ratio < 1.0:
        # Too little time - increasing chance of defect
        defect_prob = (1.0 - time_ratio) * 0.5  # Up to 50% defect chance
        quality_factor = time_ratio
    elif time_ratio <= 1.1:
        # Slightly more time - better quality
        defect_prob = max(0.0, 0.1 - (time_ratio - 1.0) * 0.5)  # Lower defect chance
        quality_factor = min(1.1, 1.0 + (time_ratio - 1.0) * 0.5)
    else:
        # Too much time - increasing chance of defect
        defect_prob = (time_ratio - 1.1) * 0.3  # Increasing defect chance
        quality_factor = max(0.5, 1.1 - (time_ratio - 1.1) * 0.2)
    
    return quality_factor, min(defect_prob, 1.0)


class PaintBoothEnv(gym.Env):
    """
    Paint Booth Scheduling Environment
    
    This environment simulates a paint manufacturing scheduling problem where
    an agent must decide which orders to process and when to optimize throughput
    and quality while managing equipment constraints.
    """
    
    def __init__(self, shift_duration: float = 8.0, time_step: float = 1.0):
        super(PaintBoothEnv, self).__init__()
        
        # Add metadata for render modes
        self.metadata = {'render_modes': ['human', 'rgb_array']}
        
        # Time management
        self.shift_duration = shift_duration * 60  # Convert to minutes
        self.time_step = time_step  # Time step in minutes
        self.current_time = 0.0
        self.max_episode_steps = int(self.shift_duration / self.time_step)
        
        # Initialize equipment
        self.water_paint_robot = PaintRobot("Water Paint Robot", PaintType.WATER)
        self.solvent_varnish_paint_robot = PaintRobot("Solvent/Varnish Paint Robot", PaintType.SOLVENT)
        
        # Flash off cabinets: 1 for water, 2 shared for solvent/varnish
        self.water_flash_off = FlashOffCabinet("Water Flash Off", PaintType.WATER)
        self.solvent_varnish_flash_off_1 = FlashOffCabinet("Solvent/Varnish Flash Off 1", PaintType.SOLVENT)
        self.solvent_varnish_flash_off_2 = FlashOffCabinet("Solvent/Varnish Flash Off 2", PaintType.SOLVENT)
        
        # Ovens: 1 for water, 2 shared for solvent/varnish
        self.water_oven = Oven("Water Oven", PaintType.WATER)
        self.solvent_varnish_oven_1 = Oven("Solvent/Varnish Oven 1", PaintType.SOLVENT)
        self.solvent_varnish_oven_2 = Oven("Solvent/Varnish Oven 2", PaintType.SOLVENT)
        
        self.buffer_zone = BufferZone()
        
        # Order management
        self.order_generator = OrderGenerator(shift_duration)
        self.pending_orders = deque()
        self.completed_panels = []
        self.defective_panels = []
        
        # Statistics
        self.total_orders_generated = 0
        self.total_panels_completed = 0
        self.total_panels_defective = 0
        self.total_quality_score = 0.0
        
        # Define action space
        # Actions: [order_selection, buffer_action]
        # order_selection: which order to process (0 = no action, 1-N = order index)
        # buffer_action: which complete order to move from buffer to varnish (0 = no action, 1-M = order index)
        # Note: Equipment selection is automatic based on paint type and availability
        max_pending_orders = 50  # Maximum orders in queue
        max_buffer_orders = 20  # Maximum complete orders that can be in buffer
        
        self.action_space = spaces.MultiDiscrete([
            max_pending_orders + 1,  # +1 for no action
            max_buffer_orders + 1   # +1 for no action
        ])
        
        # Define observation space
        # Observation includes:
        # - Current time (normalized)
        # - Equipment status (busy_until times, capacities, current loads)
        # - Pending orders information
        # - Buffer zone status
        # - Quality metrics
        
        obs_size = (
            1 +  # Current time (normalized)
            8 * 3 +  # 8 unique equipment pieces * 3 features each (busy_until, capacity, current_load)
            max_pending_orders * 4 +  # Orders * 4 features (paint_type, num_panels, arrival_time, priority)
            3 +  # Buffer zone info (num_panels, avg_wait_time, max_wait_time)
            max_buffer_orders * 3 +  # Buffer orders * 3 features (num_panels, avg_wait_time, paint_type)
            4   # Quality metrics (completion_rate, defect_rate, avg_quality, throughput)
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.max_pending_orders = max_pending_orders
        self.max_buffer_orders = max_buffer_orders
        
        # Random number generator for reproducible resets
        self._np_random = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        # Handle seed for reproducibility
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np.random.RandomState()
        
        self.current_time = 0.0
        
        # Reset all equipment (note: varnish equipment is shared with solvent equipment)
        equipment_list = [
            self.water_paint_robot, self.solvent_varnish_paint_robot,
            self.water_flash_off, self.solvent_varnish_flash_off_1, self.solvent_varnish_flash_off_2,
            self.water_oven, self.solvent_varnish_oven_1, self.solvent_varnish_oven_2
        ]
        
        for equipment in equipment_list:
            equipment.panels = []
            equipment.busy_until = 0.0
        
        self.buffer_zone.panels = []
        
        # Reset order management
        self.order_generator.panel_id_counter = 0
        self.order_generator.order_id_counter = 0
        self.pending_orders.clear()
        self.completed_panels = []
        self.defective_panels = []
        
        # Reset statistics
        self.total_orders_generated = 0
        self.total_panels_completed = 0
        self.total_panels_defective = 0
        self.total_quality_score = 0.0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one time step in the environment"""
        order_selection, buffer_action = action
        
        # Generate new orders
        new_orders = self.order_generator.generate_orders(self.current_time, self.time_step)
        for order in new_orders:
            if len(self.pending_orders) < self.max_pending_orders:
                self.pending_orders.append(order)
                self.total_orders_generated += 1
        
        # Process selected action
        reward = 0.0
        if order_selection > 0 and order_selection <= len(self.pending_orders):
            selected_order = list(self.pending_orders)[order_selection - 1]
            reward += self._process_order(selected_order, 0)  # equipment_selection unused, pass 0
        
        # Process buffer zone action
        if buffer_action > 0:
            reward += self._process_buffer_action(buffer_action)
        
        # Update all equipment and move panels through stages
        reward += self._update_equipment()
        
        # Advance time
        self.current_time += self.time_step
        
        # Check if episode is done
        done = self.current_time >= self.shift_duration
        terminated = done
        truncated = False  # Add truncated flag for new Gymnasium API
        
        # Calculate additional rewards
        reward += self._calculate_additional_rewards()
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'total_orders': self.total_orders_generated,
            'completed_panels': self.total_panels_completed,
            'defective_panels': self.total_panels_defective,
            'quality_score': self.total_quality_score / max(1, self.total_panels_completed),
            'pending_orders': len(self.pending_orders),
            'buffer_orders': len(self._get_complete_orders_in_buffer())
        }
        
        return obs, reward, terminated, truncated, info
    
    def _process_order(self, order: Order, equipment_selection: int) -> float:
        """Process a selected order (equipment_selection parameter is unused)"""
        reward = 0.0
        
        # Determine which paint robot to use (automatic based on paint type)
        if order.paint_type == PaintType.WATER:
            robot = self.water_paint_robot
        else:
            robot = self.solvent_varnish_paint_robot
        
        # Check if robot can accept the order
        if robot.can_accept_order(order, self.current_time):
            robot.add_panels(order.panels, self.current_time)
            self.pending_orders.remove(order)
            
            # Update panel stages
            for panel in order.panels:
                panel.stage = ProcessStage.PAINT_ROBOT
            
            reward += 10.0  # Reward for successfully processing an order
            
            # Bonus for processing orders quickly
            wait_time = self.current_time - order.arrival_time
            if wait_time < 30:  # Less than 30 minutes wait
                reward += 5.0
        else:
            reward -= 1.0  # Penalty for invalid action
        
        return reward
    
    def _process_buffer_action(self, buffer_action: int) -> float:
        """Process agent's decision to move complete orders from buffer to varnish robot"""
        reward = 0.0
        
        # Get list of complete orders in buffer
        complete_orders = self._get_complete_orders_in_buffer()
        
        if buffer_action <= len(complete_orders):
            selected_order_info = complete_orders[buffer_action - 1]
            order_id, panels_to_varnish = selected_order_info
            
            # Check if varnish robot (shared solvent robot) is available
            if self.solvent_varnish_paint_robot.is_available(self.current_time) and len(panels_to_varnish) <= 3:
                # Move panels to varnish robot
                self.solvent_varnish_paint_robot.add_panels(panels_to_varnish, self.current_time)
                self.buffer_zone.remove_panels(panels_to_varnish)
                
                for panel in panels_to_varnish:
                    panel.stage = ProcessStage.VARNISH_ROBOT
                
                reward += 5.0  # Reward for successfully moving order to varnish
                
                # Bonus for moving orders quickly (less buffer wait time)
                avg_wait_time = sum(self.current_time - p.stage_start_time for p in panels_to_varnish) / len(panels_to_varnish)
                if avg_wait_time < 2.0:  # Less than 2 minutes average wait
                    reward += 3.0
                elif avg_wait_time < 5.0:  # Less than 5 minutes (max allowed)
                    reward += 1.0
            else:
                reward -= 2.0  # Penalty for invalid buffer action
        else:
            reward -= 1.0  # Penalty for selecting non-existent buffer order
        
        return reward
    
    def _get_complete_orders_in_buffer(self) -> List[Tuple[int, List[Panel]]]:
        """Get list of complete orders in buffer zone that can be moved to varnish"""
        buffer_panels = self.buffer_zone.get_panels(self.current_time)
        
        # Group panels by order
        orders_in_buffer = {}
        for panel in buffer_panels:
            if panel.order_id not in orders_in_buffer:
                orders_in_buffer[panel.order_id] = []
            orders_in_buffer[panel.order_id].append(panel)
        
        # Find complete orders (all panels from the order are in buffer)
        complete_orders = []
        for order_id, panels in orders_in_buffer.items():
            if len(panels) > 0:
                expected_order_size = panels[0].order_size
                if len(panels) == expected_order_size and expected_order_size <= 3:
                    # Sort by arrival time to prioritize oldest orders first
                    earliest_time = min(panel.stage_start_time for panel in panels)
                    complete_orders.append((earliest_time, order_id, panels))
        
        # Sort by arrival time and return (order_id, panels) tuples
        complete_orders.sort(key=lambda x: x[0])
        return [(order_id, panels) for _, order_id, panels in complete_orders]
    
    def _update_equipment(self) -> float:
        """Update all equipment and move panels through stages"""
        reward = 0.0
        
        # Stage 1: Paint robots to flash off cabinets
        reward += self._move_from_paint_robots()
        
        # Stage 2: Flash off cabinets to ovens
        reward += self._move_from_flash_off()
        
        # Stage 3: Ovens to buffer zone (for water) or complete (for first stage solvent)
        reward += self._move_from_ovens()
        
        # Stage 4: Varnish robot to varnish flash off
        reward += self._move_from_varnish_robot()
        
        # Stage 5: Varnish flash off to varnish ovens
        reward += self._move_from_varnish_flash_off()
        
        # Stage 6: Varnish ovens to completion
        reward += self._move_from_varnish_ovens()
        
        return reward
    
    def _move_from_paint_robots(self) -> float:
        """Move completed panels from paint robots to flash off cabinets"""
        reward = 0.0
        
        # Water paint robot
        completed_panels = self.water_paint_robot.get_completed_panels(self.current_time)
        for panel in completed_panels:
            if self.water_flash_off.can_accept([panel], self.current_time):
                self.water_flash_off.add_panels([panel], self.current_time)
                panel.stage = ProcessStage.FLASH_OFF
                reward += 1.0
            else:
                # Penalty for bottleneck
                reward -= 2.0
                panel.defect_probability += 0.1
        
        # Solvent paint robot - check if panels are in paint stage before trying to move them
        if (self.solvent_varnish_paint_robot.panels and 
            self.solvent_varnish_paint_robot.panels[0].stage == ProcessStage.PAINT_ROBOT):
            completed_panels = self.solvent_varnish_paint_robot.get_completed_panels(self.current_time)
        else:
            completed_panels = []
        
        for panel in completed_panels:
            # Choose flash off cabinet with most space
            flash_off_options = [self.solvent_varnish_flash_off_1, self.solvent_varnish_flash_off_2]
            best_option = min(flash_off_options, key=lambda x: len(x.panels))
            
            if best_option.can_accept([panel], self.current_time):
                best_option.add_panels([panel], self.current_time)
                panel.stage = ProcessStage.FLASH_OFF
                reward += 1.0
            else:
                # Try other option
                other_option = flash_off_options[1] if best_option == flash_off_options[0] else flash_off_options[0]
                if other_option.can_accept([panel], self.current_time):
                    other_option.add_panels([panel], self.current_time)
                    panel.stage = ProcessStage.FLASH_OFF
                    reward += 1.0
                else:
                    reward -= 2.0
                    panel.defect_probability += 0.1
        
        return reward
    
    def _move_from_flash_off(self) -> float:
        """Move completed panels from flash off cabinets to ovens"""
        reward = 0.0
        
        # Water flash off to water oven
        # First identify completed panels without removing them
        flash_off_completed = []
        for panel in self.water_flash_off.panels:
            if (panel.stage == ProcessStage.FLASH_OFF and 
                self.current_time >= panel.stage_start_time + panel.stage_duration):
                flash_off_completed.append(panel)
        
        # Try to move each completed panel to water oven
        successfully_moved = []
        for panel in flash_off_completed:
            # Calculate quality based on flash off time
            actual_time = self.current_time - panel.stage_start_time
            quality_factor, defect_prob = calculate_quality_factor(actual_time, self.water_flash_off.processing_time)
            panel.quality_factor *= quality_factor
            panel.defect_probability += defect_prob
            
            if self.water_oven.can_accept([panel], self.current_time):
                self.water_oven.add_panels([panel], self.current_time)
                panel.stage = ProcessStage.OVEN
                successfully_moved.append(panel)
                reward += 1.0
            else:
                # Can't move panel - leave it in flash off for now
                reward -= 2.0
                panel.defect_probability += 0.1
        
        # Remove successfully moved panels from the flash off cabinet
        for panel in successfully_moved:
            if panel in self.water_flash_off.panels:
                self.water_flash_off.panels.remove(panel)
        
        # Solvent flash off cabinets to solvent ovens
        for flash_off in [self.solvent_varnish_flash_off_1, self.solvent_varnish_flash_off_2]:
            # First identify completed panels without removing them
            flash_off_completed = []
            for panel in flash_off.panels:
                if (panel.stage == ProcessStage.FLASH_OFF and 
                    self.current_time >= panel.stage_start_time + panel.stage_duration):
                    flash_off_completed.append(panel)
            
            # Try to move each completed panel to solvent ovens
            successfully_moved = []
            for panel in flash_off_completed:
                # Calculate quality
                actual_time = self.current_time - panel.stage_start_time
                quality_factor, defect_prob = calculate_quality_factor(actual_time, flash_off.processing_time)
                panel.quality_factor *= quality_factor
                panel.defect_probability += defect_prob
                
                # Choose oven with most space
                oven_options = [self.solvent_varnish_oven_1, self.solvent_varnish_oven_2]
                best_oven = min(oven_options, key=lambda x: len(x.panels))
                
                if best_oven.can_accept([panel], self.current_time):
                    best_oven.add_panels([panel], self.current_time)
                    panel.stage = ProcessStage.OVEN
                    successfully_moved.append(panel)
                    reward += 1.0
                else:
                    other_oven = oven_options[1] if best_oven == oven_options[0] else oven_options[0]
                    if other_oven.can_accept([panel], self.current_time):
                        other_oven.add_panels([panel], self.current_time)
                        panel.stage = ProcessStage.OVEN
                        successfully_moved.append(panel)
                        reward += 1.0
                    else:
                        # Can't move panel - leave it in flash off for now
                        reward -= 2.0
                        panel.defect_probability += 0.1
            
            # Remove successfully moved panels from the flash off cabinet
            for panel in successfully_moved:
                if panel in flash_off.panels:
                    flash_off.panels.remove(panel)
        
        return reward
    
    def _move_from_ovens(self) -> float:
        """Move completed panels from ovens"""
        reward = 0.0
        
        # Water oven to buffer zone
        all_completed = self.water_oven.get_completed_panels(self.current_time)
        completed_panels = [p for p in all_completed if p.stage == ProcessStage.OVEN]
        for panel in completed_panels:
            # Calculate quality
            actual_time = self.current_time - panel.stage_start_time
            quality_factor, defect_prob = calculate_quality_factor(actual_time, self.water_oven.processing_time)
            panel.quality_factor *= quality_factor
            panel.defect_probability += defect_prob
            
            self.buffer_zone.add_panel(panel, self.current_time)
            reward += 2.0
        
        # Solvent ovens to buffer zone - panels are ready for varnishing
        for oven in [self.solvent_varnish_oven_1, self.solvent_varnish_oven_2]:
            # First identify completed first-stage oven panels without removing them
            oven_completed = []
            for panel in oven.panels:
                if (panel.stage == ProcessStage.OVEN and 
                    self.current_time >= panel.stage_start_time + panel.stage_duration):
                    oven_completed.append(panel)
            
            # Process each completed first-stage panel
            successfully_moved = []
            for panel in oven_completed:
                # Calculate quality
                actual_time = self.current_time - panel.stage_start_time
                quality_factor, defect_prob = calculate_quality_factor(actual_time, oven.processing_time)
                panel.quality_factor *= quality_factor
                panel.defect_probability += defect_prob
                
                self.buffer_zone.add_panel(panel, self.current_time)
                successfully_moved.append(panel)
                reward += 2.0
            
            # Remove successfully moved panels from the oven
            for panel in successfully_moved:
                if panel in oven.panels:
                    oven.panels.remove(panel)
        
        return reward
    
    def _move_from_varnish_robot(self) -> float:
        """Move completed panels from varnish robot (shared solvent robot) to flash off cabinets"""
        reward = 0.0
        
        # First, identify completed varnish panels (but don't remove them yet)
        varnish_completed = []
        for panel in self.solvent_varnish_paint_robot.panels:
            if (panel.stage == ProcessStage.VARNISH_ROBOT and 
                self.current_time >= panel.stage_start_time + panel.stage_duration):
                varnish_completed.append(panel)
        
        # Try to move each completed panel to flash off cabinets
        successfully_moved = []
        for panel in varnish_completed:
            # Choose flash off cabinet with most space (shared solvent/varnish equipment)
            flash_off_options = [self.solvent_varnish_flash_off_1, self.solvent_varnish_flash_off_2]
            best_option = min(flash_off_options, key=lambda x: len(x.panels))
            
            if best_option.can_accept([panel], self.current_time):
                best_option.add_panels([panel], self.current_time)
                panel.stage = ProcessStage.VARNISH_FLASH_OFF
                successfully_moved.append(panel)
                reward += 1.0
            else:
                other_option = flash_off_options[1] if best_option == flash_off_options[0] else flash_off_options[0]
                if other_option.can_accept([panel], self.current_time):
                    other_option.add_panels([panel], self.current_time)
                    panel.stage = ProcessStage.VARNISH_FLASH_OFF
                    successfully_moved.append(panel)
                    reward += 1.0
                else:
                    # Can't move panel - leave it in the robot for now
                    reward -= 2.0
                    panel.defect_probability += 0.1
        
        # Remove successfully moved panels from the robot
        for panel in successfully_moved:
            if panel in self.solvent_varnish_paint_robot.panels:
                self.solvent_varnish_paint_robot.panels.remove(panel)
        
        # Update robot busy state if empty
        if not self.solvent_varnish_paint_robot.panels:
            self.solvent_varnish_paint_robot.busy_until = 0.0
        
        return reward
    
    def _move_from_varnish_flash_off(self) -> float:
        """Move completed panels from varnish flash off (shared) to varnish ovens (shared)"""
        reward = 0.0
        
        # Use shared solvent flash off cabinets for varnish stage
        for flash_off in [self.solvent_varnish_flash_off_1, self.solvent_varnish_flash_off_2]:
            # First identify completed varnish panels without removing them
            varnish_completed = []
            for panel in flash_off.panels:
                if (panel.stage == ProcessStage.VARNISH_FLASH_OFF and 
                    self.current_time >= panel.stage_start_time + panel.stage_duration):
                    varnish_completed.append(panel)
            
            # Try to move each completed panel to varnish ovens
            successfully_moved = []
            for panel in varnish_completed:
                # Calculate quality
                actual_time = self.current_time - panel.stage_start_time
                quality_factor, defect_prob = calculate_quality_factor(actual_time, flash_off.processing_time)
                panel.quality_factor *= quality_factor
                panel.defect_probability += defect_prob
                
                # Choose oven with most space (shared solvent/varnish ovens)
                oven_options = [self.solvent_varnish_oven_1, self.solvent_varnish_oven_2]
                best_oven = min(oven_options, key=lambda x: len(x.panels))
                
                if best_oven.can_accept([panel], self.current_time):
                    best_oven.add_panels([panel], self.current_time)
                    panel.stage = ProcessStage.VARNISH_OVEN
                    successfully_moved.append(panel)
                    reward += 1.0
                else:
                    other_oven = oven_options[1] if best_oven == oven_options[0] else oven_options[0]
                    if other_oven.can_accept([panel], self.current_time):
                        other_oven.add_panels([panel], self.current_time)
                        panel.stage = ProcessStage.VARNISH_OVEN
                        successfully_moved.append(panel)
                        reward += 1.0
                    else:
                        # Can't move panel - leave it in flash off for now
                        reward -= 2.0
                        panel.defect_probability += 0.1
            
            # Remove successfully moved panels from the flash off cabinet
            for panel in successfully_moved:
                if panel in flash_off.panels:
                    flash_off.panels.remove(panel)
        
        return reward
    
    def _move_from_varnish_ovens(self) -> float:
        """Move completed panels from varnish ovens (shared) to completion"""
        reward = 0.0
        
        # Use shared solvent ovens for varnish stage
        for oven in [self.solvent_varnish_oven_1, self.solvent_varnish_oven_2]:
            # First identify completed varnish panels without removing them
            varnish_completed = []
            for panel in oven.panels:
                if (panel.stage == ProcessStage.VARNISH_OVEN and 
                    self.current_time >= panel.stage_start_time + panel.stage_duration):
                    varnish_completed.append(panel)
            
            # Process each completed varnish panel
            successfully_processed = []
            for panel in varnish_completed:
                # Calculate final quality
                actual_time = self.current_time - panel.stage_start_time
                quality_factor, defect_prob = calculate_quality_factor(actual_time, oven.processing_time)
                panel.quality_factor *= quality_factor
                panel.defect_probability += defect_prob
                
                # Determine if panel is defective
                if np.random.random() < panel.defect_probability:
                    self.defective_panels.append(panel)
                    self.total_panels_defective += 1
                    reward -= 5.0  # Penalty for defective panel
                else:
                    self.completed_panels.append(panel)
                    self.total_panels_completed += 1
                    self.total_quality_score += panel.quality_factor
                    reward += 10.0 * panel.quality_factor  # Reward based on quality
                
                panel.stage = ProcessStage.COMPLETE
                successfully_processed.append(panel)
            
            # Remove successfully processed panels from the oven
            for panel in successfully_processed:
                if panel in oven.panels:
                    oven.panels.remove(panel)
        
        return reward
    
    def _calculate_additional_rewards(self) -> float:
        """Calculate additional rewards based on system performance"""
        reward = 0.0
        
        # Penalty for keeping orders waiting too long
        for order in self.pending_orders:
            wait_time = self.current_time - order.arrival_time
            if wait_time > 60:  # More than 1 hour
                reward -= 0.1 * (wait_time - 60)
        
        # Penalty for panels waiting too long in buffer
        for panel in self.buffer_zone.panels:
            wait_time = self.current_time - panel.stage_start_time
            if wait_time > 5:  # More than 5 minutes
                reward -= 1.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the environment state"""
        obs = []
        
        # Current time (normalized)
        obs.append(self.current_time / self.shift_duration)
        
        # Equipment status (only unique equipment pieces)
        equipment_list = [
            self.water_paint_robot, self.solvent_varnish_paint_robot,
            self.water_flash_off, self.solvent_varnish_flash_off_1, self.solvent_varnish_flash_off_2,
            self.water_oven, self.solvent_varnish_oven_1, self.solvent_varnish_oven_2
        ]
        
        for equipment in equipment_list:
            obs.append(max(0, equipment.busy_until - self.current_time) / 60.0)  # Normalized remaining busy time
            obs.append(equipment.capacity / 12.0)  # Normalized capacity
            obs.append(len(equipment.panels) / equipment.capacity)  # Current load ratio
        
        # Pending orders (limited to max_pending_orders)
        orders_list = list(self.pending_orders)[:self.max_pending_orders]
        for i in range(self.max_pending_orders):
            if i < len(orders_list):
                order = orders_list[i]
                obs.append(1.0 if order.paint_type == PaintType.WATER else 0.0)
                obs.append(len(order.panels) / 3.0)  # Normalized number of panels
                obs.append(min(1.0, (self.current_time - order.arrival_time) / 120.0))  # Normalized wait time
                obs.append(order.priority)
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # Padding for empty order slots
        
        # Buffer zone info
        obs.append(len(self.buffer_zone.panels) / 20.0)  # Normalized number of panels in buffer
        if self.buffer_zone.panels:
            avg_wait = np.mean([self.current_time - p.stage_start_time for p in self.buffer_zone.panels])
            obs.append(min(1.0, avg_wait / 10.0))  # Normalized average wait time
            max_wait = max([self.current_time - p.stage_start_time for p in self.buffer_zone.panels])
            obs.append(min(1.0, max_wait / 10.0))  # Normalized max wait time
        else:
            obs.extend([0.0, 0.0])
        
        # Complete orders in buffer zone (ready for varnishing)
        complete_orders = self._get_complete_orders_in_buffer()[:self.max_buffer_orders]
        for i in range(self.max_buffer_orders):
            if i < len(complete_orders):
                order_id, panels = complete_orders[i]
                obs.append(len(panels) / 3.0)  # Normalized number of panels in order
                avg_wait = np.mean([self.current_time - p.stage_start_time for p in panels])
                obs.append(min(1.0, avg_wait / 10.0))  # Normalized average wait time for this order
                # Determine paint type (all panels in order have same paint type)
                obs.append(1.0 if panels[0].paint_type == PaintType.WATER else 0.0)
            else:
                obs.extend([0.0, 0.0, 0.0])  # Padding for empty buffer order slots
        
        # Quality metrics
        total_processed = self.total_panels_completed + self.total_panels_defective
        completion_rate = self.total_panels_completed / max(1, total_processed)
        defect_rate = self.total_panels_defective / max(1, total_processed)
        avg_quality = self.total_quality_score / max(1, self.total_panels_completed)
        throughput = total_processed / max(1, self.current_time / 60.0)  # Panels per hour
        
        obs.extend([completion_rate, defect_rate, avg_quality, min(1.0, throughput / 20.0)])
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Paint Booth Status (Time: {self.current_time:.1f} min) ===")
            print(f"Pending Orders: {len(self.pending_orders)}")
            print(f"Completed Panels: {self.total_panels_completed}")
            print(f"Defective Panels: {self.total_panels_defective}")
            
            if self.total_panels_completed > 0:
                avg_quality = self.total_quality_score / self.total_panels_completed
                print(f"Average Quality: {avg_quality:.2f}")
            
            print(f"Buffer Zone: {len(self.buffer_zone.panels)} panels")
            
            print("\nEquipment Status:")
            equipment_list = [
                ("Water Paint Robot", self.water_paint_robot),
                ("Solvent/Varnish Robot", self.solvent_varnish_paint_robot),
                ("Water Flash Off", self.water_flash_off),
                ("Solvent/Varnish Flash Off 1", self.solvent_varnish_flash_off_1),
                ("Solvent/Varnish Flash Off 2", self.solvent_varnish_flash_off_2),
                ("Water Oven", self.water_oven),
                ("Solvent/Varnish Oven 1", self.solvent_varnish_oven_1),
                ("Solvent/Varnish Oven 2", self.solvent_varnish_oven_2),
            ]
            
            for name, equipment in equipment_list:
                busy_time = max(0, equipment.busy_until - self.current_time)
                print(f"  {name}: {len(equipment.panels)}/{equipment.capacity} panels, "
                      f"busy for {busy_time:.1f} min")
    
    def close(self):
        """Clean up the environment"""
        pass
