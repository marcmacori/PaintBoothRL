import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PaintBatchSchedulingEnv(gym.Env):
    """
    A Gymnasium environment for paint panel batch scheduling optimization.
    
    This environment simulates a paint shop with multiple stages (paint, flash-off, oven)
    and different paint types (water-based, solvent-based). The goal is to efficiently
    schedule paint panel batches through the production pipeline to maximize throughput
    and minimize delays.
    
    The environment supports:
    - Dynamic order arrivals throughout the shift
    - Variable batch sizes
    - Resource capacity constraints
    - Multi-stage processing pipeline

    Process follows:
    1. Paint
    2. Flash-off
    3. Oven
    4. Varnish
    5. Varnish Flash-off
    6. Varnish Oven
    7. Done

    Solvent and varnish share resources, water is exclusive.
    
    Paint, if paint type is solvent, and varnish are done in the solvent robots.
    Flash-off, if paint type is solvent, and varnish flash-off are done in the solvent flash cabinets.
    Oven, if paint type is solvent, and varnish oven are done in the solvent ovens.
    
    Paint, if paint type is water is exclusive and done in the water robots.
    Flash-off, if paint type is water is exclusive and done in the water flash cabinets.
    Oven, if paint type is water is exclusive and done in the water ovens.
    
    """
    
    def __init__(self):
        """
        Initialize the paint panel scheduling environment.
        
        Sets up time parameters, resource capacities, processing times,
        order generation parameters, and observation/action spaces.
        """
        super().__init__()

        # Time settings
        self.step_duration_min = 5  # each env step = 5 minutes
        self.shift_duration_hours = 8
        self.max_steps = int(self.shift_duration_hours * 60 / self.step_duration_min) 

        # Resources and capacities per unit
        self.num_water_paint_robots = 4
        self.num_solvent_paint_robots = 4
        self.robots_panel_capacity = 3  # max panels per robot at once (same order)

        self.num_water_flash_cabinets = 4
        self.num_solvent_flash_cabinets = 4
        self.cabinet_panel_capacity = 12

        self.num_water_ovens = 4
        self.num_solvent_ovens = 4
        self.oven_panel_capacity = 9

        #TODO: Add flexible flash off time + 5 minutes
        # Base processing times per panel (minutes)
        self.paint_time_per_panel_min = 20
        self.flash_off_time_per_panel_min = 10
        self.oven_time_per_panel_min = 35
        self.varnish_paint_time_per_panel_min = 20 
        self.varnish_flash_off_time_per_panel_min = 10
        self.varnish_oven_time_per_panel_min = 35 

        # Convert base times to steps
        self.paint_time_per_panel = int(self.paint_time_per_panel_min / self.step_duration_min) or 1
        self.flash_off_time_per_panel = int(self.flash_off_time_per_panel_min / self.step_duration_min) or 1
        self.oven_time_per_panel = int(self.oven_time_per_panel_min / self.step_duration_min) or 1
        self.varnish_paint_time_per_panel = int(self.varnish_paint_time_per_panel_min / self.step_duration_min) or 1
        self.varnish_flash_off_time_per_panel = int(self.varnish_flash_off_time_per_panel_min / self.step_duration_min) or 1
        self.varnish_oven_time_per_panel = int(self.varnish_oven_time_per_panel_min / self.step_duration_min) or 1

        # Order arrival simulation parameters - supports random order sizes and flexible arrival times
        self.arrival_lambda = 1.0  # Avg orders per timestep
        self.avg_panels_per_batch = 1.5
        self.min_panels_per_batch = 1
        self.max_panels_per_batch = 3  # Allow variable batch sizes
        
        # Allow arrivals throughout the entire shift, not just early timesteps
        self.arrival_window_start = 0
        self.arrival_window_end = self.max_steps - int(120 / self.step_duration_min)    # Can arrive anytime during the shift minus 2 hours

        # Order list: dynamic list of dicts for batches
        self.orders = []

        # Initialize resource units as lists of dicts to track capacity and assignments
        # Each unit tracks: currently assigned batch, panel count assigned, timer remaining
        self.paint_robots = [{'type': 'water', 'batch': None, 'panels': 0, 'timer': 0} for _ in range(self.num_water_paint_robots)] + \
                            [{'type': 'solvent', 'batch': None, 'panels': 0, 'timer': 0} for _ in range(self.num_solvent_paint_robots)]
        self.flash_cabinets = [{'type': 'water', 'batch': None, 'panels': 0, 'timer': 0} for _ in range(self.num_water_flash_cabinets)] + \
                              [{'type': 'solvent', 'batch': None, 'panels': 0, 'timer': 0} for _ in range(self.num_solvent_flash_cabinets)]
        self.ovens = [{'type': 'water', 'batch': None, 'panels': 0, 'timer': 0} for _ in range(self.num_water_ovens)] + \
                     [{'type': 'solvent', 'batch': None, 'panels': 0, 'timer': 0} for _ in range(self.num_solvent_ovens)]

        # Update resource lists in _reset_resources
        for res_list in [self.paint_robots, self.flash_cabinets, self.ovens]:
            for unit in res_list:
                unit['batch'] = None
                unit['panels'] = 0
                unit['timer'] = 0

        # Internal timestep
        self.current_step = 0

        # Dynamic maximum orders based on expected total orders for the shift
        self.max_orders = 200  # Increased to handle more orders

        # Define observation and action spaces (to be updated per step)
        self.action_space = None
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_orders * 12,), dtype=np.float32)

    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            np.ndarray: Initial observation of the environment state
        """
        self.current_step = 0
        self.orders = []
        self._generate_new_orders(initial=True)
        self._reset_resources()
        self._update_action_space()
        return self._get_obs()

    def _reset_resources(self):
        """
        Reset all resources (paint robots, flash cabinets, ovens) to their free state.
        """
        # Reset all resources to free
        for res_list in [self.paint_robots, self.flash_cabinets, self.ovens]:
            for unit in res_list:
                unit['batch'] = None
                unit['panels'] = 0
                unit['timer'] = 0

    def _generate_new_orders(self, initial=False):
        """
        Generate new orders arriving at the current timestep using a Poisson process.
        
        Args:
            initial (bool): If True, generates more orders for initial state setup
        """
        # Generate new orders arriving this step via Poisson process
        if initial:
            # Generate some initial orders to start with
            num_new = np.random.poisson(self.arrival_lambda * 3)  # More initial orders
        else:
            # Continue generating orders throughout the shift
            num_new = np.random.poisson(self.arrival_lambda)

        for _ in range(num_new):
            paint_type = np.random.choice([0,1])  # 0=water,1=solvent
            
            # Random order sizes between min and max
            panels = np.random.randint(self.min_panels_per_batch, self.max_panels_per_batch + 1)

            order = {
                'arrival_time': self.current_step,
                'paint_type': paint_type,
                'num_panels': panels,
                'current_stage': 0,
                'panels_remaining': panels,
                'assigned_units': {},  # resource type -> list of (unit_id, panels, timer)
            }

            self.orders.append(order)

    def _update_action_space(self):
        """
        Update the action space based on currently waiting orders.
        
        The action space is dynamic and includes one action for each waiting order
        plus a no-op action.
        """
        # Handle dynamic action space based on current waiting orders
        waiting_orders = [i for i, o in enumerate(self.orders) if o['current_stage'] == 0 and o['arrival_time'] <= self.current_step]
        self.waiting_orders = waiting_orders
        self.action_space = spaces.Discrete(len(waiting_orders) + 1)  # last for no-op

    def _get_obs(self):
        """
        Generate the current observation of the environment state.
        
        Returns:
            np.ndarray: Flattened observation array containing order information
                       including arrival status, paint type, panel count, stage,
                       remaining panels, and timer information
        """
        # Streamline observation space by reducing features
        obs = np.zeros((self.max_orders, 6), dtype=np.float32)

        for i, order in enumerate(self.orders[:self.max_orders]):
            arrived = 1 if order['arrival_time'] <= self.current_step else 0
            obs[i, 0] = arrived
            obs[i, 1] = order['paint_type']
            obs[i, 2] = order['num_panels'] / self.max_panels_per_batch  # Normalize panel count
            obs[i, 3] = order['current_stage'] / 4  # normalized stage
            obs[i, 4] = order['panels_remaining'] / max(order['num_panels'], 1)

            # Sum timers of assigned units (normalized)
            total_timer = 0
            if order['assigned_units']:
                for units in order['assigned_units'].values():
                    for unit in units:
                        total_timer += unit[2]
            obs[i, 5] = total_timer / 60  # normalized assuming max timer ~60 steps

        return obs.flatten()

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to take - index of waiting order to assign to paint robot,
                         or no-op if action >= number of waiting orders
        
        Returns:
            tuple: (observation, reward, done, info)
                - observation (np.ndarray): New state observation
                - reward (float): Reward for this step
                - done (bool): Whether the episode is finished
                - info (dict): Additional information
        """
        reward = 0
        done = False

        # Generate new orders throughout the shift
        self._generate_new_orders()

        # Update order status: move orders from waiting to paint stage based on arrival
        for order in self.orders:
            if order['current_stage'] == 0 and order['arrival_time'] <= self.current_step:
                # Order is waiting to be painted
                pass

        # Decode action: assign waiting order to an available paint robot
        if action < len(self.waiting_orders):
            order_idx = self.waiting_orders[action]
            order = self.orders[order_idx]
            if order['current_stage'] == 0 and order['arrival_time'] <= self.current_step:
                self._assign_paint_robot(order)

        # Update all resources and orders processing timers
        self._update_resources_and_orders()

        # Optimize reward function to focus on completed orders
        completed_orders = sum(1 for o in self.orders if o['current_stage'] == 4)
        total_orders = len([o for o in self.orders if o['arrival_time'] <= self.current_step])
        reward = completed_orders - total_orders * 0.05  # Reward completion, smaller penalty for total load

        # Check if shift ended or all arrived orders are done
        arrived_orders = [o for o in self.orders if o['arrival_time'] <= self.current_step]
        done = (self.current_step >= self.max_steps or 
                (len(arrived_orders) > 0 and all(o['current_stage'] == 7 for o in arrived_orders)))

        self.current_step += 1
        self._update_action_space()

        return self._get_obs(), reward, done, {}

    def _assign_paint_robot(self, order):
        """
        Assign an order to available paint robots of the matching paint type.
        
        Args:
            order (dict): Order to assign to paint robots
        """
        #  Better assignment logic with proper resource tracking
        robots = [robot for robot in self.paint_robots if robot['type'] == ('water' if order['paint_type'] == 0 else 'solvent')]
        panels_to_assign = order['panels_remaining']

        if panels_to_assign == 0:
            order['current_stage'] += 1
            return

        assigned_panels = 0
        order['assigned_units']['paint'] = []
        
        for i, robot in enumerate(robots):
            if robot['batch'] is None:
                capacity = self.robots_panel_capacity
                assign = min(panels_to_assign - assigned_panels, capacity)
                if assign > 0:
                    robot['batch'] = order
                    robot['panels'] = assign
                    robot['timer'] = assign * self.paint_time_per_panel
                    order['assigned_units']['paint'].append((i, assign, robot['timer']))
                    assigned_panels += assign
                    if assigned_panels >= panels_to_assign:
                        break

        order['panels_remaining'] -= assigned_panels
        if assigned_panels > 0:
            order['current_stage'] = 1  # painting

    def _update_resources_and_orders(self):
        """
        Update all resource timers and advance orders through processing stages.
        """
        # Progress timers for all assigned units, free resources when timers expire, move batches through stages
        self._process_resource_list(self.paint_robots, 'paint')
        self._process_resource_list(self.flash_cabinets, 'flash')
        self._process_resource_list(self.ovens, 'oven')

    def _process_resource_list(self, resource_list, rtype):
        """
        Process a list of resources, updating timers and freeing completed units.
        
        Args:
            resource_list (list): List of resource units to process
            rtype (str): Resource type ('paint', 'flash', or 'oven')
        """
        for unit in resource_list:
            if unit['batch'] is not None:
                unit['timer'] -= 1
                if unit['timer'] <= 0:
                    # Free resource capacity
                    order = unit['batch']
                    panels_finished = unit['panels']
                    unit['batch'] = None
                    unit['panels'] = 0
                    unit['timer'] = 0
                    self._advance_order_stage(order, rtype, unit['type'], panels_finished)

    def _advance_order_stage(self, order, rtype, paint_type, panels_finished):
        """
        Advance an order to the next processing stage after completing current stage.
        
        Args:
            order (dict): Order to advance
            rtype (str): Current resource type that finished processing
            paint_type (str): Paint type ('water' or 'solvent')
            panels_finished (int): Number of panels that finished processing
        """
        # Update order based on finished stage and assign next resource(s)
        if rtype == 'paint':
            if order['current_stage'] == 1:
                # Assign panels to flash-off cabinets of matching paint type
                self._assign_flash_cabinets(order, paint_type, panels_finished)

        elif rtype == 'flash':
            if order['current_stage'] == 2:
                self._assign_ovens(order, paint_type, panels_finished)

        elif rtype == 'oven':
            if order['current_stage'] == 3:
                order['current_stage'] = 4  # done
                order['panels_remaining'] += panels_finished

    def _assign_flash_cabinets(self, order, paint_type, panels_to_assign):
        """
        Assign panels to flash-off cabinets of the matching paint type.
        
        Args:
            order (dict): Order to assign
            paint_type (str): Paint type ('water' or 'solvent')
            panels_to_assign (int): Number of panels to assign
        """
        cabinets = [unit for unit in self.flash_cabinets if unit['type'] == ('water' if paint_type == 0 else 'solvent')]
        self._assign_to_resource_units(order, cabinets, panels_to_assign, self.flash_off_time_per_panel)
        order['current_stage'] = 2

    def _assign_ovens(self, order, paint_type, panels_to_assign):
        """
        Assign panels to ovens of the matching paint type.
        
        Args:
            order (dict): Order to assign
            paint_type (str): Paint type ('water' or 'solvent')
            panels_to_assign (int): Number of panels to assign
        """
        ovens = [unit for unit in self.ovens if unit['type'] == ('water' if paint_type == 0 else 'solvent')]
        self._assign_to_resource_units(order, ovens, panels_to_assign, self.oven_time_per_panel)
        order['current_stage'] = 3

    def _assign_to_resource_units(self, order, resource_units, panels_to_assign, time_per_panel):
        """
        Generic method to assign panels to available resource units.
        
        Args:
            order (dict): Order to assign
            resource_units (list): List of available resource units
            panels_to_assign (int): Number of panels to assign
            time_per_panel (int): Processing time per panel for this resource type
        """
        assigned_panels = 0
        if 'assigned_units_stage' not in order:
            order['assigned_units_stage'] = {}

        if order['current_stage'] not in order['assigned_units_stage']:
            order['assigned_units_stage'][order['current_stage']] = []

        for i, unit in enumerate(resource_units):
            if unit['batch'] is None:
                capacity = self.cabinet_panel_capacity if unit['type'] == 'flash' else self.oven_panel_capacity
                assign = min(panels_to_assign - assigned_panels, capacity)
                if assign > 0:
                    unit['batch'] = order
                    unit['panels'] = assign
                    unit['timer'] = assign * time_per_panel
                    order['assigned_units_stage'][order['current_stage']].append((i, assign, unit['timer']))
                    assigned_panels += assign
                    if assigned_panels >= panels_to_assign:
                        break
        order['panels_remaining'] -= assigned_panels

    def render(self):
        """
        Render the current state of the environment for debugging and visualization.
        
        Prints information about current timestep, orders, and resource utilization.
        """
        print(f"\nStep {self.current_step} / {self.max_steps}")
        print(f"Orders in system: {len(self.orders)}")
        for i, order in enumerate(self.orders):
            print(f"Order {i}: Arrival {order['arrival_time']}, Paint Type {order['paint_type']}," 
                  f" Panels {order['num_panels']}, Stage {order['current_stage']}, Panels Remaining {order['panels_remaining']}")
        print(f"Paint Robots: {self.paint_robots}")
        print(f"Flash Cabinets: {self.flash_cabinets}")
        print(f"Ovens: {self.ovens}")
if __name__ == "__main__":

    # Create and test the environment
    env = PaintBatchSchedulingEnv()

    print(env.flash_cabinets)