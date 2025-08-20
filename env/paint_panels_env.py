import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PaintBatchSchedulingEnv(gym.Env):
    """
    A Gymnasium environment for paint panel batch scheduling optimization.
    """

    def __init__(self):
        super().__init__()

        # Time settings
        self.step_duration_min = 1
        self.shift_duration_hours = 8
        self.max_steps = int(self.shift_duration_hours * 60 / self.step_duration_min)

        # Resources
        self.num_water_paint_robots = 4
        self.num_solvent_paint_robots = 4
        self.robots_panel_capacity = 3

        self.num_water_flash_cabinets = 4
        self.num_solvent_flash_cabinets = 4
        self.cabinet_panel_capacity = 12

        self.num_water_ovens = 4
        self.num_solvent_ovens = 4
        self.oven_panel_capacity = 9

        # Base processing times
        self.paint_time_per_panel_min = 20
        self.flash_off_time_per_panel_min = 10
        self.oven_time_per_panel_min = 35

        self.time_variation_percent = 0.10
        self.buffer_time_min = 2
        self.buffer_time_steps = int(self.buffer_time_min / self.step_duration_min) or 1

        self.timing_modes = {
            'fast': {'variation': -0.10, 'buffer_factor': 0.0, 'quality_risk': 0.05},
            'normal': {'variation': 0.0, 'buffer_factor': 0.5, 'quality_risk': 0.01},
            'careful': {'variation': 0.10, 'buffer_factor': 1.0, 'quality_risk': 0.0}
        }

        self.total_buffer_time_used = 0
        self.total_defects = 0
        self.max_buffer_penalty = 0.1
        self.defect_penalty = 0.2

        # Convert base times to steps
        self.paint_time_per_panel = int(self.paint_time_per_panel_min / self.step_duration_min) or 1
        self.flash_off_time_per_panel = int(self.flash_off_time_per_panel_min / self.step_duration_min) or 1
        self.oven_time_per_panel = int(self.oven_time_per_panel_min / self.step_duration_min) or 1

        # Order arrivals
        self.arrival_lambda = 1.0
        self.avg_panels_per_batch = 1.5
        self.min_panels_per_batch = 1
        self.max_panels_per_batch = 3

        self.arrival_window_start = 0
        self.arrival_window_end = self.max_steps - int(120 / self.step_duration_min)

        self.orders = []

        # Resource units
        self.paint_robots = [{'type': 'water', 'batch': None, 'panels': 0, 'timer': 0}
                             for _ in range(self.num_water_paint_robots)] + \
                            [{'type': 'solvent', 'batch': None, 'panels': 0, 'timer': 0}
                             for _ in range(self.num_solvent_paint_robots)]
        self.flash_cabinets = [{'type': 'water', 'batch': None, 'panels': 0, 'timer': 0}
                               for _ in range(self.num_water_flash_cabinets)] + \
                              [{'type': 'solvent', 'batch': None, 'panels': 0, 'timer': 0}
                               for _ in range(self.num_solvent_flash_cabinets)]
        self.ovens = [{'type': 'water', 'batch': None, 'panels': 0, 'timer': 0}
                      for _ in range(self.num_water_ovens)] + \
                     [{'type': 'solvent', 'batch': None, 'panels': 0, 'timer': 0}
                      for _ in range(self.num_solvent_ovens)]

        # Internal state
        self.current_step = 0
        self.max_orders = 100

        self.action_space = spaces.Discrete(1)  # will be updated dynamically
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_orders * 8,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.orders = []
        self.total_buffer_time_used = 0
        self.total_defects = 0
        self._generate_new_orders(initial=True)
        self._reset_resources()
        self._update_action_space()
        return self._get_obs(), {}

    def _reset_resources(self):
        for res_list in [self.paint_robots, self.flash_cabinets, self.ovens]:
            for unit in res_list:
                unit['batch'] = None
                unit['panels'] = 0
                unit['timer'] = 0

    def _generate_new_orders(self, initial=False):
        num_new = np.random.poisson(self.arrival_lambda * (3 if initial else 1))
        for _ in range(num_new):
            paint_type = np.random.choice([0, 1])
            panels = np.random.randint(self.min_panels_per_batch, self.max_panels_per_batch + 1)
            order = {
                'id': len(self.orders),
                'arrival_time': self.current_step,
                'paint_type': paint_type,
                'num_panels': panels,
                'current_stage': 0,
                'panels_remaining': panels,
                'assigned_units': {},
                'status': "waiting"
            }
            self.orders.append(order)

    def _calculate_flexible_time(self, base_time_steps, timing_mode='normal'):
        mode_params = self.timing_modes[timing_mode]
        variation = mode_params['variation']
        varied_time = base_time_steps * (1 + variation)
        buffer_factor = mode_params['buffer_factor']
        buffer_used = int(self.buffer_time_steps * buffer_factor)
        has_defect = np.random.random() < mode_params['quality_risk']
        total_time = int(varied_time + buffer_used)
        return max(1, total_time), buffer_used, has_defect

    def _get_base_time_for_stage(self, stage_type):
        mapping = {'paint': self.paint_time_per_panel,
                   'flash': self.flash_off_time_per_panel,
                   'oven': self.oven_time_per_panel}
        return mapping.get(stage_type, 1)

    def _get_paint_type_string(self, code):
        return 'water' if code == 0 else 'solvent'

    def _update_action_space(self):
        self.waiting_paint_orders = [i for i, o in enumerate(self.orders)
                                     if o['current_stage'] == 0 and o['arrival_time'] <= self.current_step]
        self.waiting_flash_orders = [i for i, o in enumerate(self.orders)
                                     if o['current_stage'] == 1 and o['panels_remaining'] == 0]
        self.waiting_oven_orders = [i for i, o in enumerate(self.orders)
                                    if o['current_stage'] == 2 and o['panels_remaining'] == 0]
        num_actions = (len(self.waiting_paint_orders) +
                       len(self.waiting_flash_orders) +
                       len(self.waiting_oven_orders)) * 3 + 1
        self.action_space = spaces.Discrete(num_actions)
        self.valid_actions = list(range(num_actions))

    def _get_obs(self):
        obs = np.zeros((self.max_orders, 8), dtype=np.float32)
        for i, order in enumerate(self.orders[:self.max_orders]):
            arrived = 1 if order['arrival_time'] <= self.current_step else 0
            obs[i, 0] = arrived
            obs[i, 1] = order['paint_type']
            obs[i, 2] = order['num_panels'] / self.max_panels_per_batch
            obs[i, 3] = order['current_stage'] / 4
            obs[i, 4] = order['panels_remaining'] / max(order['num_panels'], 1)
            obs[i, 5] = 1 if order['panels_remaining'] == 0 and order['current_stage'] < 4 else 0
            total_timer = 0
            if order['assigned_units']:
                for units in order['assigned_units'].values():
                    for unit in units:
                        total_timer += unit[2]
            obs[i, 6] = total_timer / 65
            obs[i, 7] = min((self.current_step - order['arrival_time']) / self.max_steps, 1) if arrived else 0
        return obs.flatten()

    def step(self, action):
        reward = 0
        self._generate_new_orders()
        self._execute_action(action)
        self._update_resources_and_orders()
        completed_orders = sum(1 for o in self.orders if o['current_stage'] == 4)
        total_orders = len([o for o in self.orders if o['arrival_time'] <= self.current_step])
        buffer_penalty = (self.total_buffer_time_used / max(self.max_steps, 1)) * self.max_buffer_penalty
        defect_penalty = self.total_defects * self.defect_penalty
        reward = completed_orders - total_orders * 0.05 - buffer_penalty - defect_penalty
        arrived_orders = [o for o in self.orders if o['arrival_time'] <= self.current_step]
        done = (self.current_step >= self.max_steps or
                (len(arrived_orders) > 0 and all(o['current_stage'] == 4 for o in arrived_orders)))
        self.current_step += 1
        self._update_action_space()
        return self._get_obs(), reward, done, {}, {}

    def _execute_action(self, action):
        if action >= self.action_space.n - 1:
            return
        action_idx = action
        timing_modes = ['fast', 'normal', 'careful']
        stage_configs = [
            ('paint', self.waiting_paint_orders, 0, self._assign_paint_robot),
            ('flash', self.waiting_flash_orders, 1,
             lambda o, t: self._assign_to_stage(o, 'flash', t)),
            ('oven', self.waiting_oven_orders, 2,
             lambda o, t: self._assign_to_stage(o, 'oven', t))
        ]
        for stage_type, waiting_orders, required_stage, assign_func in stage_configs:
            actions_count = len(waiting_orders) * 3
            if action_idx < actions_count:
                order_slot = action_idx // 3
                timing_mode = timing_modes[action_idx % 3]
                if order_slot < len(waiting_orders):
                    order = self.orders[waiting_orders[order_slot]]
                    if stage_type == 'paint':
                        if order['current_stage'] == 0 and order['arrival_time'] <= self.current_step:
                            assign_func(order, timing_mode)
                    else:
                        if order['current_stage'] == required_stage and order['panels_remaining'] == 0:
                            assign_func(order, timing_mode)
                return
            action_idx -= actions_count

    def _assign_paint_robot(self, order, timing_mode="normal"):
        paint_type = self._get_paint_type_string(order['paint_type'])
        robots = [robot for robot in self.paint_robots if robot['type'] == paint_type]
        panels_to_assign = order['num_panels']
        
        # Debug: Check if any robots are available
        available_robots = [i for i, robot in enumerate(robots) if robot['batch'] is None]
        
        for i, robot in enumerate(robots):
            if robot['batch'] is None and panels_to_assign <= self.robots_panel_capacity:
                base_time = panels_to_assign * self._get_base_time_for_stage("paint")
                flexible_time, buffer_used, has_defect = self._calculate_flexible_time(base_time, timing_mode)
                self.total_buffer_time_used += buffer_used
                if has_defect:
                    self.total_defects += 1
                robot['batch'] = order
                robot['panels'] = panels_to_assign
                robot['timer'] = flexible_time
                order['assigned_units']['paint'] = [(i, panels_to_assign, flexible_time)]
                order['current_stage'] = 1
                order['panels_remaining'] = panels_to_assign
                order['status'] = "processing"
                return
        
        # If we get here, no robot was assigned
        pass

    def _update_resources_and_orders(self):
        self._process_resource_list(self.paint_robots, 'paint')
        self._process_resource_list(self.flash_cabinets, 'flash')
        self._process_resource_list(self.ovens, 'oven')

    def _process_resource_list(self, resource_list, rtype):
        for unit in resource_list:
            if unit['batch'] is not None:
                unit['timer'] -= 1
                if unit['timer'] <= 0:
                    order = unit['batch']
                    panels_finished = unit['panels']
                    pass
                    unit['batch'] = None
                    unit['panels'] = 0
                    unit['timer'] = 0
                    self._advance_order_stage(order, rtype, unit['type'], panels_finished)

    def _advance_order_stage(self, order, rtype, paint_type, panels_finished):
        if rtype == 'paint' and order['current_stage'] == 1:
            order['panels_remaining'] = 0
            order['status'] = "waiting"
        elif rtype == 'flash' and order['current_stage'] == 2:
            order['panels_remaining'] = 0
            order['status'] = "waiting"
        elif rtype == 'oven' and order['current_stage'] == 3:
            order['current_stage'] = 4
            order['panels_remaining'] = panels_finished
            order['status'] = "completed"

    def _assign_to_resource_units(self, order, resource_units, panels_to_assign, stage_type, timing_mode='normal'):
        assigned_panels = 0
        if 'assigned_units_stage' not in order:
            order['assigned_units_stage'] = {}
        if order['current_stage'] not in order['assigned_units_stage']:
            order['assigned_units_stage'][order['current_stage']] = []
        for i, unit in enumerate(resource_units):
            if unit['batch'] is None:
                capacity = self.cabinet_panel_capacity if stage_type == 'flash' else self.oven_panel_capacity
                assign = min(panels_to_assign - assigned_panels, capacity)
                if assign > 0:
                    base_time = assign * self._get_base_time_for_stage(stage_type)
                    flexible_time, buffer_used, has_defect = self._calculate_flexible_time(base_time, timing_mode)
                    self.total_buffer_time_used += buffer_used
                    if has_defect:
                        self.total_defects += 1
                    unit['batch'] = order
                    unit['panels'] = assign
                    unit['timer'] = flexible_time
                    order['assigned_units_stage'][order['current_stage']].append((i, assign, unit['timer']))
                    assigned_panels += assign
                    pass
                    if assigned_panels >= panels_to_assign:
                        break
        order['panels_remaining'] -= assigned_panels
        pass

    def _assign_to_stage(self, order, stage_type, timing_mode='normal'):
        paint_type = self._get_paint_type_string(order['paint_type'])
        panels_to_assign = order['num_panels']
        if stage_type == 'flash':
            resource_units = [unit for unit in self.flash_cabinets if unit['type'] == paint_type]
            new_stage = 2
        elif stage_type == 'oven':
            resource_units = [unit for unit in self.ovens if unit['type'] == paint_type]
            new_stage = 3
        else:
            return
        
        # Debug: Check if any units are available
        available_units = [i for i, unit in enumerate(resource_units) if unit['batch'] is None]
        
        self._assign_to_resource_units(order, resource_units, panels_to_assign, stage_type, timing_mode)
        order['current_stage'] = new_stage
        order['panels_remaining'] = panels_to_assign
        order['status'] = "processing"

    def render(self):
        print(f"\nStep {self.current_step} / {self.max_steps}")
        print(f"Orders in system: {len(self.orders)}")
        for i, order in enumerate(self.orders):
            print(f"Order {i}: Arrival {order['arrival_time']}, Type {order['paint_type']}, "
                  f"Panels {order['num_panels']}, Stage {order['current_stage']}, Remaining {order['panels_remaining']}")


def run_fifo_simulation():
    env = PaintBatchSchedulingEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        # Choose the first available action (FIFO behavior)
        fifo_action = 0
        if env.action_space.n > 1:
            # Try to find the first valid action that assigns an order
            # Priority: flash orders first (to clear bottlenecks), then oven, then paint
            if len(env.waiting_flash_orders) > 0:
                # Choose first flash order with normal timing
                paint_actions = len(env.waiting_paint_orders) * 3
                fifo_action = 1 + paint_actions + 1  # First flash order, normal timing
            elif len(env.waiting_oven_orders) > 0:
                # Choose first oven order with normal timing
                paint_actions = len(env.waiting_paint_orders) * 3
                flash_actions = len(env.waiting_flash_orders) * 3
                fifo_action = 1 + paint_actions + flash_actions + 1  # First oven order, normal timing
            elif len(env.waiting_paint_orders) > 0:
                # Choose first paint order with normal timing (action 1 + 1 = action 2)
                fifo_action = 1 + 1  # First paint order, normal timing
        
        obs, reward, done, _, _ = env.step(fifo_action)
        total_reward += reward
        step_count += 1
        
        # Debug output every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}: Action space size: {env.action_space.n}, "
                  f"Waiting paint: {len(env.waiting_paint_orders)}, "
                  f"Waiting flash: {len(env.waiting_flash_orders)}, "
                  f"Waiting oven: {len(env.waiting_oven_orders)}")
    
    print("\nFIFO simulation finished.")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward}")
    print(f"Total Orders: {len(env.orders)}")
    print(f"Total Orders Completed: {sum(1 for o in env.orders if o['current_stage'] == 4)}")
    print(f"Defects: {env.total_defects}, Buffer Time Used: {env.total_buffer_time_used}")


if __name__ == "__main__":
    run_fifo_simulation()
