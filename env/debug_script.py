#!/usr/bin/env python3
"""
Debug Script for Paint Panel Scheduling Environment

This script provides comprehensive debugging and testing capabilities for the
PaintBatchSchedulingEnv environment. It includes:

1. Environment initialization and basic functionality tests
2. Resource allocation validation
3. Order processing simulation
4. Performance metrics calculation
5. Visualization of environment state
6. Stress testing with various scenarios

Usage:
    python debug_script.py --verbose --test-all --stress-test
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import traceback

# Add the env directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'env'))

try:
    from paint_panels_env import PaintBatchSchedulingEnv
except ImportError as e:
    print(f"Error importing environment: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)


class EnvironmentDebugger:
    """Debug utility class for the Paint Panel Scheduling Environment."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.env = None
        self.test_results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with optional verbosity control."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")
    
    def initialize_environment(self) -> bool:
        """Initialize the environment and perform basic validation."""
        try:
            self.log("Initializing PaintBatchSchedulingEnv...")
            self.env = PaintBatchSchedulingEnv()
            
            # Basic environment validation
            self.log("Validating environment properties...")
            
            # Check time settings
            assert self.env.step_duration_min == 5, "Step duration should be 5 minutes"
            assert self.env.shift_duration_hours == 8, "Shift duration should be 8 hours"
            assert self.env.max_steps == 96, "Max steps should be 96 (8 hours * 60 min / 5 min)"
            
            # Check resource capacities
            assert self.env.num_water_paint_robots == 4, "Should have 4 water paint robots"
            assert self.env.num_solvent_paint_robots == 4, "Should have 4 solvent paint robots"
            assert self.env.robots_panel_capacity == 3, "Robot panel capacity should be 3"
            
            self.log("Environment initialization successful!")
            return True
            
        except Exception as e:
            self.log(f"Environment initialization failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def test_reset_functionality(self) -> bool:
        """Test environment reset functionality."""
        try:
            self.log("Testing environment reset...")
            
            # Reset environment
            obs = self.env.reset()
            
            # Validate observation
            assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
            assert obs.shape == (self.env.max_orders * 12,), f"Observation shape should be ({self.env.max_orders * 12},)"
            assert obs.dtype == np.float32, "Observation should be float32"
            
            # Check initial state
            assert self.env.current_step == 0, "Current step should be 0 after reset"
            assert len(self.env.orders) > 0, "Should have initial orders after reset"
            
            self.log(f"Reset successful. Initial orders: {len(self.env.orders)}")
            return True
            
        except Exception as e:
            self.log(f"Reset test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def test_action_space(self) -> bool:
        """Test action space functionality."""
        try:
            self.log("Testing action space...")
            
            # Reset to get initial state
            self.env.reset()
            
            # Check action space
            assert self.env.action_space is not None, "Action space should not be None"
            assert hasattr(self.env.action_space, 'n'), "Action space should have 'n' attribute"
            
            self.log(f"Action space size: {self.env.action_space.n}")
            self.log(f"Waiting orders: {len(self.env.waiting_orders)}")
            
            # Test valid actions
            for action in range(min(5, self.env.action_space.n)):
                try:
                    obs, reward, done, truncated, info = self.env.step(action)
                    self.log(f"Action {action} executed successfully")
                    break
                except Exception as e:
                    self.log(f"Action {action} failed: {e}", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"Action space test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def test_order_generation(self) -> bool:
        """Test order generation and properties."""
        try:
            self.log("Testing order generation...")
            
            self.env.reset()
            
            # Analyze generated orders
            water_orders = [o for o in self.env.orders if o['paint_type'] == 0]
            solvent_orders = [o for o in self.env.orders if o['paint_type'] == 1]
            
            self.log(f"Total orders: {len(self.env.orders)}")
            self.log(f"Water-based orders: {len(water_orders)}")
            self.log(f"Solvent-based orders: {len(solvent_orders)}")
            
            # Check order properties
            for order in self.env.orders:
                assert 'arrival_time' in order, "Order should have arrival_time"
                assert 'paint_type' in order, "Order should have paint_type"
                assert 'num_panels' in order, "Order should have num_panels"
                assert 'current_stage' in order, "Order should have current_stage"
                assert 'panels_remaining' in order, "Order should have panels_remaining"
                assert 'assigned_units' in order, "Order should have assigned_units"
                
                assert order['paint_type'] in [0, 1], "Paint type should be 0 or 1"
                assert 1 <= order['num_panels'] <= 3, "Panel count should be between 1 and 3"
                assert order['current_stage'] == 0, "Initial stage should be 0"
                assert order['panels_remaining'] == order['num_panels'], "Initial panels_remaining should equal num_panels"
            
            return True
            
        except Exception as e:
            self.log(f"Order generation test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def test_resource_allocation(self) -> bool:
        """Test resource allocation and management."""
        try:
            self.log("Testing resource allocation...")
            
            self.env.reset()
            
            # Check resource initialization
            assert len(self.env.paint_robots) == 8, "Should have 8 paint robots total"
            assert len(self.env.flash_cabinets) == 8, "Should have 8 flash cabinets total"
            assert len(self.env.ovens) == 8, "Should have 8 ovens total"
            
            # Check resource types
            water_robots = [r for r in self.env.paint_robots if r['type'] == 'water']
            solvent_robots = [r for r in self.env.paint_robots if r['type'] == 'solvent']
            
            assert len(water_robots) == 4, "Should have 4 water robots"
            assert len(solvent_robots) == 4, "Should have 4 solvent robots"
            
            # Check initial resource state
            for robot in self.env.paint_robots:
                assert robot['batch'] is None, "Initial robot should have no batch"
                assert robot['panels'] == 0, "Initial robot should have 0 panels"
                assert robot['timer'] == 0, "Initial robot should have 0 timer"
            
            self.log("Resource allocation test passed!")
            return True
            
        except Exception as e:
            self.log(f"Resource allocation test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def test_step_functionality(self) -> bool:
        """Test environment step functionality."""
        try:
            self.log("Testing step functionality...")
            
            self.env.reset()
            initial_orders = len(self.env.orders)
            
            # Take a few steps
            for step in range(5):
                if self.env.action_space.n > 1:
                    action = 0  # Choose first waiting order
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    self.log(f"Step {step}: reward={reward}, done={done}, orders={len(self.env.orders)}")
                    
                    # Check observation
                    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
                    assert obs.shape == (self.env.max_orders * 12,), "Observation shape should be correct"
                    
                    # Check reward
                    assert isinstance(reward, (int, float)), "Reward should be numeric"
                    
                    # Check done flag
                    assert isinstance(done, bool), "Done should be boolean"
                    
                    if done:
                        self.log("Episode completed")
                        break
                else:
                    # No waiting orders, take no-op action
                    obs, reward, done, truncated, info = self.env.step(0)
                    self.log(f"Step {step}: no-op action, reward={reward}")
            
            return True
            
        except Exception as e:
            self.log(f"Step functionality test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test and calculate performance metrics."""
        try:
            self.log("Testing performance metrics...")
            
            self.env.reset()
            
            # Run a short episode
            total_reward = 0
            steps_taken = 0
            orders_completed = 0
            
            for step in range(20):  # Run for 20 steps
                if self.env.action_space.n > 1:
                    action = 0
                else:
                    action = 0  # no-op
                
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps_taken += 1
                
                # Count completed orders
                completed = [o for o in self.env.orders if o['current_stage'] >= 7]
                orders_completed = len(completed)
                
                if done:
                    break
            
            # Calculate metrics
            avg_reward = total_reward / max(steps_taken, 1)
            throughput = orders_completed / max(steps_taken, 1)
            
            self.log(f"Performance Metrics:")
            self.log(f"  Total Reward: {total_reward}")
            self.log(f"  Average Reward: {avg_reward:.3f}")
            self.log(f"  Steps Taken: {steps_taken}")
            self.log(f"  Orders Completed: {orders_completed}")
            self.log(f"  Throughput: {throughput:.3f} orders/step")
            
            return True
            
        except Exception as e:
            self.log(f"Performance metrics test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def stress_test(self, num_episodes: int = 10) -> bool:
        """Run stress test with multiple episodes."""
        try:
            self.log(f"Running stress test with {num_episodes} episodes...")
            
            episode_rewards = []
            episode_lengths = []
            episode_throughputs = []
            
            for episode in range(num_episodes):
                self.log(f"Episode {episode + 1}/{num_episodes}")
                
                self.env.reset()
                total_reward = 0
                steps = 0
                orders_completed = 0
                
                while steps < 100:  # Limit steps per episode
                    if self.env.action_space.n > 1:
                        action = 0
                    else:
                        action = 0
                    
                    obs, reward, done, truncated, info = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    # Count completed orders
                    completed = [o for o in self.env.orders if o['current_stage'] >= 7]
                    orders_completed = len(completed)
                    
                    if done:
                        break
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                episode_throughputs.append(orders_completed / max(steps, 1))
            
            # Calculate statistics
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_throughput = np.mean(episode_throughputs)
            
            self.log(f"Stress Test Results:")
            self.log(f"  Average Reward: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
            self.log(f"  Average Episode Length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
            self.log(f"  Average Throughput: {avg_throughput:.3f} ± {np.std(episode_throughputs):.3f}")
            
            return True
            
        except Exception as e:
            self.log(f"Stress test failed: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def visualize_state(self, save_plot: bool = False):
        """Visualize current environment state."""
        try:
            self.log("Creating state visualization...")
            
            if self.env is None:
                self.log("Environment not initialized", "ERROR")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Paint Panel Scheduling Environment State (Step {self.env.current_step})')
            
            # Plot 1: Order distribution by paint type
            water_orders = [o for o in self.env.orders if o['paint_type'] == 0]
            solvent_orders = [o for o in self.env.orders if o['paint_type'] == 1]
            
            axes[0, 0].bar(['Water', 'Solvent'], [len(water_orders), len(solvent_orders)])
            axes[0, 0].set_title('Orders by Paint Type')
            axes[0, 0].set_ylabel('Number of Orders')
            
            # Plot 2: Order stages distribution
            stage_counts = {}
            for order in self.env.orders:
                stage = order['current_stage']
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
            if stage_counts:
                stages = list(stage_counts.keys())
                counts = list(stage_counts.values())
                axes[0, 1].bar(stages, counts)
                axes[0, 1].set_title('Orders by Current Stage')
                axes[0, 1].set_xlabel('Stage')
                axes[0, 1].set_ylabel('Number of Orders')
            
            # Plot 3: Resource utilization
            resource_types = ['Water Robots', 'Solvent Robots', 'Water Flash', 'Solvent Flash', 'Water Ovens', 'Solvent Ovens']
            utilization = []
            
            # Calculate utilization for each resource type
            water_robots_util = sum(1 for r in self.env.paint_robots if r['type'] == 'water' and r['batch'] is not None) / 4
            solvent_robots_util = sum(1 for r in self.env.paint_robots if r['type'] == 'solvent' and r['batch'] is not None) / 4
            water_flash_util = sum(1 for r in self.env.flash_cabinets if r['type'] == 'water' and r['batch'] is not None) / 4
            solvent_flash_util = sum(1 for r in self.env.flash_cabinets if r['type'] == 'solvent' and r['batch'] is not None) / 4
            water_ovens_util = sum(1 for r in self.env.ovens if r['type'] == 'water' and r['batch'] is not None) / 4
            solvent_ovens_util = sum(1 for r in self.env.ovens if r['type'] == 'solvent' and r['batch'] is not None) / 4
            
            utilization = [water_robots_util, solvent_robots_util, water_flash_util, 
                          solvent_flash_util, water_ovens_util, solvent_ovens_util]
            
            axes[1, 0].bar(resource_types, utilization)
            axes[1, 0].set_title('Resource Utilization')
            axes[1, 0].set_ylabel('Utilization Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Panel distribution
            panel_counts = [order['num_panels'] for order in self.env.orders]
            if panel_counts:
                axes[1, 1].hist(panel_counts, bins=[1, 2, 3, 4], alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Panel Count Distribution')
                axes[1, 1].set_xlabel('Panels per Order')
                axes[1, 1].set_ylabel('Number of Orders')
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig('environment_state.png', dpi=300, bbox_inches='tight')
                self.log("State visualization saved as 'environment_state.png'")
            else:
                plt.show()
            
        except Exception as e:
            self.log(f"Visualization failed: {e}", "ERROR")
            traceback.print_exc()
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all debug tests and return results."""
        self.log("Running comprehensive debug tests...")
        
        tests = [
            ("Environment Initialization", self.initialize_environment),
            ("Reset Functionality", self.test_reset_functionality),
            ("Action Space", self.test_action_space),
            ("Order Generation", self.test_order_generation),
            ("Resource Allocation", self.test_resource_allocation),
            ("Step Functionality", self.test_step_functionality),
            ("Performance Metrics", self.test_performance_metrics),
        ]
        
        results = {}
        for test_name, test_func in tests:
            self.log(f"\n{'='*50}")
            self.log(f"Running: {test_name}")
            self.log('='*50)
            
            try:
                result = test_func()
                results[test_name] = result
                status = "PASSED" if result else "FAILED"
                self.log(f"{test_name}: {status}")
            except Exception as e:
                self.log(f"{test_name}: FAILED - {e}", "ERROR")
                results[test_name] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary."""
        self.log("\n" + "="*60)
        self.log("DEBUG TEST SUMMARY")
        self.log("="*60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            self.log(f"{test_name}: {status}")
        
        self.log(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            self.log("All tests passed! Environment appears to be working correctly.", "INFO")
        else:
            self.log(f"{total - passed} tests failed. Please review the errors above.", "WARNING")


def main():
    """Main function to run the debug script."""
    parser = argparse.ArgumentParser(description='Debug script for Paint Panel Scheduling Environment')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    parser.add_argument('--stress-test', action='store_true', help='Run stress test')
    parser.add_argument('--visualize', action='store_true', help='Create state visualization')
    parser.add_argument('--save-plot', action='store_true', help='Save visualization to file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for stress test')
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = EnvironmentDebugger(verbose=args.verbose)
    
    try:
        if args.test_all:
            # Run all tests
            results = debugger.run_all_tests()
            debugger.print_summary(results)
        
        if args.stress_test:
            # Run stress test
            debugger.initialize_environment()
            debugger.stress_test(args.episodes)
        
        if args.visualize:
            # Create visualization
            debugger.initialize_environment()
            debugger.env.reset()
            debugger.visualize_state(save_plot=args.save_plot)
        
        # If no specific test requested, run all tests
        if not any([args.test_all, args.stress_test, args.visualize]):
            results = debugger.run_all_tests()
            debugger.print_summary(results)
    
    except KeyboardInterrupt:
        print("\nDebug script interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
