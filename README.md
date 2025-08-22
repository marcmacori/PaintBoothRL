# Paint Booth Scheduling Environment

A comprehensive OpenAI Gym environment for optimizing paint manufacturing scheduling using reinforcement learning.

## Problem Statement

Develop a reinforcement learning (RL) agent to optimize the scheduling and resource allocation of paint panel batches across multi-stage processing equipment to minimize total processing time, equipment idle time, and defect rates. The environment models a realistic paint manufacturing process with time-dependent quality constraints, equipment capacity limitations, and dynamic order generation patterns.

## Business Impact Goal

Efficient batch scheduling reduces overall production cycle time and maximizes equipment utilization, leading to increased throughput and reduced operational costs. By minimizing idle times, bottlenecks, and defect rates, the RL-driven scheduling system enables faster order fulfillment and improved plant capacity without additional capital investment. The dynamic scheduling policies provide actionable insights for process improvements, better resource planning, and scalability to handle varying production demands.

## Process Overview

The paint booth environment simulates a complete paint manufacturing process with the following stages:

1. **Order Generation**: Orders arrive following Poisson distribution with peaks at shift start, 3.5 hours, and 6.5 hours
2. **Paint Stage**: Dedicated robots for water-based and solvent-based paints (15 min, max 3 panels per order)
3. **Flash Off**: Segregated cabinets by paint type (10 min, 12 panel capacity each)
4. **Oven Stage 1**: Initial curing (water: 1 oven, solvent: 2 ovens, 9 panel capacity each)
5. **Buffer Zone**: Temporary storage before varnishing (max 5 min, quality degrades over time)
6. **Varnish Stage**: All panels must be varnished using solvent equipment
7. **Final Curing**: Complete the manufacturing process

## Key Features

### Equipment Configuration
- **Paint Robots**: 1 water, 1 solvent/varnish (shared) (15 min processing, 3 panel capacity)
  - *Batch processors*: Must be empty to accept new orders, become busy during processing
- **Flash Off Cabinets**: 1 water, 2 solvent/varnish (shared) (10 min processing, 12 panel capacity)
  - *Continuous processors*: Can accept new panels anytime up to capacity
- **Ovens**: 1 water, 2 solvent/varnish (shared) (9 panel capacity)
  - *Continuous processors*: Can accept new panels anytime up to capacity
- **Buffer Zone**: 5-minute maximum wait time with quality degradation

### Quality System
- **Time-dependent quality**: Panels processed too quickly or too slowly have higher defect probability
- **Optimal timing**: Slight over-processing (up to 10% extra time) can improve quality
- **Buffer degradation**: Extended buffer time increases defect probability
- **Quality scoring**: Completed panels receive quality scores affecting rewards

### Order Characteristics
- **Dynamic generation**: Poisson distribution with shift-based peaks
- **Paint types**: Water-based or solvent-based (randomly assigned)
- **Panel quantities**: 1-3 panels per order
- **Priority system**: Orders accumulate priority over time

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd captone_project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from env import PaintBoothEnv

# Create environment
env = PaintBoothEnv(shift_duration=8.0, time_step=1.0)

# Reset environment
obs = env.reset()

# Take actions
action = [1, 0]  # Process first order, default equipment choice
obs, reward, done, info = env.step(action)

# Render current state
env.render()
```

## Environment Details

### Action Space
- **MultiDiscrete([51, 21])**: 
  - First element: Order selection (0 = no action, 1-50 = order index)
  - Second element: Buffer action (0 = no action, 1-20 = move complete order from buffer to varnish)
  
**Note**: Equipment selection is automatic based on paint type and equipment availability.

### Observation Space
- **Box(0.0, 1.0, (232,))**: Normalized observations including:
  - Current time
  - Equipment status (8 unique pieces × 3 features each)
  - Pending orders (50 max × 4 features each)
  - Buffer zone status (3 features)
  - Quality metrics (4 features)

### Reward Structure
- **+10**: Successfully processing an order
- **+5**: Quick processing bonus (< 30 min wait)
- **+1-2**: Moving panels between stages
- **+10 × quality**: Completing panels (quality-weighted)
- **-1**: Invalid actions
- **-2**: Equipment bottlenecks
- **-5**: Defective panels
- **-0.1 × excess_time**: Long waiting penalties

## Usage Examples

### Random Agent
```python
env = PaintBoothEnv()
obs = env.reset()

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if step % 30 == 0:  # Render every 30 minutes
        env.render()
```

### Simple Heuristic Agent
```python
env = PaintBoothEnv()
obs = env.reset()

while not done:
    # Always process first available order
    if len(env.pending_orders) > 0:
        action = [1, 0]
    else:
        action = [0, 0]
    
    obs, reward, done, info = env.step(action)
```

## Performance Metrics

The environment tracks several key performance indicators:

- **Throughput**: Panels completed per hour
- **Quality Rate**: Percentage of non-defective panels
- **Equipment Utilization**: Percentage of time each piece of equipment is busy
- **Average Quality Score**: Mean quality factor of completed panels
- **Order Wait Time**: Time from order arrival to processing
- **Buffer Utilization**: Time panels spend in buffer zone

## Advanced Features

### Order Generation Patterns
- Configurable shift duration and peak times
- Poisson distribution with time-varying rates
- Realistic paint type and panel quantity distributions

### Quality Modeling
- Non-linear quality functions based on processing time deviations
- Cumulative quality degradation through multiple stages
- Stochastic defect determination

### Equipment Constraints
- Realistic capacity limitations
- Processing time requirements
- Equipment-specific paint type restrictions
- **Batch vs Continuous Processing**:
  - *Batch processors* (Paint Robots): Must complete current job before accepting new work
  - *Continuous processors* (Flash Off, Ovens): Can accept new panels while processing others

## File Structure

```
captone_project/
├── env/
│   ├── __init__.py
│   └── paint_booth_env.py      # Main environment implementation
├── example_usage.py            # Usage examples and demos
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Key Classes

- **PaintBoothEnv**: Main gym environment
- **OrderGenerator**: Generates orders with realistic patterns
- **Equipment**: Base class for all processing equipment
- **BatchProcessor**: Equipment that processes in batches (becomes busy)
- **ContinuousProcessor**: Equipment that accepts panels continuously
- **PaintRobot**: Specialized robot for painting operations (batch processor)
- **FlashOffCabinet**: Flash off processing equipment (continuous processor)
- **Oven**: Curing oven equipment (continuous processor)
- **BufferZone**: Temporary storage with quality degradation

## Future Enhancements

- **Machine Failures**: Random equipment downtime
- **Multithreading**: Parallel simulation for faster training
- **Advanced Scheduling**: Priority-based order processing
- **Maintenance Windows**: Scheduled equipment maintenance
- **Energy Optimization**: Power consumption modeling
- **Real-time Adaptation**: Dynamic parameter adjustment
- **Add more machines**: Allow for easy tweaking of machinery number
- **Observation space based on machines**: Allow for above point to work
- **Machine Selection when Multiple are available**
- **REVIEW README**

## License

This project is part of the PCML AI Imperial capstone project.

## Contact

This will not be maintaned it is a on time project.