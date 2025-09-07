# PaintBoothRL

A reinforcement learning environment for dynamic oven batching with energy tariffs. This project simulates a production system where an agent must decide when to launch oven batches to balance throughput, energy costs, and lateness penalties.

## Overview

The environment simulates a paint booth production system with:
- **Ovens**: Process painted panels in batches with heating/cooling dynamics
- **Jobs**: Arrive dynamically with due dates and lateness penalties
- **Energy Tariffs**: Time-varying energy costs (peak/off-peak pricing)
- **Multi-objective Optimization**: Balance throughput, energy efficiency, and on-time delivery

## Features

- **Dynamic Job Arrivals**: Realistic shift-based arrival patterns
- **Energy Cost Optimization**: Time-of-day pricing with peak/off-peak rates
- **Oven Temperature Dynamics**: Realistic heating and cooling behavior
- **Multiple Agent Types**: Random, FIFO heuristic, and PPO reinforcement learning
- **Comprehensive Analysis**: Episode tracking, visualization, and performance metrics

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Episode Analysis

```bash
python oven_batching/training/run_episode_analysis.py
```

This will run episodes with all available agents and generate:
- Performance comparison plots
- Detailed episode analysis
- CSV data exports

### Train PPO Agent

```bash
python oven_batching/training/train_PPO_compare_agents.py
```

## Environment Details

### Actions
- **Wait**: Do nothing, advance time
- **Launch**: Start processing a batch of panels
- **Heat**: Preheat an oven to operating temperature

### Rewards
- **Completion Rewards**: Positive for on-time job completion
- **Energy Penalties**: Cost based on time-of-day tariffs
- **Lateness Penalties**: Negative for late deliveries
- **Action Validation**: Small penalties for invalid actions

### Observation Space
- Queue statistics (length, waiting times, urgency)
- Oven status (busy, heating, temperature, time to completion)
- Current energy tariff
- Time progression

## Results

Results are saved in the `results/` directory:
- `episode_analysis/`: Individual agent performance analysis
- `results_trainining_compare_script/`: Comparative performance metrics

## Project Structure

```
oven_batching/
├── environment/     # Core environment implementation
├── agents/         # Agent implementations (Random, FIFO, PPO)
└── training/       # Training scripts and analysis tools
```

## Dependencies

- `gymnasium`: RL environment framework
- `stable-baselines3`: PPO implementation
- `numpy`, `pandas`: Data handling
- `matplotlib`: Visualization
- `torch`: Neural network backend

## License

Random Personal Use - no need for license
