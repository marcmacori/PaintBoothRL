# Dynamic Oven Batching Environment - Agent Comparison

This project implements and compares different agents for the Dynamic Oven Batching Environment, including a PPO (Proximal Policy Optimization) reinforcement learning agent.

## Environment Overview

The Dynamic Oven Batching Environment simulates a production system with ovens used for curing painted panels. The agent must decide when to launch oven batches to balance:

- **Throughput**: Maximize the number of completed jobs
- **Energy Cost**: Minimize energy consumption considering time-of-day tariffs
- **Lateness Penalties**: Avoid completing jobs after their due dates

### Reward System

The environment uses a complex reward system with multiple components:

1. **Throughput Reward**: +1.0 for each completed job
2. **Energy Cost Penalty**: Negative reward based on energy consumption and tariffs
   - Peak hours (8:00-18:00): 1.5x multiplier
   - Shoulder hours (6:00-8:00, 18:00-22:00): 1.2x multiplier
   - Off-peak hours (22:00-6:00): 0.8x multiplier
3. **Lateness Penalty**: Negative reward for jobs completed after due date
4. **Idle Penalty**: Small negative reward for waiting (encourages proactive behavior)
5. **Underfill Penalty**: Penalty for launching batches with <50% capacity

### Dynamic Arrival Patterns

The environment uses shift-based arrival rates:
- **Morning shift (6:00-14:00)**: 2.5x base rate with peaks at start/end
- **Afternoon shift (14:00-22:00)**: 2.0x base rate
- **Night shift (22:00-6:00)**: 1.5x base rate

## Agents

### 1. Random Agent (`random_agent.py`)
A simple baseline that selects actions randomly from valid options.

**Strategy**: Randomly choose between wait, launch, and heat actions.

### 2. FIFO Agent (`fifo_agent.py`)
A heuristic agent that processes jobs in First-In-First-Out order.

**Strategy**:
- Launch batches when ovens are ready and jobs are available
- Heat ovens when they're cold and jobs are waiting
- Wait when no immediate actions are possible

### 3. PPO Agent (`ppo_agent.py`)
A basic reinforcement learning agent using Proximal Policy Optimization from stable-baselines3.

**Features**:
- Simple MLP policy network
- Basic PPO implementation with standard hyperparameters
- Action masking for valid actions
- Easy training and model saving/loading

## Usage

### Quick Test
Run the test script to verify all agents work correctly:

```bash
cd oven_batching
python test_agents.py
```

### Agent Comparison
Run the full comparison experiment:

```bash
cd oven_batching
python training/compare_agents.py
```

This will:
1. Train a PPO agent for 100,000 timesteps
2. Evaluate all three agents over 20 episodes each
3. Generate comparison plots and save results to CSV files

### Custom Training
You can also train and use agents individually:

```python
from environment.core import DynamicOvenBatchingEnv
from agents.ppo_agent import PPOAgent

# Create environment
env = DynamicOvenBatchingEnv(
    num_ovens=2,
    oven_capacity=9,
    batch_time=10.0,
    horizon=1440.0,  # 24 hours
    seed=42
)

# Create and train PPO agent
ppo_agent = BasicPPOAgent(env)
ppo_agent.train(total_timesteps=50000, log_dir="./my_ppo_logs")

# Use the trained agent
obs, info = env.reset()
action, _ = ppo_agent.predict(obs, deterministic=True)
```

## Expected Results

Based on the reward structure and environment dynamics, you should expect:

1. **Random Agent**: Lowest performance, serves as baseline
2. **FIFO Agent**: Moderate performance, good throughput but may not optimize energy costs
3. **PPO Agent**: Best performance, learns to balance throughput, energy costs, and lateness penalties

The PPO agent should learn strategies like:
- Heating ovens during off-peak hours
- Batching jobs efficiently to minimize underfill penalties
- Prioritizing urgent jobs to avoid lateness penalties
- Timing batch launches to optimize energy tariffs

## File Structure

```
oven_batching/
├── environment/
│   └── core.py              # Main environment implementation
├── agents/
│   ├── random_agent.py      # Random baseline agent
│   ├── fifo_agent.py        # FIFO heuristic agent
│   └── ppo_agent.py         # PPO reinforcement learning agent
├── training/
│   └── compare_agents.py    # Agent comparison script
├── test_agents.py           # Test script for all agents
└── README.md               # This file
```

## Dependencies

The project requires:
- `gymnasium>=0.29.1`
- `stable-baselines3>=2.7.0`
- `torch>=2.8.0`
- `numpy>=2.3.2`
- `matplotlib>=3.10.5`
- `pandas>=2.3.2`
- `tqdm>=4.67.1`

All dependencies are listed in `requirements.txt`.

## Configuration

You can customize the environment parameters in the comparison script:

```python
env_config = {
    'num_ovens': 2,              # Number of ovens
    'oven_capacity': 9,          # Jobs per batch
    'batch_time': 10.0,          # Minutes per batch
    'horizon': 1440.0,           # Simulation time (minutes)
    'arrival_rate': 0.5,         # Jobs per minute
    'energy_alpha': 1.0,         # Energy cost weight
    'lateness_beta': 2.0,        # Lateness penalty weight
    'use_dynamic_arrivals': True # Use shift-based arrivals
}
```

## Troubleshooting

1. **Import Errors**: Make sure you're running scripts from the correct directory
2. **Training Issues**: Reduce `training_timesteps` for faster testing
3. **Memory Issues**: Reduce batch size or number of environments
4. **CUDA Issues**: The PPO agent will automatically use CPU if CUDA is not available
