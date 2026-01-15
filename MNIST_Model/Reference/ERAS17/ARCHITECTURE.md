# Architecture Documentation

## System Overview

This project implements an autonomous car navigation system using **Twin Delayed Deep Deterministic Policy Gradient (TD3)**, a state-of-the-art deep reinforcement learning algorithm for continuous control tasks.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PyQt6 GUI Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Map Canvas   │  │ Reward Chart │  │ Control Panel│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CarBrain (RL Agent)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  State Representation (10D)                          │  │
│  │  • 7 Distance Sensors                                │  │
│  │  • sin/cos(angle_to_target)                          │  │
│  │  • normalized_distance                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              TD3 Neural Networks                     │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │   Actor    │  │  Critic 1  │  │  Critic 2  │    │  │
│  │  │  (Policy)  │  │ (Q-value)  │  │ (Q-value)  │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Experience Replay Buffer                     │  │
│  │         (Off-policy learning)                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Environment (Physics)                     │
│  • Collision Detection                                      │
│  • Sensor Raycasting                                        │
│  • Reward Computation                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 State Space Design

### Input Dimensions: 10

```python
state = [
    sensor_0,              # Front-left sensor (normalized 0-1)
    sensor_1,              # ...
    sensor_2,              # ...
    sensor_3,              # Front center sensor
    sensor_4,              # ...
    sensor_5,              # ...
    sensor_6,              # Front-right sensor
    sin(angle_to_target),  # Sine of relative angle
    cos(angle_to_target),  # Cosine of relative angle
    normalized_distance    # Distance to target (0-1)
]
```

### Sensor Configuration

- **Number of Sensors**: 7 (evenly distributed)
- **Sensor Range**: 25 pixels
- **Angular Spread**: 90 degrees (±45° from forward direction)
- **Normalization**: Distance values normalized by `SENSOR_DIST`

### Why This State Representation?

1. **Sensor-based**: No map required, generalizes to unseen environments
2. **Trigonometric encoding**: `sin/cos` preserves angle continuity (no discontinuity at ±180°)
3. **Normalized**: All values in [0, 1] range for stable neural network training
4. **Markovian**: Contains sufficient information for decision-making

---

## 🎯 Action Space Design

### Output Dimensions: 2 (Continuous)

```python
action = [
    steering,  # Range: [-5.0, 5.0] degrees per step
    speed      # Range: [0.5, 5.0] pixels per step
]
```

### Action Constraints

- **Steering**: Bounded by `tanh` activation × 5.0
- **Speed**: Bounded by `sigmoid` activation × 4.5 + 0.5
- **Exploration Noise**: Gaussian noise added during training (decays over time)

---

## 🧠 Neural Network Architecture

### Actor Network (Policy)

```
Input (10) 
    ↓
Linear(10 → 400) + ReLU
    ↓
Linear(400 → 300) + ReLU
    ↓
Linear(300 → 2)
    ↓
[tanh(steering) × 5.0, sigmoid(speed) × 4.5 + 0.5]
```

**Parameters**: ~123,402

### Critic Network (Q-function)

**Twin Critics** (reduces overestimation bias):

```
Critic 1:
State(10) + Action(2) → Concat(12)
    ↓
Linear(12 → 400) + ReLU
    ↓
Linear(400 → 300) + ReLU
    ↓
Linear(300 → 1) → Q1-value

Critic 2: (identical architecture)
    → Q2-value
```

**Parameters per critic**: ~124,301  
**Total Critic Parameters**: ~248,602

### Target Networks

- **Actor Target**: Slow-moving copy of Actor (Polyak averaging, τ=0.003)
- **Critic Target**: Slow-moving copy of Critics (Polyak averaging, τ=0.003)

**Purpose**: Stabilizes training by providing consistent targets

---

## 🔄 TD3 Algorithm Flow

### Training Loop

```python
for each step:
    # 1. Action Selection
    state = get_state()
    action = actor(state) + exploration_noise
    
    # 2. Environment Interaction
    next_state, reward, done = step(action)
    
    # 3. Store Experience
    replay_buffer.add((state, action, reward, next_state, done))
    
    # 4. Optimize (if buffer has enough samples)
    if len(replay_buffer) > BATCH_SIZE:
        # Sample mini-batch
        batch = replay_buffer.sample(BATCH_SIZE)
        
        # Compute target Q-value
        next_action = actor_target(next_state) + clipped_noise
        target_Q1 = critic_target.Q1(next_state, next_action)
        target_Q2 = critic_target.Q2(next_state, next_action)
        target_Q = reward + γ × min(target_Q1, target_Q2)  # Clipped Double Q-learning
        
        # Update Critics
        current_Q1 = critic.Q1(state, action)
        current_Q2 = critic.Q2(state, action)
        critic_loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)
        
        # Delayed Policy Update (every 2 steps)
        if step % 2 == 0:
            actor_loss = -critic.Q1(state, actor(state)).mean()
            
            # Polyak averaging for target networks
            actor_target ← τ × actor + (1-τ) × actor_target
            critic_target ← τ × critic + (1-τ) × critic_target
```

### Key TD3 Features

1. **Twin Critics**: Reduces overestimation bias
2. **Delayed Policy Updates**: Updates actor less frequently than critics
3. **Target Policy Smoothing**: Adds noise to target actions
4. **Clipped Double Q-learning**: Takes minimum of two Q-values

---

## 🎮 Reward Function Design

### Components

```python
reward = 0.0

# 1. Distance Progress (Potential-Based Reward Shaping)
dist_progress = prev_distance - current_distance
reward += dist_progress × 2.5

# 2. Angle Progress (Anti-Orbit Mechanism)
angle_progress = abs(prev_angle) - abs(current_angle)
reward += angle_progress × 1.5

# 3. Anti-Orbiting Penalty
if distance < 40 and angle_progress < 0.01 and abs(angle) > 0.7:
    reward -= 1.5  # Penalize circling behavior

# 4. Terminal Rewards
if collision:
    reward = -50.0
    done = True
elif distance < 25 and abs(angle) < 0.3:
    reward = 100.0  # Target reached
    switch_to_next_target()

# 5. No-Progress Detection
if abs(dist_progress) < 0.01:
    no_progress_steps += 1
    if no_progress_steps > 40:
        done = True  # Force reset
```

### Reward Shaping Theory

Based on **Potential-Based Reward Shaping (PBRS)** (Ng et al., 1999):

- Preserves optimal policy
- Accelerates learning by providing dense feedback
- Potential function: `Φ(s) = -distance_to_target`

---

## 🔧 Hyperparameters

### Physics Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CAR_WIDTH` | 15 | Car width in pixels |
| `CAR_HEIGHT` | 12 | Car height in pixels |
| `SENSOR_DIST` | 25 | Sensor range in pixels |
| `SENSOR_ANGLE` | 90 | Total sensor spread (degrees) |

### RL Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 64 | Mini-batch size for training |
| `GAMMA` | 0.99 | Discount factor |
| `LR` | 0.0005 | Learning rate (Adam optimizer) |
| `TAU` | 0.003 | Polyak averaging coefficient |
| `BUFFER_SIZE` | 100,000 | Replay buffer capacity |

### TD3 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy_noise` | 0.2 | Target policy smoothing noise |
| `noise_clip` | 0.5 | Noise clipping range |
| `policy_freq` | 2 | Delayed policy update frequency |
| `expl_noise` | 0.4 → 0.1 | Exploration noise (decays) |

---

## 📦 Module Structure

### Core Components

```
ERAS17/
├── Autonomous_DrivingRL.py                    # Main implementation
│   ├── ReplayBuffer          # Experience storage
│   ├── Actor                 # Policy network
│   ├── Critic                # Value network (twin)
│   ├── CarBrain              # RL agent
│   ├── RewardChart           # Visualization widget
│   └── NeuralNavApp          # Main application
│
├── city_velocity.py          # Velocity control variant
├── city_Autonomous_DrivingRL.py               # Intermediate version
└── citymap_assignment.py     # Original assignment
```

### Class Responsibilities

#### `ReplayBuffer`
- Stores transitions `(s, a, r, s', done)`
- Implements circular buffer (FIFO)
- Provides random sampling for mini-batches

#### `Actor`
- Maps states to actions
- Uses `tanh` and `sigmoid` for bounded outputs
- Trained to maximize Q-value

#### `Critic`
- Estimates Q-value for state-action pairs
- Twin architecture (Q1, Q2)
- Trained to minimize TD error

#### `CarBrain`
- Manages car physics and sensors
- Computes rewards
- Orchestrates training loop
- Handles multi-target navigation

#### `NeuralNavApp`
- PyQt6 GUI application
- Real-time visualization
- User interaction (placing car/targets)
- Training control (start/pause)

---

## 🔬 Advanced Features

### 1. Multi-Target Navigation

```python
# Sequential waypoint system
targets = [(x1, y1), (x2, y2), ...]
current_target_idx = 0

def switch_to_next_target():
    if current_target_idx < len(targets) - 1:
        current_target_idx += 1
    else:
        # All targets reached
        reset_episode()
```

### 2. No-Progress Detection

Prevents infinite loops and stuck states:

```python
if abs(distance_progress) < 0.01:
    no_progress_steps += 1
    
if no_progress_steps > 40:
    done = True  # Force episode termination
```

### 3. Exploration Decay

```python
# Starts at 0.4, decays to 0.1
expl_noise = max(0.1, 0.4 - episode * 0.0001)
```

### 4. Consecutive Crash Reset

```python
if crash:
    consecutive_crashes += 1
    if consecutive_crashes >= 3:
        reset_to_start_position()
```

---

## 🧪 Design Decisions

### Why TD3 over DDPG?

- **Overestimation Bias**: TD3's twin critics reduce Q-value overestimation
- **Stability**: Delayed policy updates prevent oscillations
- **Sample Efficiency**: Better performance with same amount of data

### Why Sensor-Based State?

- **Generalization**: Works on unseen maps
- **Scalability**: No need to process entire map image
- **Biological Inspiration**: Similar to how real autonomous vehicles use LIDAR

### Why Potential-Based Reward Shaping?

- **Theoretical Guarantee**: Preserves optimal policy (Ng et al., 1999)
- **Dense Feedback**: Faster learning than sparse rewards
- **Interpretability**: Easy to understand and debug

---

## 📚 References

1. **TD3 Paper**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
2. **DDPG Paper**: [Continuous Control with Deep RL](https://arxiv.org/abs/1509.02971)
3. **Reward Shaping**: [Policy Invariance Under Reward Transformations](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf)
4. **Double Q-Learning**: [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461)

---

## 🔍 Performance Characteristics

### Computational Complexity

- **Forward Pass**: O(1) - constant time per action
- **Training Step**: O(batch_size) - linear in batch size
- **Memory**: O(buffer_size) - dominated by replay buffer

### Training Time

- **Typical Convergence**: 500-2000 episodes
- **Steps per Episode**: 50-500 (depends on map complexity)
- **Real-time Factor**: ~100-200 steps/second (on modern CPU)

---

**Last Updated**: January 15, 2026
