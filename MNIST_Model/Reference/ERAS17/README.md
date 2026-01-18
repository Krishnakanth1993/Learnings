# Autonomous Car Navigation with TD3

A deep reinforcement learning implementation for autonomous car navigation using Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Overview

This project implements an autonomous car that learns to navigate complex environments using **sensor-based perception** and **continuous control**. The agent learns to:

- Navigate through corridors and tight spaces
- Handle sharp turns and U-turns
- Reach multiple sequential targets
- Avoid obstacles using distance sensors
- Generalize to unseen maps

### Key Features

- ✅ **TD3 Algorithm**: State-of-the-art actor-critic method
- ✅ **Continuous Control**: Smooth steering and speed control
- ✅ **Sensor-Based Navigation**: 7 distance sensors (no map required)
- ✅ **Advanced Reward Shaping**: Potential-based shaping with anti-orbit mechanisms
- ✅ **Multiple Targets**: Sequential waypoint navigation
- ✅ **Real-time Visualization**: PyQt6-based GUI with live metrics

---

## 📁 Project Structure

```
ERAS17/
├── Autonomous_DrivingRL.py     # Main implementation (latest version)
├── city_velocity.py            # Version with velocity control
├── city_new.py                 # Intermediate version
├── citymap_assignment.py       # Original assignment
├── city_map.png               # Default training map
├── README.md                   # This file - Project overview
├── ARCHITECTURE.md             # 🏗️ System architecture and design
├── INSTALLATION.md             # 📦 Setup and installation guide
├── USAGE.md                    # 🎮 How to use the application
├── TRAINING.md                 # 🎓 Advanced training techniques
├── TROUBLESHOOTING.md          # 🔧 Common issues and solutions
├── CONTRIBUTING.md             # 🤝 Contribution guidelines
├── CHANGELOG.md                # 📝 Version history and updates
├── CONCEPTS_GUIDE.md           # 📚 Comprehensive RL concepts guide
└── OVERESTIMATION_BIAS.md      # 📚 Deep dive into Q-learning bias
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision
pip install PyQt6
pip install numpy
```

### Running the Application

```bash
python Autonomous_DrivingRL.py
```

### Usage

1. **Click on the map** to place the car (starting position)
2. **Click again** to place target(s) - you can add multiple sequential targets
3. **Right-click** when done placing targets
4. **Press SPACE** or click "START" to begin training
5. Watch the car learn to navigate!

---

## 📚 Documentation

### 📖 Getting Started

- **[Installation Guide](INSTALLATION.md)** - Complete setup instructions
  - System requirements
  - Multiple installation methods (pip, conda, poetry)
  - Troubleshooting installation issues
  - Verification steps

- **[Usage Guide](USAGE.md)** - How to use the application
  - User interface overview
  - Mouse and keyboard controls
  - Understanding metrics and charts
  - Training scenarios and examples

### 🎓 Advanced Topics

- **[Architecture Documentation](ARCHITECTURE.md)** - System design deep dive
  - High-level architecture overview
  - State and action space design
  - Neural network architecture
  - TD3 algorithm implementation
  - Reward function design

- **[Training Guide](TRAINING.md)** - Advanced training techniques
  - Training phases and progression
  - Hyperparameter tuning
  - Reward function customization
  - Ablation studies
  - Curriculum learning
  - Debugging training issues

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
  - Installation problems
  - Runtime errors
  - Training issues (spinning, crashing, orbiting)
  - Performance optimization
  - Debugging checklist

### 🛠️ Development

- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute
  - Code style guidelines
  - Testing requirements
  - Pull request process
  - Feature development guidelines

- **[Changelog](CHANGELOG.md)** - Version history
  - Release notes
  - Breaking changes
  - Migration guides
  - Roadmap

---

## 📚 Learning Resources

### 📖 Concept Guides

We've created comprehensive guides to help you understand the concepts used in this implementation:

#### 1. **[Deep RL Concepts Guide](CONCEPTS_GUIDE.md)**
   
   A complete guide covering:
   - **TD3 Algorithm**: Actor-critic architecture, twin critics, delayed updates
   - **Experience Replay**: Off-policy learning and sample efficiency
   - **Reward Shaping**: Distance, orientation, and anti-orbit rewards
   - **Continuous Actions**: Steering and speed control
   - **Target Networks**: Polyak averaging and stability
   - **Exploration Strategies**: Noise-based exploration with decay
   - **State Representation**: Sensor-based observations
   - **Neural Networks**: MLP architecture and activation functions
   - **Training Loop**: Optimization and loss functions
   - **Advanced Techniques**: No-progress detection, angle progress tracking
   - **Failure Cases & Solutions**: Why naive approaches fail and how to fix them
   - **Robotics Concepts**: PBRS, anti-orbiting, geometry detection, corridor centering

#### 2. **[Overestimation Bias Explained](OVERESTIMATION_BIAS.md)**
   
   Deep dive into Q-learning's fundamental challenge:
   - **Intuitive Explanation**: Restaurant review analogy
   - **Mathematical Foundation**: Why `max` operator causes bias
   - **Propagation**: How bias compounds through training
   - **Solutions**: Double Q-learning and TD3's clipped approach
   - **Empirical Evidence**: Comparison graphs and experiments
   - **Practical Debugging**: How to detect bias in your training

### 📄 Key Papers Referenced

1. **TD3** - [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018)
2. **DDPG** - [Continuous Control with Deep RL](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015)
3. **DQN** - [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
4. **Reward Shaping** - [Policy Invariance Under Reward Transformations](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf) (Ng et al., 1999)
5. **Double Q-Learning** - [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)

### 🌐 Online Resources

- **[Spinning Up in Deep RL](https://spinningup.openai.com/)** - Best starting point
- **[Sutton & Barto - RL Book](http://incompleteideas.net/book/the-book-2nd.html)** - Free online
- **[Berkeley CS285](http://rail.eecs.berkeley.edu/deeprlcourse/)** - Deep RL course
- **[Lilian Weng's Blog](https://lilianweng.github.io/)** - Excellent RL explanations
- **[LaValle's Planning Algorithms](http://planning.cs.uiuc.edu/)** - Motion planning

---

## 🏗️ Architecture

### System Overview

```mermaid
flowchart TB
    subgraph Environment["🌍 Environment"]
        Map["City Map"]
        Car["🚗 Car Agent"]
        Target["🎯 Target"]
    end

    subgraph Sensors["📡 Perception"]
        S0["Sensor 0"]
        S1["Sensor 1"]
        S2["Sensor 2"]
        S3["Sensor 3"]
        S4["Sensor 4"]
        S5["Sensor 5"]
        S6["Sensor 6"]
    end

    subgraph Brain["🧠 CarBrain (TD3)"]
        Actor["Actor Network"]
        Critic1["Critic 1"]
        Critic2["Critic 2"]
        Buffer["Replay Buffer"]
    end

    Car --> Sensors
    Sensors --> |State| Actor
    Actor --> |Action| Car
    Car --> |Experience| Buffer
    Buffer --> |Mini-batch| Critic1
    Buffer --> |Mini-batch| Critic2
    Critic1 --> |Q-values| Actor
    Critic2 --> |Q-values| Actor
```

### State Space (10 dimensions)

```mermaid
flowchart LR
    subgraph Sensors["Distance Sensors (7)"]
        direction TB
        S0["sensor_0"]
        S1["sensor_1"]
        S2["sensor_2"]
        S3["sensor_3"]
        S4["sensor_4"]
        S5["sensor_5"]
        S6["sensor_6"]
    end

    subgraph Target["Target Info (3)"]
        direction TB
        Sin["sin(angle)"]
        Cos["cos(angle)"]
        Dist["norm_distance"]
    end

    Sensors --> State["State Vector\n(10 dims)"]
    Target --> State
```

### Action Space (2 dimensions)

```mermaid
flowchart LR
    subgraph Actions["Continuous Actions"]
        Steering["🔄 Steering\n[-5.0, 5.0] deg/step"]
        Speed["⚡ Speed\n[0.5, 5.0] px/step"]
    end

    Actor["Actor\nNetwork"] --> Actions
    Actions --> Car["🚗 Car\nMovement"]
```

### Neural Network Architecture

```mermaid
flowchart LR
    subgraph Actor["Actor Network (Policy)"]
        direction LR
        A_In["Input\n(10)"] --> A_H1["Hidden 1\n(400)\nReLU"]
        A_H1 --> A_H2["Hidden 2\n(300)\nReLU"]
        A_H2 --> A_Out["Output\n(2)\nTanh"]
    end

    subgraph Critic["Twin Critics (Q-Networks)"]
        direction LR
        C_In["State + Action\n(12)"] --> C_H1["Hidden 1\n(400)\nReLU"]
        C_H1 --> C_H2["Hidden 2\n(300)\nReLU"]
        C_H2 --> C_Out["Q-Value\n(1)"]
    end

    Actor -.-> |"Action"| Critic
```

### TD3 Training Flow

```mermaid
flowchart TD
    subgraph Training["TD3 Training Loop"]
        Sample["Sample Mini-batch\nfrom Replay Buffer"]
        Sample --> Noise["Add Target Policy\nNoise (clipped)"]
        Noise --> TargetQ["Compute Target Q\nmin(Q1', Q2')"]
        TargetQ --> CriticLoss["Update Critics\n(MSE Loss)"]
        CriticLoss --> Check{"Step % 2 == 0?"}
        Check --> |Yes| ActorUpdate["Update Actor\n(Policy Gradient)"]
        Check --> |No| Skip["Skip Actor Update"]
        ActorUpdate --> Polyak["Soft Update\nTarget Networks\n(τ = 0.003)"]
        Skip --> Next["Next Step"]
        Polyak --> Next
    end
```

---

## 🎮 Hyperparameters

### Physics
```python
CAR_WIDTH = 15
CAR_HEIGHT = 12
SENSOR_DIST = 25        # Sensor range
SENSOR_ANGLE = 90       # Total sensor spread
```

### Reinforcement Learning
```python
BATCH_SIZE = 64
GAMMA = 0.99            # Discount factor
LR = 0.0005             # Learning rate
TAU = 0.003             # Polyak averaging
MAX_CONSECUTIVE_CRASHES = 3
```

### TD3 Specific
```python
policy_noise = 0.2      # Target policy smoothing
noise_clip = 0.5        # Noise clipping range
policy_freq = 2         # Delayed policy updates
expl_noise = 0.4        # Initial exploration (decays)
```

---

## 🧪 Advanced Features

### 1. Potential-Based Reward Shaping (PBRS)
```python
# Distance progress
reward += (prev_dist - dist) * 2.5

# Angle progress (anti-orbit)
reward += (abs(prev_angle) - abs(angle)) * 1.5
```

**Why it works**: Preserves optimal policy while accelerating learning (Ng et al., 1999)

### 2. Anti-Orbiting Mechanism
```python
# Detects circling behavior
if dist < 40 and angle_progress < 0.01 and abs(angle_norm) > 0.7:
    reward -= 1.5
```

**Solves**: Potential field local minimum problem

### 3. No-Progress Detection
```python
if abs(dist_progress) < 0.01:
    no_progress_steps += 1
    
if no_progress_steps > 40:
    done = True  # Force episode reset
```

**Prevents**: Infinite loops and stuck states

### 4. Terminal Cone
```python
if dist < 25 and abs(angle_norm) < 0.3:
    reward = 100.0  # Target reached
```

**Requires**: Both proximity AND proper alignment

---

## 🔬 Experiment Suggestions

### Ablation Studies

Try removing components to understand their importance:

```python
# 1. Remove angle progress
# angle_progress = abs(self.prev_angle_norm) - abs(angle_norm)
# reward += angle_progress * 1.5
# Expected: Agent orbits targets

# 2. Remove no-progress detection
# if self.no_progress_steps > 40:
#     done = True
# Expected: Infinite episodes, stuck in corners

# 3. Disable twin critics (simulate DDPG)
target_Q = target_Q1  # Instead of min(Q1, Q2)
# Expected: Overestimation, crashes, instability
```

### Hyperparameter Tuning

```python
# Learning rate
LR = [1e-4, 5e-4, 1e-3]

# Batch size
BATCH_SIZE = [32, 64, 128]

# Reward weights
distance_weight = [1.0, 2.5, 5.0]
angle_weight = [0.5, 1.5, 3.0]
```

---

## 📊 Monitoring Training

### Key Metrics

- **Episode Score**: Cumulative reward per episode
- **Exploration Noise**: Decays from 0.4 → 0.1
- **Buffer Size**: Experience replay capacity
- **Consecutive Crashes**: Auto-resets after 3 crashes

### Debug Logging

```python
# Every 50 steps:
Step 1000: AngleDiff=45.2, Turn=2.34, Speed=3.21, Dist=123.4, DistDiff=-5.2
```

### Reward Chart

- **Blue line**: Raw episode scores
- **Yellow line**: 10-episode moving average
- **Dashed line**: Zero baseline

---

## 🐛 Common Issues & Solutions

### Issue: Car spins in place
**Cause**: Angle reward dominates distance reward  
**Solution**: Reduce angle reward weight or add speed shaping penalty

### Issue: Car crashes frequently
**Cause**: Sensor range too short or speed too high  
**Solution**: Increase `SENSOR_DIST` or reduce max speed

### Issue: Car orbits target
**Cause**: Anti-orbit mechanism disabled  
**Solution**: Ensure angle progress reward is active

### Issue: Training unstable
**Cause**: Learning rate too high or overestimation bias  
**Solution**: Reduce `LR` or verify twin critics are working

---

## 🎓 Understanding the Code

### Recommended Reading Order

1. **Start with**: [concepts_guide.md](CONCEPTS_GUIDE.md) - Section 1 (TD3 Overview)
2. **Then read**: [overestimation_bias_explained.md](OVERESTIMATION_BIAS.md)
3. **Deep dive**: concepts_guide.md - Sections 3-11 (Reward Shaping, Advanced Techniques, Failure Cases)
4. **Explore code**: Start with `CarBrain.__init__()` in `Autonomous_DrivingRL.py`
5. **Understand training**: Follow `game_loop()` → `step()` → `optimize()`

### Code Structure

```mermaid
classDiagram
    class ReplayBuffer {
        +storage: deque
        +add(state, action, reward, next_state, done)
        +sample(batch_size)
    }

    class Actor {
        +fc1: Linear(10, 400)
        +fc2: Linear(400, 300)
        +fc3: Linear(300, 2)
        +forward(state) → action
    }

    class Critic {
        +fc1: Linear(12, 400)
        +fc2: Linear(400, 300)
        +fc3: Linear(300, 1)
        +forward(state, action) → Q-value
    }

    class CarBrain {
        +actor: Actor
        +critic_1: Critic
        +critic_2: Critic
        +memory: ReplayBuffer
        +__init__()
        +get_state() → state
        +step(action) → reward, done
        +optimize()
        +select_action(state) → action
    }

    class NeuralNavApp {
        +brain: CarBrain
        +game_loop()
        +setup_ui()
        +update_charts()
    }

    CarBrain --> Actor : uses
    CarBrain --> Critic : uses (×2)
    CarBrain --> ReplayBuffer : stores experience
    NeuralNavApp --> CarBrain : controls
```

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different reward functions
- Try alternative RL algorithms (SAC, PPO)
- Add new features (curriculum learning, HER)
- Improve visualization

---

## 📝 License

MIT License - Feel free to use for learning and research

---

## 🙏 Acknowledgments

- **OpenAI Spinning Up** - Educational resources
- **Fujimoto et al.** - TD3 algorithm
- **Ng et al.** - Reward shaping theory
- **PyQt6** - GUI framework

---

## 📧 Contact

For questions about the implementation or concepts, refer to the detailed guides in `.gemini/antigravity/brain/`.

---

## 📖 Quick Reference Table

| I want to... | See this document |
|--------------|-------------------|
| Install the project | [INSTALLATION.md](INSTALLATION.md) |
| Run the application | [USAGE.md](USAGE.md) or [Quick Start](#-quick-start) |
| Understand the architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Improve training performance | [TRAINING.md](TRAINING.md) |
| Fix a bug or error | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Contribute to the project | [CONTRIBUTING.md](CONTRIBUTING.md) |
| See what's changed | [CHANGELOG.md](CHANGELOG.md) |
| Learn about TD3 | [concepts_guide.md](CONCEPTS_GUIDE.md) |
| Understand Q-learning bias | [overestimation_bias_explained.md](OVERESTIMATION_BIAS.md) |

---

**Happy Learning! 🚗💨**
