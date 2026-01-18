# 🚗 NeuralNav: Autonomous Car Navigation Assignment

NeuralNav is an interactive reinforcement learning simulation where a self-driving car learns to navigate a city map using a **Deep Q-Network (DQN)**. This project is designed as an educational assignment for the **TSAI ERA** program, focusing on hyperparameter tuning and understanding RL dynamics.

![NeuralNav UI](Mangalore_CityMap.png)

## 🌟 Key Features

- **Deep Q-Learning**: Implements a DQN with a target network for stable learning.
- **Prioritized Experience Replay**: A custom buffer system that prioritizes successful episodes to accelerate learning.
- **Interactive UI**: Built with PyQt6, allowing users to:
  - Load custom city maps.
  - Set car starting positions and multiple sequential targets.
  - Visualize real-time sensor data and reward history.
- **Physics Simulation**: Realistic car movement with adjustable speed, turn angles, and sensor arrays.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ERAS16A2
   ```

2. **Install dependencies**:
   This project uses `PyQt6`, `torch`, and `numpy`.
   ```bash
   pip install PyQt6 torch numpy
   ```
   *Note: Using `uv` or `conda` for environment management is recommended.*

## 🚀 Usage Guide

1. **Run the application**:
   ```bash
   python citymap_assignment.py
   ```

2. **Setup the Simulation**:
   - **Step 1**: Click anywhere on the map to set the **Car's Starting Position**.
   - **Step 2**: Click on the map to set **Target(s)**. You can add multiple targets in sequence.
   - **Step 3**: **Right-click** to finalize the setup.
   - **Step 4**: Press **SPACE** or click **START** to begin the training.

3. **Controls**:
   - **RESET ALL**: Clears the map and resets the brain.
   - **LOAD MAP**: Import your own `.png` or `.jpg` city layout.
   - **PAUSE/RESUME**: Toggle the simulation loop.

## 🏗️ Architecture & Design

### System Overview

```mermaid
flowchart TB
    subgraph Environment["🌍 Environment (CityMap)"]
        MAP[City Map<br/>Sand/Road Detection]
        SENSORS[7 Directional Sensors]
        PHYSICS[Physics Engine<br/>Movement & Collision]
    end

    subgraph Agent["🚗 Agent (Car)"]
        STATE[State Vector<br/>9 Dimensions]
        ACTION[Action Selection<br/>5 Discrete Actions]
    end

    subgraph Brain["🧠 DQN Brain"]
        POLICY[Policy Network<br/>Q-Value Estimation]
        TARGET[Target Network<br/>Stable Q-Targets]
        REPLAY[Prioritized<br/>Experience Replay]
    end

    MAP --> SENSORS
    SENSORS --> STATE
    STATE --> POLICY
    POLICY --> ACTION
    ACTION --> PHYSICS
    PHYSICS --> |Reward| REPLAY
    STATE --> REPLAY
    REPLAY --> |Training Batch| POLICY
    POLICY -.-> |Soft Update τ| TARGET
    TARGET --> |Target Q-Values| POLICY
```

### Neural Network Architecture

```mermaid
flowchart LR
    subgraph Input["📥 Input Layer"]
        I1[Sensor 1]
        I2[Sensor 2]
        I3[Sensor 3]
        I4[Sensor 4]
        I5[Sensor 5]
        I6[Sensor 6]
        I7[Sensor 7]
        I8[Angle to Target]
        I9[Distance to Target]
    end

    subgraph Hidden["🔷 Hidden Layers"]
        H1[FC Layer 1<br/>128 neurons<br/>ReLU]
        H2[FC Layer 2<br/>128 neurons<br/>ReLU]
    end

    subgraph Output["📤 Output Layer"]
        O1[Q Left]
        O2[Q Straight]
        O3[Q Right]
        O4[Q Sharp Left]
        O5[Q Sharp Right]
    end

    I1 & I2 & I3 & I4 & I5 & I6 & I7 & I8 & I9 --> H1
    H1 --> H2
    H2 --> O1 & O2 & O3 & O4 & O5
```

### Training Loop

```mermaid
sequenceDiagram
    participant Env as 🌍 Environment
    participant Agent as 🚗 Agent
    participant Policy as 📊 Policy Net
    participant Target as 🎯 Target Net
    participant Buffer as 💾 Replay Buffer

    loop Each Step
        Env->>Agent: State (sensors, angle, distance)
        Agent->>Policy: Forward pass
        Policy->>Agent: Q-values for all actions
        
        alt Exploration (ε probability)
            Agent->>Agent: Random action
        else Exploitation
            Agent->>Agent: argmax(Q-values)
        end
        
        Agent->>Env: Execute action
        Env->>Agent: Reward + Next State
        Agent->>Buffer: Store (s, a, r, s', done)
    end

    loop Training (every N steps)
        Buffer->>Policy: Sample prioritized batch
        Policy->>Target: Get target Q-values
        Target->>Policy: Q_target = r + γ × max(Q_next)
        Policy->>Policy: Update weights (MSE Loss)
        Policy-->>Target: Soft update (τ)
    end
```

### Reward System

```mermaid
flowchart TD
    subgraph Rewards["💰 Reward Structure"]
        direction TB
        
        A{Action Taken} --> B{On Sand?}
        B -->|Yes| C[🔴 Negative Reward<br/>Penalty for boundary]
        B -->|No| D{Closer to Target?}
        D -->|Yes| E[🟢 Positive Reward<br/>Progress bonus]
        D -->|No| F[🟡 Small Penalty<br/>Encourage efficiency]
        
        G{Reached Target?} -->|Yes| H[🏆 Large Bonus<br/>Goal achieved!]
        G -->|No| I[Continue Episode]
    end
```

---

## 🧠 Reinforcement Learning Details

### State Space (9 Dimensions)
- **7 Sensors**: Detecting "sand" (boundaries) at various angles (-45° to 45°).
- **Angle to Target**: Normalized orientation relative to the goal.
- **Distance to Target**: Normalized Euclidean distance.

### Action Space (5 Discrete Actions)
- `0`: Left Turn
- `1`: Straight
- `2`: Right Turn
- `3`: Sharp Left
- `4`: Sharp Right

### Hyperparameters
The following parameters are critical for convergence:
- **Learning Rate (LR)**: Controls how fast the network updates.
- **Gamma ($\gamma$)**: The discount factor for future rewards.
- **Epsilon ($\epsilon$)**: The exploration rate (starts at 1.0 and decays).
- **Tau ($\tau$)**: Soft update coefficient for the target network.

## 📝 Assignment: "Fix the Parameters"

Several critical parameters in `citymap_assignment.py` are intentionally set to incorrect values. Your task is to find the `# FIX ME!` comments and set appropriate values to help the car learn effectively.

### Common Questions
- **What happens if the boundary-signal is weak?** The car will ignore the road and drive straight through "sand" to reach the target.
- **What is the effect of reducing Gamma?** The car becomes "short-sighted," focusing only on immediate rewards and failing to plan long-term paths to distant targets.
- **What does Temperature do?** While not used in this specific $\epsilon$-greedy implementation, temperature in RL controls the randomness of action selection in Softmax policies.

## 📊 Visualization
The UI includes a **Reward History Chart** showing:
- **Raw Scores**: Performance of individual episodes.
- **Moving Average (10)**: The overall learning trend (look for an upward curve!).

---
*Developed for the TSAI ERA Program.*
