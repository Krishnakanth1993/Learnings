# GridWorld Value Iteration - Real-time Visualization

A comprehensive implementation of the **Value Iteration algorithm** for a 4×4 GridWorld environment with real-time interactive visualization.

## 📋 Overview

This project demonstrates the **Bellman Equation** for value iteration in a reinforcement learning context. An agent starts at the top-left corner (state 0) and tries to reach the bottom-right corner (state 15) in a 4×4 grid. The agent moves with equal probability in all four directions (up, down, left, right).

### Key Features

- ✅ **Complete Value Iteration Implementation** using the Bellman equation
- ✅ **Real-time Interactive Visualization** with animated grid updates
- ✅ **Convergence Tracking** with delta history chart
- ✅ **Premium UI Design** with glassmorphism and smooth animations
- ✅ **Playback Controls** (Play, Pause, Reset, Step-by-step)
- ✅ **Speed Control** for visualization playback

## 🎯 Problem Statement

**GridWorld Setup:**
- **Grid Size:** 4×4 (16 states total)
- **Start State:** Top-left corner (State 0)
- **Terminal State:** Bottom-right corner (State 15)
- **Actions:** Up, Down, Left, Right (equal probability: 0.25 each)
- **Rewards:** -1 for each move, 0 for terminal state
- **Discount Factor (γ):** 1.0 (no discounting)
- **Convergence Threshold (θ):** 1e-4

## 📐 Bellman Equation

The value iteration algorithm uses the Bellman equation to iteratively update state values:

```
V(s) = Σ_a π(a|s) * Σ_s' P(s'|s,a) * [R(s,a,s') + γ * V(s')]
```

Where:
- `V(s)` = Value of state s
- `π(a|s)` = Policy probability (0.25 for uniform random policy)
- `P(s'|s,a)` = Transition probability (1.0 for deterministic transitions)
- `R(s,a,s')` = Reward (-1 for all non-terminal states)
- `γ` = Discount factor (1.0)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- NumPy
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd c:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERAS16
```

2. **Install dependencies:**
```bash
pip install numpy
```

### Running the Project

**Step 1: Run the Value Iteration Algorithm**

```bash
python gridworld_value_iteration.py
```

This will:
- Initialize the GridWorld environment
- Run value iteration until convergence
- Print iteration progress in real-time
- Display the final value function as a grid
- Save the iteration history to `gridworld_history.json`

**Expected Output:**
```
Starting Value Iteration for 4x4 GridWorld
Gamma: 1.0, Theta: 0.0001
Terminal State: 15
------------------------------------------------------------
Iteration    0 | Delta: 0.250000 | Max V: 0.0000 | Min V: -0.2500
Iteration    1 | Delta: 0.187500 | Max V: 0.0000 | Min V: -0.4375
...
Iteration  470 | Delta: 0.000099 | Max V: 0.0000 | Min V: -59.4237
------------------------------------------------------------
Converged after 471 iterations!

Final Value Function:
============================================================
-59.424 | -57.424 | -54.281 | -51.710
------------------------------------------------------------
-57.424 | -54.567 | -49.710 | -45.139
------------------------------------------------------------
-54.281 | -49.710 | -40.854 | -29.998
------------------------------------------------------------
-51.710 | -45.139 | -29.998 |   0.000
============================================================
```

**Step 2: Start the Web Server**

```bash
python -m http.server 8000
```

**Step 3: Open the Visualization**

Open your web browser and navigate to:
```
http://localhost:8000/gridworld_visualization.html
```

## 🎨 Visualization Features

### Interactive Controls

- **▶ Start:** Begin the animated playback of value iteration
- **⏸ Pause:** Pause the animation
- **🔄 Reset:** Reset to initial state (all values = 0)
- **⏭ Step:** Advance one iteration at a time
- **Speed Slider:** Adjust animation speed (10ms - 500ms per iteration)

### Visual Elements

1. **4×4 Grid Display**
   - Color-coded cells based on value (red = low, green = high)
   - Start state (S0) highlighted in pink gradient
   - Terminal state (S15) highlighted in green gradient
   - Real-time value updates with smooth animations

2. **Statistics Panel**
   - Status indicator (Running/Paused/Converged)
   - Current iteration number
   - Delta (maximum change in values)
   - Progress bar
   - Convergence threshold
   - Discount factor

3. **Convergence Chart**
   - Real-time line chart showing delta over iterations
   - Visualizes convergence behavior
   - Powered by Chart.js

## 🎯 Policy Visualization (New Feature)

We have added a **Policy Distribution Visualization** feature that allows you to see the optimal action directions after convergence.

### ✨ Key Features

- **Interactive Policy Toggle:** Switch between viewing state values and optimal policy arrows using the "🎯 Show Policy" button.
- **Visual Policy Display:** Large directional arrows (↑ ↓ ← →) indicate the best action(s) from each state.
- **Multiple Optimal Actions:** When multiple actions are equally good (ties), all optimal arrows are displayed.
- **Smooth Transitions:** Premium fade animations when toggling between views.

### 🔍 Interpreting the Policy

After convergence (471 iterations), the visualization shows the optimal policy:

- **Single Arrow:** Only one optimal action exists (e.g., State 1 → Right).
- **Multiple Arrows:** Multiple actions yield the same optimal expected value (e.g., State 0 → Down OR Right).
- **Target Icon (🎯):** Represents the terminal/goal state.

**Why Multiple Arrows?**
Due to the uniform random policy during value iteration and the grid's symmetry, states equidistant from the goal often have multiple paths with the same expected cost. The algorithm correctly identifies all these optimal actions.

### 🎮 How to Use Policy View

1. **Run the Python script** to generate the latest data including policy information.
2. **Open the visualization** in your browser.
3. **Click "🎯 Show Policy"** to overlay the optimal action arrows on the grid.
4. **Click "📊 Show Values"** to return to the standard value view.

---

## 📊 Results

### Convergence Analysis

- **Total Iterations:** 471
- **Final Delta:** 0.000099 (below threshold of 0.0001)
- **Value Range:** -59.424 to 0.000

### Value Function & Policy Interpretation

The final results show:
- **Terminal state (S15):** 0.000 (goal state)
- **Policy Flow:** All optimal actions effectively guide the agent towards the bottom-right corner.
- **Values:** Negative values represent the expected cumulative reward (steps) to reach the goal.

## 🧮 Algorithm Details

### Pseudocode

```
Initialize:
  - Set grid size (4×4)
  - Define rewards (-1 per move, 0 for terminal)
  - Initialize V(s) = 0 for all states
  - Set γ = 1.0, θ = 1e-4
  - Define actions: [up, down, left, right]

Repeat until convergence:
  - δ ← 0
  - V_new ← copy of V
  
  For each state s (except terminal):
    - v ← 0
    - For each action a:
      - s' ← get_next_state(s, a)
      - v ← v + π(a|s) × [R(s) + γ × V(s')]
    - V_new(s) ← v
    - δ ← max(δ, |V_new(s) - V(s)|)
  
  - V ← V_new
  - If δ < θ, stop

Extract Policy:
  For each state s:
    - Compute Q(s,a) for all actions
    - Find max_q = max(Q(s,a))
    - Policy(s) = {a | Q(s,a) == max_q}

Return V, Policy
```

### Implementation Highlights

1. **State Representation:** States are numbered 0-15 (row-major order)
2. **Boundary Handling:** Actions that would move outside the grid keep the agent in the same state
3. **Uniform Random Policy:** Each action has probability 0.25
4. **Synchronous Updates:** All state values are updated simultaneously in each iteration
5. **Policy Extraction:** Post-convergence calculation of optimal actions based on final values

## 📁 File Structure

```
ERAS16/
├── gridworld_value_iteration.py    # Core algorithm & policy extraction
├── gridworld_visualization.html    # Interactive web visualization
├── gridworld_history.json          # Generated history & policy data
└── README_GridWorld.md             # Comprehensive documentation
```

## 🔧 Customization

### Modify Grid Size

In `gridworld_value_iteration.py`:
```python
grid = GridWorld(size=5, gamma=1.0, theta=1e-4)  # Change to 5×5
```

### Adjust Convergence Threshold

```python
grid = GridWorld(size=4, gamma=1.0, theta=1e-5)  # Stricter convergence
```

### Change Discount Factor

```python
grid = GridWorld(size=4, gamma=0.9, theta=1e-4)  # Add discounting
```

### Modify Animation Speed

In the visualization, use the speed slider or edit the default in HTML:
```javascript
<input type="range" id="speedSlider" min="10" max="500" value="50">
```

## 🎓 Educational Value

This project demonstrates:

1. **Reinforcement Learning Fundamentals**
   - Markov Decision Processes (MDPs)
   - Value functions & Bellman equations
   - Policy extraction & evaluation

2. **Dynamic Programming**
   - Iterative policy evaluation
   - Convergence criteria
   - State-value updates

3. **Visualization Techniques**
   - Real-time data updates
   - Interactive controls & charts
   - **Visualizing abstract concepts (Values vs. Policies)**

## 🌟 Design Highlights

The visualization features a **premium, modern design**:

- **Glassmorphism:** Frosted glass effect with backdrop blur
- **Gradient Backgrounds:** Vibrant purple gradient theme
- **Smooth Animations:** CSS transitions and keyframe animations
- **Responsive Layout:** Grid-based layout that adapts to screen size
- **Interactive Elements:** Hover effects and micro-animations
- **Professional Typography:** Clean, modern font styling

## 📚 References

- **Sutton & Barto:** *Reinforcement Learning: An Introduction* (Chapter 4 - Dynamic Programming)
- **Bellman Equation:** Foundation of dynamic programming in RL
- **Value Iteration:** Model-based RL algorithm for finding optimal policies

## 🐛 Troubleshooting

**Issue:** Visualization shows "No data - Run Python script first"
- **Solution:** Make sure to run `gridworld_value_iteration.py` first to generate `gridworld_history.json`

**Issue:** Web server port 8000 already in use
- **Solution:** Use a different port: `python -m http.server 8080`

**Issue:** Chart not displaying
- **Solution:** Ensure internet connection for Chart.js CDN, or download Chart.js locally

## 📝 License

This project is created for educational purposes.

## 👤 Author

Created as part of reinforcement learning studies.

---

**Enjoy exploring Value Iteration and Optimal Policies! 🎯**
