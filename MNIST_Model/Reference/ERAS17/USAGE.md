# Usage Guide

Complete guide to using the Autonomous Car Navigation application.

---

## 🚀 Quick Start

### Launch the Application

```bash
python Autonomous_DrivingRL.py
```

### Basic Workflow

1. **Place the Car**: Click on the map to set the starting position
2. **Add Target(s)**: Click again to place one or more targets
3. **Finish Setup**: Right-click when done placing targets
4. **Start Training**: Press `SPACE` or click the "START" button
5. **Watch & Learn**: Observe the car learning to navigate

---

## 🎮 User Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Autonomous Car Navigation - TD3                            │
├──────────────────────────────────┬──────────────────────────┤
│                                  │  📊 Control Panel        │
│                                  │  ┌────────────────────┐  │
│         Map Canvas               │  │ Episode: 42        │  │
│      (Interactive Area)          │  │ Score: 125.3       │  │
│                                  │  │ Noise: 0.35        │  │
│     🚗 ← Car                     │  │ Buffer: 2048       │  │
│     🎯 ← Target                  │  │ Crashes: 1/3       │  │
│                                  │  └────────────────────┘  │
│                                  │                          │
│                                  │  [START] [PAUSE] [RESET] │
│                                  │                          │
├──────────────────────────────────┴──────────────────────────┤
│  📈 Reward Chart                                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ╱╲    ╱╲                                              │ │
│  │ ╱  ╲  ╱  ╲╱                                            │ │
│  │      ╲╱                                                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Components

- **Map Canvas**: Interactive area where the car navigates
- **Control Panel**: Displays real-time training metrics
- **Reward Chart**: Visualizes episode rewards over time
- **Buttons**: Control training (Start, Pause, Reset)

---

## 🖱️ Mouse Controls

### Left Click
- **First Click**: Place the car at the clicked position
- **Subsequent Clicks**: Add target waypoints (up to 8 targets)

### Right Click
- **Finalize Setup**: Confirm target placement and prepare for training

### Mouse Hover
- View coordinates and pixel color (useful for debugging)

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Start/Resume training |
| `P` | Pause training |
| `R` | Reset episode (keep current targets) |
| `ESC` | Quit application |

---

## 📊 Understanding the Metrics

### Episode
- **Definition**: One complete run from start to target(s) or crash
- **Range**: Increments from 0
- **Typical Duration**: 50-500 steps per episode

### Score (Episode Reward)
- **Definition**: Cumulative reward for the current episode
- **Positive**: Good progress toward target
- **Negative**: Crashes or poor navigation
- **Target**: Aim for consistently positive scores

### Exploration Noise
- **Definition**: Amount of randomness added to actions
- **Range**: 0.4 (high exploration) → 0.1 (exploitation)
- **Decay**: Gradually decreases as training progresses
- **Purpose**: Balances exploration vs. exploitation

### Buffer Size
- **Definition**: Number of experiences stored in replay buffer
- **Maximum**: 100,000 transitions
- **Training Starts**: When buffer size > 64 (BATCH_SIZE)
- **Optimal**: Larger buffer = more diverse training data

### Consecutive Crashes
- **Definition**: Number of crashes in a row
- **Range**: 0-3
- **Auto-Reset**: After 3 crashes, car returns to start position
- **Purpose**: Prevents getting stuck in bad states

---

## 📈 Reward Chart Interpretation

### Visual Elements

- **Blue Line**: Raw episode rewards
- **Yellow Line**: 10-episode moving average (smoothed trend)
- **Dashed Line**: Zero baseline
- **X-axis**: Episode number
- **Y-axis**: Reward value

### What to Look For

#### Healthy Training
```
Reward
  │     ╱╲  ╱╲╱╲
  │    ╱  ╲╱    ╲╱╲
  │   ╱            ╲  ← Upward trend
  │  ╱
  │ ╱
  └─────────────────── Episodes
```

#### Unstable Training
```
Reward
  │  ╱╲    ╱╲
  │ ╱  ╲  ╱  ╲
  │      ╲╱    ╲╱╲  ← High variance
  │              ╲╱
  │
  └─────────────────── Episodes
```

#### Converged Training
```
Reward
  │ ────────────────  ← Stable high reward
  │
  │
  │
  │
  └─────────────────── Episodes
```

---

## 🎯 Setting Up Training Scenarios

### Scenario 1: Simple Straight Line

```
Start: (100, 300)
Target: (500, 300)
```

**Expected Behavior**: Car learns to drive straight in ~50 episodes

### Scenario 2: 90-Degree Turn

```
Start: (100, 100)
Target: (500, 500)
```

**Expected Behavior**: Car learns to navigate diagonal path in ~100 episodes

### Scenario 3: Multi-Target Waypoints

```
Start: (100, 100)
Targets: [(200, 200), (400, 200), (400, 400), (200, 400)]
```

**Expected Behavior**: Car learns sequential navigation in ~300 episodes

### Scenario 4: Narrow Corridor

Place targets that require navigating through tight spaces.

**Expected Behavior**: Car learns precise steering in ~500 episodes

---

## 🔧 Customizing Training

### Using a Custom Map

1. Prepare an image (PNG or JPG):
   - **Black pixels**: Walls/obstacles
   - **White pixels**: Drivable area
   - **Recommended size**: 600x600 to 1200x1200 pixels

2. Place the image in the project directory

3. Modify `Autonomous_DrivingRL.py` (around line 500):

```python
# Change this line
map_path = "city_map.png"

# To your custom map
map_path = "my_custom_map.png"
```

### Adjusting Hyperparameters

Edit `Autonomous_DrivingRL.py` (lines 30-45):

```python
# Physics
CAR_WIDTH = 15          # Increase for larger car
SENSOR_DIST = 25        # Increase for longer-range sensors
SENSOR_ANGLE = 90       # Increase for wider field of view

# Training
BATCH_SIZE = 64         # Increase for more stable updates
LR = 0.0005             # Decrease if training is unstable
GAMMA = 0.99            # Discount factor (0.9-0.99)
TAU = 0.003             # Target network update rate
```

### Modifying Reward Function

Edit `CarBrain.step()` method (around line 250):

```python
# Example: Add speed penalty
reward -= abs(speed - 3.0) * 0.1  # Encourage speed of 3.0

# Example: Add smoothness reward
reward -= abs(steering) * 0.05    # Penalize sharp turns
```

---

## 📝 Training Tips

### 1. Start Simple
- Begin with a single target in an open area
- Gradually increase complexity (more targets, tighter spaces)

### 2. Monitor Metrics
- **Score increasing**: Good progress
- **Noise decreasing**: Transitioning to exploitation
- **Buffer filling**: More diverse experiences

### 3. Be Patient
- **First 100 episodes**: Random exploration, expect crashes
- **100-500 episodes**: Learning basic navigation
- **500+ episodes**: Refining policy, smooth navigation

### 4. When to Reset
- **Stuck in local minimum**: Reset and try different start/target positions
- **Consistently negative scores**: Check reward function or hyperparameters
- **No improvement after 1000 episodes**: Reduce learning rate or increase batch size

---

## 🐛 Common Issues During Usage

### Issue 1: Car Spins in Place

**Cause**: Angle reward dominates distance reward

**Solution**:
```python
# Reduce angle reward weight (line ~280)
reward += angle_progress * 0.5  # Instead of 1.5
```

### Issue 2: Car Crashes Frequently

**Cause**: Sensor range too short or speed too high

**Solution**:
```python
# Increase sensor range
SENSOR_DIST = 40  # Instead of 25

# Or reduce max speed
# In Actor.forward() (line ~105)
speed = torch.sigmoid(x[:, 1]) * 3.0 + 0.5  # Instead of 4.5
```

### Issue 3: Car Orbits Target

**Cause**: Anti-orbit mechanism disabled or too weak

**Solution**:
```python
# Increase anti-orbit penalty (line ~290)
if dist < 40 and angle_progress < 0.01 and abs(angle_norm) > 0.7:
    reward -= 3.0  # Instead of 1.5
```

### Issue 4: Training Unstable (Reward Oscillates)

**Cause**: Learning rate too high

**Solution**:
```python
# Reduce learning rate
LR = 0.0001  # Instead of 0.0005
```

---

## 💾 Saving and Loading Models

### Saving a Trained Model

Add this code to `Autonomous_DrivingRL.py` after training:

```python
# In CarBrain class
def save_model(self, path="trained_model.pth"):
    torch.save({
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'episode': self.episode,
    }, path)
    print(f"Model saved to {path}")
```

Call it after good performance:

```python
# In game_loop() or finalize_episode()
if episode_reward > 200:  # Threshold for "good" performance
    brain.save_model(f"model_ep{episode}.pth")
```

### Loading a Trained Model

```python
def load_model(self, path="trained_model.pth"):
    checkpoint = torch.load(path)
    self.actor.load_state_dict(checkpoint['actor'])
    self.critic.load_state_dict(checkpoint['critic'])
    self.episode = checkpoint['episode']
    print(f"Model loaded from {path}")
```

---

## 🎥 Recording Training Sessions

### Using Screen Recording

**Windows**: Windows + G (Game Bar)  
**macOS**: Cmd + Shift + 5  
**Linux**: SimpleScreenRecorder, OBS Studio

### Logging Training Data

Add logging to `finalize_episode()`:

```python
import csv

def log_episode(self, episode, reward):
    with open('training_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, reward, self.expl_noise, len(self.replay_buffer.storage)])
```

---

## 📊 Analyzing Training Results

### Plotting Rewards

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load logged data
df = pd.read_csv('training_log.csv', names=['episode', 'reward', 'noise', 'buffer'])

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['reward'], alpha=0.3, label='Raw')
plt.plot(df['episode'], df['reward'].rolling(10).mean(), label='Moving Avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig('training_curve.png')
```

---

## 🎓 Learning Progression

### Beginner (Episodes 0-100)
- **Objective**: Understand the interface
- **Task**: Train on a single target in open space
- **Success**: Car reaches target occasionally

### Intermediate (Episodes 100-500)
- **Objective**: Handle multiple targets
- **Task**: Train on 3-5 sequential waypoints
- **Success**: Car completes all targets >50% of the time

### Advanced (Episodes 500+)
- **Objective**: Navigate complex environments
- **Task**: Train on narrow corridors and tight turns
- **Success**: Smooth, efficient navigation with minimal crashes

---

## 🔍 Debugging Tips

### Enable Debug Logging

Uncomment debug prints in `step()` method (line ~250):

```python
if self.total_steps % 50 == 0:
    print(f"Step {self.total_steps}: Dist={dist:.1f}, Angle={angle_deg:.1f}, Reward={reward:.2f}")
```

### Visualize Sensors

Add sensor visualization in `paintEvent()`:

```python
# Draw sensor lines
for i, sensor_val in enumerate(state[:7]):
    angle = self.angle + (i - 3) * (SENSOR_ANGLE / 6)
    length = sensor_val * SENSOR_DIST
    end_x = self.x + length * math.cos(math.radians(angle))
    end_y = self.y + length * math.sin(math.radians(angle))
    painter.drawLine(int(self.x), int(self.y), int(end_x), int(end_y))
```

---

**Next Steps**: See [TRAINING.md](TRAINING.md) for advanced training techniques.

**Last Updated**: January 15, 2026
