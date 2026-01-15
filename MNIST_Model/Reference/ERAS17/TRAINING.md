# Training Guide

Advanced training techniques and best practices for the Autonomous Car Navigation project.

---

## 🎯 Training Objectives

### Short-term Goals (0-500 episodes)
- Learn basic obstacle avoidance
- Reach single targets consistently
- Reduce crash frequency

### Medium-term Goals (500-1500 episodes)
- Navigate multi-target sequences
- Handle 90-degree turns smoothly
- Maintain stable positive rewards

### Long-term Goals (1500+ episodes)
- Master complex environments
- Generalize to unseen maps
- Achieve near-optimal navigation

---

## 📊 Training Phases

### Phase 1: Random Exploration (Episodes 0-100)

**Characteristics:**
- High exploration noise (0.4)
- Frequent crashes
- Negative or low rewards
- Buffer filling up

**What's Happening:**
- Agent explores state-action space randomly
- Collecting diverse experiences
- No meaningful policy yet

**What to Do:**
- **Be patient** - this is normal
- Monitor buffer size (should reach ~6400 by episode 100)
- Ensure car reaches different parts of the map

**Red Flags:**
- Car stuck in one corner
- Buffer not growing
- Application freezing

### Phase 2: Initial Learning (Episodes 100-300)

**Characteristics:**
- Noise decreasing (0.4 → 0.3)
- Occasional target reaches
- Reward variance high but trending upward
- Fewer consecutive crashes

**What's Happening:**
- Policy gradient starting to work
- Learning basic sensor-action mappings
- Discovering reward structure

**What to Do:**
- Watch for first successful target reach
- Note which strategies emerge (e.g., wall-following)
- Consider saving model checkpoints

**Red Flags:**
- Rewards still consistently negative after 300 episodes
- Car always crashes in same spot
- No improvement in moving average

### Phase 3: Policy Refinement (Episodes 300-1000)

**Characteristics:**
- Noise low (0.2-0.1)
- Consistent target reaching
- Smooth reward curve
- Efficient navigation paths

**What's Happening:**
- Exploiting learned policy
- Fine-tuning steering and speed
- Optimizing trajectories

**What to Do:**
- Test on different start/target positions
- Introduce new challenges (more targets, tighter spaces)
- Analyze failure cases

**Red Flags:**
- Stuck in local minimum (e.g., always orbiting)
- Overfitting to specific start position
- Performance degrading

### Phase 4: Mastery (Episodes 1000+)

**Characteristics:**
- Noise at minimum (0.1)
- Near-perfect navigation
- Stable high rewards
- Generalizes to new scenarios

**What's Happening:**
- Policy converged
- Robust to perturbations
- Ready for deployment

**What to Do:**
- Test on completely new maps
- Experiment with different car sizes
- Save final model for inference

---

## 🔧 Hyperparameter Tuning

### Learning Rate (LR)

**Default:** 0.0005

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.0001 | Slow, stable learning | Unstable training, oscillating rewards |
| 0.0005 | Balanced (default) | Most scenarios |
| 0.001 | Fast, potentially unstable | Simple environments, quick prototyping |

**Tuning Tips:**
- If rewards oscillate wildly → decrease LR
- If learning is too slow (no improvement after 500 episodes) → increase LR
- Monitor critic loss (should decrease over time)

### Batch Size

**Default:** 64

| Value | Effect | When to Use |
|-------|--------|-------------|
| 32 | Less stable, faster updates | Limited memory, simple tasks |
| 64 | Balanced (default) | Most scenarios |
| 128 | More stable, slower updates | Complex tasks, unstable training |

**Tuning Tips:**
- Larger batch = more stable gradients but slower learning
- Smaller batch = faster updates but higher variance
- Must be ≤ buffer size

### Discount Factor (GAMMA)

**Default:** 0.99

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.9 | Short-sighted (values immediate rewards) | Short episodes, sparse rewards |
| 0.99 | Balanced (default) | Most scenarios |
| 0.999 | Far-sighted (values long-term rewards) | Very long episodes |

**Tuning Tips:**
- Higher GAMMA = agent plans further ahead
- Lower GAMMA = agent focuses on immediate rewards
- For multi-target tasks, use higher GAMMA

### Exploration Noise

**Default:** 0.4 → 0.1 (decays)

```python
# Current decay schedule
expl_noise = max(0.1, 0.4 - episode * 0.0001)
```

**Alternative Schedules:**

```python
# Faster decay (for simple tasks)
expl_noise = max(0.1, 0.4 - episode * 0.0003)

# Slower decay (for complex tasks)
expl_noise = max(0.1, 0.4 - episode * 0.00005)

# Exponential decay
expl_noise = max(0.1, 0.4 * 0.995 ** episode)
```

### Polyak Averaging (TAU)

**Default:** 0.003

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.001 | Very slow target updates | Highly unstable training |
| 0.003 | Balanced (default) | Most scenarios |
| 0.01 | Faster target updates | Stable training, quick adaptation |

**Tuning Tips:**
- Lower TAU = more stable but slower adaptation
- Higher TAU = faster adaptation but less stable
- Rarely needs adjustment

---

## 🎨 Reward Function Design

### Current Reward Structure

```python
reward = 0.0

# 1. Distance progress (PBRS)
reward += (prev_dist - dist) * 2.5

# 2. Angle progress (anti-orbit)
reward += (abs(prev_angle) - abs(angle)) * 1.5

# 3. Anti-orbiting penalty
if dist < 40 and angle_progress < 0.01 and abs(angle_norm) > 0.7:
    reward -= 1.5

# 4. Terminal rewards
if collision:
    reward = -50.0
elif target_reached:
    reward = 100.0
```

### Customization Examples

#### Example 1: Encourage Faster Navigation

```python
# Add time penalty
reward -= 0.1  # Small penalty per step

# Bonus for quick completion
if target_reached:
    time_bonus = max(0, 100 - steps_taken)
    reward += time_bonus
```

#### Example 2: Penalize Sharp Turns

```python
# Add smoothness reward
steering_penalty = abs(action[0]) * 0.05
reward -= steering_penalty
```

#### Example 3: Encourage Centering in Corridors

```python
# Compute distance to nearest walls (left and right sensors)
left_dist = state[0]
right_dist = state[6]
centering_reward = -abs(left_dist - right_dist) * 0.5
reward += centering_reward
```

#### Example 4: Multi-Objective Reward

```python
# Weighted combination
w_dist = 2.5
w_angle = 1.5
w_speed = 0.3
w_smooth = 0.1

reward = (
    w_dist * dist_progress +
    w_angle * angle_progress +
    w_speed * (speed - 2.0) +  # Encourage speed ~2.0
    w_smooth * (-abs(steering))
)
```

### Reward Shaping Best Practices

1. **Maintain PBRS**: Ensure potential-based structure to preserve optimal policy
2. **Balance Weights**: Distance progress should dominate other components
3. **Avoid Conflicting Signals**: Don't reward both speed and caution simultaneously
4. **Test Incrementally**: Add one component at a time and observe effects
5. **Normalize**: Keep reward magnitudes similar across components

---

## 🧪 Ablation Studies

### What is an Ablation Study?

Systematically removing components to understand their importance.

### Study 1: Remove Angle Progress Reward

```python
# Comment out this line
# reward += angle_progress * 1.5
```

**Expected Result:** Car orbits targets instead of approaching directly

**Conclusion:** Angle progress is critical for direct navigation

### Study 2: Remove Twin Critics (Simulate DDPG)

```python
# In optimize() method, change:
target_Q = target_Q1  # Instead of torch.min(target_Q1, target_Q2)
```

**Expected Result:** Overestimation bias, unstable training, more crashes

**Conclusion:** Twin critics reduce Q-value overestimation

### Study 3: Remove No-Progress Detection

```python
# Comment out:
# if self.no_progress_steps > 40:
#     done = True
```

**Expected Result:** Episodes never end, car gets stuck

**Conclusion:** No-progress detection prevents infinite loops

### Study 4: Remove Target Policy Smoothing

```python
# In optimize(), remove noise from target actions:
next_action = self.actor_target(next_state)  # No noise added
```

**Expected Result:** Slightly less stable training

**Conclusion:** Target smoothing reduces variance in Q-targets

---

## 📈 Curriculum Learning

### What is Curriculum Learning?

Training on progressively harder tasks, like human education.

### Curriculum 1: Distance-Based

```python
# Week 1: Close targets (distance < 100)
# Week 2: Medium targets (distance 100-300)
# Week 3: Far targets (distance > 300)
```

### Curriculum 2: Complexity-Based

```python
# Stage 1: Open space, single target
# Stage 2: Open space, 3 targets
# Stage 3: Narrow corridors, single target
# Stage 4: Narrow corridors, 5 targets
```

### Curriculum 3: Sensor-Based

```python
# Level 1: 7 sensors, SENSOR_DIST=40
# Level 2: 7 sensors, SENSOR_DIST=25
# Level 3: 5 sensors, SENSOR_DIST=25
# Level 4: 3 sensors, SENSOR_DIST=20
```

### Implementation

```python
def get_curriculum_stage(episode):
    if episode < 200:
        return "easy"
    elif episode < 500:
        return "medium"
    else:
        return "hard"

# In setup:
stage = get_curriculum_stage(episode)
if stage == "easy":
    place_target_nearby()
elif stage == "medium":
    place_target_medium_distance()
else:
    place_target_far_with_obstacles()
```

---

## 🔍 Debugging Training Issues

### Issue 1: Rewards Not Improving

**Symptoms:**
- Flat reward curve after 500 episodes
- Moving average stuck near zero

**Diagnosis:**
```python
# Add logging in optimize()
print(f"Actor Loss: {actor_loss.item():.4f}")
print(f"Critic Loss: {critic_loss.item():.4f}")
print(f"Q-values: {current_Q1.mean().item():.2f}")
```

**Solutions:**
- Reduce learning rate (LR = 0.0001)
- Increase batch size (BATCH_SIZE = 128)
- Check reward function (are rewards too sparse?)
- Verify target networks are updating (check TAU)

### Issue 2: Catastrophic Forgetting

**Symptoms:**
- Performance degrades after initial improvement
- Agent "forgets" how to reach targets

**Diagnosis:**
```python
# Check buffer diversity
print(f"Unique states: {len(set(replay_buffer.storage))}")
```

**Solutions:**
- Increase buffer size (max_size = 200000)
- Reduce learning rate
- Add experience prioritization

### Issue 3: Overestimation Bias

**Symptoms:**
- Q-values explode (> 1000)
- Unstable training
- Frequent crashes

**Diagnosis:**
```python
# Monitor Q-values
print(f"Q1: {current_Q1.mean().item():.2f}, Q2: {current_Q2.mean().item():.2f}")
```

**Solutions:**
- Verify twin critics are active
- Reduce learning rate
- Increase TAU (slower target updates)

### Issue 4: High Variance

**Symptoms:**
- Reward curve oscillates wildly
- Inconsistent performance

**Diagnosis:**
```python
# Check gradient norms
actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
print(f"Actor Grad Norm: {actor_grad_norm:.4f}")
```

**Solutions:**
- Increase batch size
- Add gradient clipping
- Reduce exploration noise

---

## 💾 Checkpointing Strategy

### When to Save Checkpoints

```python
# Save every N episodes
if episode % 100 == 0:
    save_checkpoint(f"checkpoint_ep{episode}.pth")

# Save on new best performance
if episode_reward > best_reward:
    best_reward = episode_reward
    save_checkpoint("best_model.pth")

# Save on milestones
if episode in [100, 500, 1000, 2000]:
    save_checkpoint(f"milestone_{episode}.pth")
```

### Checkpoint Contents

```python
def save_checkpoint(path):
    torch.save({
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_target_state_dict': actor_target.state_dict(),
        'critic_target_state_dict': critic_target.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
        'critic_optimizer': critic_optimizer.state_dict(),
        'replay_buffer': replay_buffer.storage,
        'best_reward': best_reward,
        'expl_noise': expl_noise,
    }, path)
```

---

## 📊 Logging and Visualization

### Comprehensive Logging

```python
import csv
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'episode', 'reward', 'steps', 
                'expl_noise', 'buffer_size', 'crashes',
                'actor_loss', 'critic_loss', 'avg_q_value'
            ])
    
    def log(self, **kwargs):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                kwargs.get('episode', 0),
                kwargs.get('reward', 0),
                kwargs.get('steps', 0),
                kwargs.get('expl_noise', 0),
                kwargs.get('buffer_size', 0),
                kwargs.get('crashes', 0),
                kwargs.get('actor_loss', 0),
                kwargs.get('critic_loss', 0),
                kwargs.get('avg_q_value', 0),
            ])
```

### TensorBoard Integration (Optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/car_navigation')

# In training loop:
writer.add_scalar('Reward/episode', episode_reward, episode)
writer.add_scalar('Loss/actor', actor_loss, episode)
writer.add_scalar('Loss/critic', critic_loss, episode)
writer.add_scalar('Noise/exploration', expl_noise, episode)
```

View in browser:
```bash
tensorboard --logdir=runs
```

---

## 🎯 Advanced Techniques

### 1. Hindsight Experience Replay (HER)

Learn from failures by relabeling goals:

```python
# After episode ends (even if failed):
for transition in episode_buffer:
    # Original transition
    replay_buffer.add(transition)
    
    # Relabeled transition (treat final position as goal)
    relabeled_transition = relabel_goal(transition, final_position)
    replay_buffer.add(relabeled_transition)
```

### 2. Prioritized Experience Replay

Sample important transitions more frequently:

```python
class PrioritizedReplayBuffer:
    def sample(self, batch_size):
        # Sample based on TD error
        priorities = [abs(td_error) for td_error in self.td_errors]
        probs = priorities / sum(priorities)
        indices = np.random.choice(len(self.storage), batch_size, p=probs)
        return [self.storage[i] for i in indices]
```

### 3. Automatic Hyperparameter Tuning

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    
    # Train with these hyperparameters
    avg_reward = train(lr, batch_size, gamma)
    return avg_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 📚 Further Reading

### Papers
- [TD3 Paper](https://arxiv.org/abs/1802.09477) - Original algorithm
- [Reward Shaping](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf) - Theory
- [HER Paper](https://arxiv.org/abs/1707.01495) - Hindsight Experience Replay

### Tutorials
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [Berkeley CS285](http://rail.eecs.berkeley.edu/deeprlcourse/)

---

**Last Updated**: January 15, 2026
