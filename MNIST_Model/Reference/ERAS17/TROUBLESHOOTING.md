# Troubleshooting Guide

Common issues and solutions for the Autonomous Car Navigation project.

---

## 🚨 Installation Issues

### Issue: PyQt6 Import Error

**Error Message:**
```
ImportError: cannot import name 'QApplication' from 'PyQt6.QtWidgets'
```

**Cause:** Incomplete or corrupted PyQt6 installation

**Solutions:**

1. **Reinstall PyQt6:**
   ```bash
   pip uninstall PyQt6
   pip install PyQt6 --no-cache-dir
   ```

2. **Check Python version:**
   ```bash
   python --version  # Must be 3.8+
   ```

3. **Try system-specific installation:**
   ```bash
   # Windows
   pip install PyQt6-Qt6
   
   # macOS
   brew install pyqt6
   pip install PyQt6
   ```

---

### Issue: PyTorch CUDA Not Available

**Error Message:**
```
CUDA not available, using CPU
```

**Cause:** PyTorch installed without CUDA support

**Solutions:**

1. **Check CUDA availability:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

2. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision
   
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

**Note:** CPU-only training works fine for this project!

---

### Issue: NumPy Version Conflict

**Error Message:**
```
ImportError: numpy.core.multiarray failed to import
```

**Cause:** NumPy version incompatibility

**Solutions:**

```bash
pip install --upgrade numpy
# or
pip install numpy==1.24.0
```

---

## 🐛 Runtime Errors

### Issue: Application Crashes on Startup

**Error Message:**
```
Segmentation fault (core dumped)
```

**Cause:** Graphics driver or Qt platform plugin issue

**Solutions:**

1. **Set Qt platform plugin:**
   ```bash
   # Linux
   export QT_QPA_PLATFORM=xcb
   
   # macOS
   export QT_QPA_PLATFORM=cocoa
   ```

2. **Update graphics drivers:**
   - **NVIDIA:** Download from nvidia.com
   - **AMD:** Download from amd.com
   - **Intel:** Update through system updates

3. **Run with software rendering:**
   ```bash
   export QT_QPA_PLATFORM=offscreen
   python Autonomous_DrivingRL.py
   ```

---

### Issue: Map Image Not Found

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'city_map.png'
```

**Cause:** Map file missing or incorrect path

**Solutions:**

1. **Check file exists:**
   ```bash
   ls city_map.png
   # or on Windows
   dir city_map.png
   ```

2. **Use absolute path in code:**
   ```python
   # In Autonomous_DrivingRL.py, line ~500
   map_path = r"C:\full\path\to\city_map.png"
   ```

3. **Create a simple test map:**
   ```python
   from PIL import Image
   import numpy as np
   
   # Create 600x600 white image with black borders
   img = np.ones((600, 600, 3), dtype=np.uint8) * 255
   img[:10, :] = 0  # Top border
   img[-10:, :] = 0  # Bottom border
   img[:, :10] = 0  # Left border
   img[:, -10:] = 0  # Right border
   
   Image.fromarray(img).save('test_map.png')
   ```

---

### Issue: Out of Memory Error

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Cause:** Replay buffer or batch size too large

**Solutions:**

1. **Reduce buffer size:**
   ```python
   # In Autonomous_DrivingRL.py, line ~65
   self.replay_buffer = ReplayBuffer(max_size=50000)  # Instead of 100000
   ```

2. **Reduce batch size:**
   ```python
   # Line ~35
   BATCH_SIZE = 32  # Instead of 64
   ```

3. **Use CPU instead of GPU:**
   ```python
   # In CarBrain.__init__()
   self.device = torch.device("cpu")  # Force CPU
   ```

---

## 🎮 Training Issues

### Issue: Car Spins in Place

**Symptoms:**
- Car rotates without moving forward
- High steering values, low speed
- Reward stuck near zero

**Diagnosis:**
```python
# Add debug print in step()
print(f"Action: steer={action[0]:.2f}, speed={action[1]:.2f}")
```

**Solutions:**

1. **Reduce angle reward weight:**
   ```python
   # Line ~280
   reward += angle_progress * 0.5  # Instead of 1.5
   ```

2. **Add speed reward:**
   ```python
   # Encourage forward movement
   reward += speed * 0.2
   ```

3. **Penalize excessive steering:**
   ```python
   reward -= abs(steering) * 0.1
   ```

---

### Issue: Car Crashes Frequently

**Symptoms:**
- Consecutive crashes = 3/3
- Negative rewards
- Car doesn't avoid walls

**Diagnosis:**
```python
# Print sensor values
state = self.get_state()
print(f"Sensors: {state[:7]}")  # Should show low values near walls
```

**Solutions:**

1. **Increase sensor range:**
   ```python
   # Line ~30
   SENSOR_DIST = 40  # Instead of 25
   ```

2. **Add crash penalty:**
   ```python
   # In step() method
   if collision:
       reward = -100.0  # Increase from -50.0
   ```

3. **Reduce max speed:**
   ```python
   # In Actor.forward()
   speed = torch.sigmoid(x[:, 1]) * 3.0 + 0.5  # Instead of 4.5
   ```

---

### Issue: Car Orbits Target

**Symptoms:**
- Car circles around target
- Never reaches target
- Distance stays constant

**Diagnosis:**
```python
# Check angle progress
print(f"Angle: {angle_deg:.1f}, AngleProgress: {angle_progress:.3f}")
```

**Solutions:**

1. **Increase anti-orbit penalty:**
   ```python
   # Line ~290
   if dist < 40 and angle_progress < 0.01 and abs(angle_norm) > 0.7:
       reward -= 3.0  # Instead of 1.5
   ```

2. **Widen terminal cone:**
   ```python
   # Line ~310
   if dist < 30 and abs(angle_norm) < 0.5:  # Instead of 25 and 0.3
       reward = 100.0
   ```

3. **Add direct approach reward:**
   ```python
   # Reward moving toward target at correct angle
   if abs(angle_norm) < 0.3:
       reward += dist_progress * 5.0  # Bonus for aligned approach
   ```

---

### Issue: Training Unstable (Oscillating Rewards)

**Symptoms:**
- Reward chart shows wild oscillations
- Performance inconsistent
- Moving average doesn't improve

**Diagnosis:**
```python
# Monitor losses
print(f"Actor Loss: {actor_loss.item():.4f}")
print(f"Critic Loss: {critic_loss.item():.4f}")
```

**Solutions:**

1. **Reduce learning rate:**
   ```python
   # Line ~37
   LR = 0.0001  # Instead of 0.0005
   ```

2. **Increase batch size:**
   ```python
   # Line ~35
   BATCH_SIZE = 128  # Instead of 64
   ```

3. **Add gradient clipping:**
   ```python
   # In optimize() method, after loss.backward()
   torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
   torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
   ```

---

### Issue: No Learning After 500 Episodes

**Symptoms:**
- Reward flat at zero or negative
- No improvement in behavior
- Buffer full but no learning

**Diagnosis:**
```python
# Check if optimize() is being called
print(f"Buffer size: {len(self.replay_buffer.storage)}")
print(f"Optimizing: {len(self.replay_buffer.storage) >= BATCH_SIZE}")
```

**Solutions:**

1. **Verify reward function:**
   ```python
   # Add debug logging
   print(f"Reward: {reward:.2f}, Dist: {dist:.1f}, DistProgress: {dist_progress:.2f}")
   ```

2. **Check network initialization:**
   ```python
   # Print initial Q-values
   state = torch.randn(1, 10)
   action = torch.randn(1, 2)
   q_val = self.critic.Q1(state, action)
   print(f"Initial Q-value: {q_val.item():.2f}")  # Should be small
   ```

3. **Reset and retrain:**
   ```python
   # Delete saved models and restart
   # Sometimes helps escape bad local minima
   ```

---

### Issue: Catastrophic Forgetting

**Symptoms:**
- Performance degrades after initial improvement
- Agent "forgets" successful strategies
- Reward decreases over time

**Diagnosis:**
```python
# Check buffer diversity
unique_states = len(set([tuple(s) for s, _, _, _, _ in replay_buffer.storage]))
print(f"Unique states: {unique_states}")
```

**Solutions:**

1. **Increase buffer size:**
   ```python
   self.replay_buffer = ReplayBuffer(max_size=200000)
   ```

2. **Reduce learning rate:**
   ```python
   LR = 0.0001
   ```

3. **Save checkpoints and revert:**
   ```python
   # Load previous best model
   checkpoint = torch.load('best_model.pth')
   self.actor.load_state_dict(checkpoint['actor'])
   ```

---

## 🖥️ Performance Issues

### Issue: Slow Training (Low FPS)

**Symptoms:**
- Training very slow (< 10 steps/second)
- GUI laggy
- High CPU usage

**Solutions:**

1. **Reduce visualization frequency:**
   ```python
   # In game_loop(), update GUI less often
   if self.total_steps % 10 == 0:  # Instead of every step
       self.update()
   ```

2. **Disable chart updates during training:**
   ```python
   # Only update chart every 100 steps
   if episode % 100 == 0:
       self.reward_chart.update_chart(episode_reward)
   ```

3. **Use GPU if available:**
   ```python
   self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

---

### Issue: High Memory Usage

**Symptoms:**
- RAM usage > 8GB
- System slowing down
- Application crashes with memory error

**Solutions:**

1. **Reduce buffer size:**
   ```python
   self.replay_buffer = ReplayBuffer(max_size=50000)
   ```

2. **Clear old episodes:**
   ```python
   # Periodically clear buffer
   if episode % 1000 == 0:
       self.replay_buffer.storage = []
       self.replay_buffer.ptr = 0
   ```

3. **Use float32 instead of float64:**
   ```python
   # Ensure all tensors are float32
   state = torch.FloatTensor(state)  # Not DoubleTensor
   ```

---

## 🎨 Visualization Issues

### Issue: Reward Chart Not Updating

**Symptoms:**
- Chart remains empty
- No blue/yellow lines
- Scores not displayed

**Diagnosis:**
```python
# In finalize_episode()
print(f"Episode {episode}: Reward = {episode_reward}")
```

**Solutions:**

1. **Check chart update call:**
   ```python
   # Ensure this is called in finalize_episode()
   self.reward_chart.update_chart(episode_reward)
   ```

2. **Force repaint:**
   ```python
   self.reward_chart.update()
   self.reward_chart.repaint()
   ```

---

### Issue: Car Not Visible on Map

**Symptoms:**
- Can't see car after placing it
- Map appears but no car sprite

**Solutions:**

1. **Check car position:**
   ```python
   print(f"Car position: ({self.x}, {self.y})")
   ```

2. **Verify paintEvent:**
   ```python
   # Ensure car is drawn in paintEvent()
   painter.setBrush(QColor(255, 0, 0))  # Red car
   painter.drawRect(int(self.x), int(self.y), CAR_WIDTH, CAR_HEIGHT)
   ```

---

## 🔧 Code Issues

### Issue: AttributeError: 'CarBrain' object has no attribute 'X'

**Common Missing Attributes:**
- `set_start_pos`
- `prev_distance`
- `no_progress_steps`

**Solutions:**

1. **Check initialization:**
   ```python
   # In CarBrain.__init__()
   self.prev_distance = 0
   self.no_progress_steps = 0
   self.prev_angle_norm = 0
   ```

2. **Ensure reset() initializes all variables:**
   ```python
   def reset(self):
       self.prev_distance = math.hypot(self.target[0] - self.x, self.target[1] - self.y)
       self.no_progress_steps = 0
       # ... etc
   ```

---

### Issue: TypeError: 'NoneType' object is not subscriptable

**Cause:** Target not set before accessing

**Solutions:**

```python
# Check target exists before using
if self.target is None:
    return  # or set default target

# Or ensure target is set in __init__
self.target = (300, 300)  # Default target
```

---

## 📊 Debugging Checklist

When training isn't working:

- [ ] Check reward function is being called
- [ ] Verify state has correct dimensions (10)
- [ ] Ensure actions are in valid range
- [ ] Confirm optimize() is being called
- [ ] Check buffer has enough samples (> BATCH_SIZE)
- [ ] Verify losses are decreasing
- [ ] Ensure target networks are updating
- [ ] Check exploration noise is decaying
- [ ] Confirm no NaN values in tensors
- [ ] Verify gradients are flowing (not zero)

---

## 🆘 Getting More Help

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In key methods:
logger.debug(f"State: {state}")
logger.debug(f"Action: {action}")
logger.debug(f"Reward: {reward}")
```

### Create Minimal Reproducible Example

```python
# Simplify to isolate issue
# Remove GUI, use simple environment
# Test network forward pass in isolation
```

### Check System Information

```python
import sys
import torch
import platform

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Platform: {platform.platform()}")
print(f"CPU Count: {os.cpu_count()}")
```

---

## 📚 Additional Resources

- [PyTorch Troubleshooting](https://pytorch.org/docs/stable/notes/faq.html)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [TD3 GitHub Issues](https://github.com/sfujim/TD3/issues)

---

**Last Updated**: January 15, 2026
