# Contributing Guidelines

Thank you for your interest in contributing to the Autonomous Car Navigation project!

---

## 🎯 Ways to Contribute

### 1. Report Bugs
- Use GitHub Issues (if available)
- Provide detailed reproduction steps
- Include system information (OS, Python version, etc.)

### 2. Suggest Features
- Describe the feature and its benefits
- Explain use cases
- Consider implementation complexity

### 3. Improve Documentation
- Fix typos and clarify explanations
- Add examples and tutorials
- Translate to other languages

### 4. Submit Code
- Bug fixes
- New features
- Performance improvements
- Code refactoring

---

## 🚀 Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub (if applicable)
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/ERAS17.git
cd ERAS17
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision PyQt6 numpy

# Install development dependencies
pip install pytest black flake8 mypy
```

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## 📝 Code Style Guidelines

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Use 4 spaces for indentation (not tabs)
def my_function():
    pass

# Maximum line length: 100 characters
# (slightly longer than PEP 8's 79 for readability)

# Use descriptive variable names
sensor_distance = 25  # Good
sd = 25              # Bad

# Add docstrings to functions
def compute_reward(distance, angle):
    """
    Compute reward based on distance and angle to target.
    
    Args:
        distance (float): Distance to target in pixels
        angle (float): Angle to target in degrees
    
    Returns:
        float: Computed reward value
    """
    pass

# Use type hints
def get_state(self) -> np.ndarray:
    pass
```

### Formatting with Black

```bash
# Format all Python files
black Autonomous_DrivingRL.py city_velocity.py

# Check formatting without changing files
black --check Autonomous_DrivingRL.py
```

### Linting with Flake8

```bash
# Check code quality
flake8 Autonomous_DrivingRL.py --max-line-length=100

# Ignore specific errors
flake8 Autonomous_DrivingRL.py --ignore=E501,W503
```

---

## 🧪 Testing Guidelines

### Write Tests for New Features

```python
# tests/test_car_brain.py
import pytest
import torch
from New import CarBrain, Actor, Critic

def test_actor_output_shape():
    """Test that Actor outputs correct action shape."""
    actor = Actor(state_dim=10, action_dim=2)
    state = torch.randn(1, 10)
    action = actor(state)
    assert action.shape == (1, 2)

def test_critic_output_shape():
    """Test that Critic outputs correct Q-value shape."""
    critic = Critic(state_dim=10, action_dim=2)
    state = torch.randn(1, 10)
    action = torch.randn(1, 2)
    q1, q2 = critic(state, action)
    assert q1.shape == (1, 1)
    assert q2.shape == (1, 1)

def test_reward_computation():
    """Test reward function logic."""
    # Mock CarBrain instance
    brain = CarBrain(mock_map_image)
    
    # Test distance progress reward
    brain.prev_distance = 100
    brain.x, brain.y = 50, 50
    brain.target = (60, 50)  # Distance now 10
    
    _, reward, _, _ = brain.step([0, 2.0])  # Straight ahead
    assert reward > 0  # Should be positive for progress
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_car_brain.py

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## 🔍 Code Review Checklist

Before submitting, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] Type hints are used where appropriate
- [ ] Tests are written for new features
- [ ] Existing tests pass
- [ ] No unnecessary print statements (use logging instead)
- [ ] No hardcoded paths (use relative paths or config)
- [ ] Comments explain "why", not "what"
- [ ] Variable names are descriptive
- [ ] No unused imports or variables

---

## 📦 Pull Request Process

### 1. Commit Your Changes

```bash
# Stage changes
git add Autonomous_DrivingRL.py

# Commit with descriptive message
git commit -m "feat: Add curriculum learning support

- Implement progressive difficulty stages
- Add stage transition logic
- Update documentation"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat: Add prioritized experience replay

Implement PER to sample important transitions more frequently.
This improves sample efficiency by 20-30%.

Closes #42
```

```
fix: Resolve car spinning issue

Reduce angle reward weight from 1.5 to 0.5 to prevent
excessive rotation behavior.

Fixes #38
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

- Provide clear title and description
- Reference related issues
- Include screenshots/videos for UI changes
- List breaking changes (if any)

### 4. Address Review Comments

- Be responsive to feedback
- Make requested changes
- Push updates to the same branch

---

## 🎨 Feature Development Guidelines

### Adding a New RL Algorithm (e.g., SAC)

1. **Create new file**: `sac_implementation.py`
2. **Implement core components**:
   - Actor network
   - Critic network
   - Entropy regularization
   - Training loop
3. **Add tests**: `tests/test_sac.py`
4. **Update documentation**: Add to `ARCHITECTURE.md`
5. **Provide example**: Show how to use it

### Adding a New Reward Component

1. **Implement in `step()` method**
2. **Add configuration parameter**:
   ```python
   # At top of file
   REWARD_SMOOTHNESS_WEIGHT = 0.1
   
   # In step()
   smoothness_reward = -abs(steering) * REWARD_SMOOTHNESS_WEIGHT
   reward += smoothness_reward
   ```
3. **Document in `TRAINING.md`**
4. **Add ablation study results**

### Adding a New Visualization

1. **Create widget class**:
   ```python
   class SensorVisualization(QWidget):
       def __init__(self):
           super().__init__()
           # ...
       
       def paintEvent(self, event):
           # Draw sensor rays
           pass
   ```
2. **Integrate into main UI**
3. **Add toggle option** (optional)
4. **Update `USAGE.md`**

---

## 🐛 Bug Fix Guidelines

### 1. Reproduce the Bug

- Write a test that fails due to the bug
- Document exact steps to reproduce

### 2. Identify Root Cause

- Use debugger or print statements
- Check related code sections
- Review recent changes

### 3. Implement Fix

- Make minimal changes
- Ensure fix doesn't break other functionality
- Add regression test

### 4. Verify Fix

- Run all tests
- Test manually in application
- Check for edge cases

---

## 📚 Documentation Guidelines

### Code Comments

```python
# Good: Explain WHY
# Use Polyak averaging to slowly update target networks
# This stabilizes training by providing consistent targets
target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Bad: Explain WHAT (obvious from code)
# Copy data to target parameter
target_param.data.copy_(...)
```

### Docstrings

```python
def compute_reward(self, distance: float, angle: float, collision: bool) -> float:
    """
    Compute reward for current state-action pair.
    
    Uses potential-based reward shaping (PBRS) to provide dense feedback
    while preserving optimal policy. Includes anti-orbiting mechanism to
    prevent local minimum where agent circles the target.
    
    Args:
        distance: Euclidean distance to target in pixels
        angle: Relative angle to target in degrees [-180, 180]
        collision: Whether car collided with obstacle
    
    Returns:
        Scalar reward value. Positive for progress, negative for crashes.
        
    References:
        Ng et al. (1999) - Policy Invariance Under Reward Transformations
    """
    pass
```

### README Updates

When adding features, update:
- Feature list
- Quick start guide (if applicable)
- Hyperparameters table (if new params added)
- Architecture diagram (if structure changed)

---

## 🔬 Experiment Guidelines

### Running Experiments

```python
# Create experiment directory
experiments/
├── baseline/
│   ├── config.json
│   ├── training_log.csv
│   └── model.pth
├── with_her/
│   ├── config.json
│   ├── training_log.csv
│   └── model.pth
└── results.md
```

### Documenting Results

```markdown
# Experiment: Hindsight Experience Replay

## Hypothesis
Adding HER will improve sample efficiency by 30%.

## Setup
- Baseline: Standard TD3
- Treatment: TD3 + HER
- Episodes: 1000 each
- Seeds: 5 random seeds
- Map: city_map.png

## Results
| Method | Avg Reward | Success Rate | Episodes to Converge |
|--------|-----------|--------------|---------------------|
| Baseline | 85.3 ± 12.1 | 72% | 650 |
| TD3 + HER | 112.7 ± 8.4 | 89% | 420 |

## Conclusion
HER improves sample efficiency by 35% and increases success rate by 17%.
Recommend integrating into main codebase.
```

---

## 🎓 Learning Resources for Contributors

### Deep RL Background

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Sutton & Barto - RL Book](http://incompleteideas.net/book/)
- [Berkeley CS285](http://rail.eecs.berkeley.edu/deeprlcourse/)

### PyTorch

- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### PyQt6

- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [Qt for Python](https://doc.qt.io/qtforpython/)

---

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md (Contributors section)
- Release notes
- Documentation

---

## 📧 Communication

### Questions?

- Open a GitHub Discussion (if available)
- Comment on related issues
- Email maintainers (if provided)

### Reporting Security Issues

Please report security vulnerabilities privately to maintainers, not in public issues.

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Last Updated**: January 15, 2026
