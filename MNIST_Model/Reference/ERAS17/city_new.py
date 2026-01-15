
"""
===============================================================================
ASSIGNMENT: FIX THE AUTONOMOUS CAR NAVIGATION
===============================================================================

Welcome Students!

This code implements a self-driving car using Deep Q-Network (DQN) reinforcement learning.
However, several critical parameters have been intentionally set to INCORRECT values
that will prevent the car from learning properly.

YOUR TASK:
Find and fix all parameters marked with "FIX ME" comments. Use your understanding
of reinforcement learning, neural networks, and the physics of car navigation to
set appropriate values.

HINTS:
- Read the comments carefully - they explain what each parameter does
- Think about reasonable ranges (e.g., learning rates are usually 0.0001 to 0.01)
- Consider the physics (can a car sensor realistically be 1000 pixels away?)
- Test your fixes by running the program and observing the learning behavior

GRADING CRITERIA:
1. Car successfully learns to navigate (primary goal)
2. Appropriate hyperparameter values chosen
3. Understanding demonstrated in comments you add

Good luck! 🚗💨
===============================================================================
"""

import sys
import os
import math
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C") # Green
C_SENSOR_OFF= QColor("#BF616A") # Red

# ==========================================
# PHYSICS PARAMETERS - FIX ME!
# ==========================================
# CAR_WIDTH = 10     
# CAR_HEIGHT = 15  
# SENSOR_DIST = 40  # FIX ME! Distance sensors look ahead (pixels) - Currently unrealistic!
# SENSOR_ANGLE = 45    # FIX ME! Angle spread of sensors (degrees) - Too narrow!
# SPEED = 4          # FIX ME! Forward speed (pixels/step) - Way too fast!
# TURN_SPEED = 0.2  # FIX ME! Regular turn angle (degrees/step) - Too slow! --0.5
# SHARP_TURN = 15      # FIX ME! Sharp turn angle for tight corners (degrees) - Too small!

CAR_WIDTH = 10
CAR_HEIGHT = 15

SENSOR_DIST = 60
# Reason: Sensors must see far enough ahead to anticipate walls at speed.
# 40 px is borderline; 60–80 px is typical for a 1000×800 map.

SENSOR_ANGLE = 90
# Reason: Total spread of ±45° is reasonable for forward-facing perception.
# Too narrow reduces obstacle anticipation.

SPEED = 2.0
# Reason: 4 px/step causes frequent tunneling through obstacles.
# 1.5–2.5 px/step balances stability and learning signal.

TURN_SPEED = 2.0
# Reason: 0.2°/step is effectively no steering.
# Continuous control needs a few degrees per step to correct heading.

SHARP_TURN = 10
# Reason: 15° is insufficient for tight corners.
# 20–30° allows recovery without oscillation.


# ==========================================
# REINFORCEMENT LEARNING HYPERPARAMETERS - FIX ME!
# ==========================================
BATCH_SIZE = 64    # FIX ME! Number of experiences sampled per training step - Too small!
                    # Hint: Typically 32-512 for stability

GAMMA = 0.99        # FIX ME! Discount factor for future rewards (0 to 1)
                    # Too low - agent won't care about future!
                    # Hint: Usually 0.9-0.99 ---0.99

LR = 0.0003           # FIX ME! Learning rate for neural network optimizer
                    # Way too high - will cause instability!
                    # Hint: Usually 0.0001 to 0.01

TAU = 0.003           # FIX ME! Polyak averaging coefficient for soft target updates
                    # Too high - defeats the purpose of target network!
                    # Hint: Usually 0.001 to 0.01   --- 0.003

MAX_CONSECUTIVE_CRASHES =3 # FIX ME! Reset after this many crashes
                               # Too high - will waste time on bad configs
                               # Hint: Usually 2-10

# Target Colors (for multiple targets)
TARGET_COLORS = [
    QColor(0, 255, 255),      # Cyan
    QColor(255, 100, 255),    # Magenta
    QColor(0, 255, 100),      # Green
    QColor(255, 150, 0),      # Orange
    QColor(100, 150, 255),    # Blue
    QColor(255, 50, 150),     # Pink
    QColor(150, 255, 50),     # Lime
    QColor(255, 255, 0),      # Yellow
]

# ==========================================
# 2. NEURAL NETWORK (TD3 ARCHITECTURE)
# ==========================================

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.asarray(state))
            batch_next_states.append(np.asarray(next_state))
            batch_actions.append(np.asarray(action))
            batch_rewards.append(np.asarray(reward))
            batch_dones.append(np.asarray(done))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# ==========================================
# 3. PHYSICS & LOGIC
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # RL Init - TD3
        self.input_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        self.action_dim = 1 # Steering angle
        self.max_action = 25.0 # Max steering angle in degrees
        
        self.actor = Actor(self.input_dim, self.action_dim, self.max_action)
        self.actor_target = Actor(self.input_dim, self.action_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        
        self.critic = Critic(self.input_dim, self.action_dim)
        self.critic_target = Critic(self.input_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
        self.replay_buffer = ReplayBuffer(max_size=100000)
        
        self.steps = 0
        self.consecutive_crashes = 0
        self.episode_scores = deque(maxlen=100) # Added back for UI stats
        
        # Dummy attributes to prevent UI crashes if referenced, though logic is changed
        self.priority_memory = [] 
        self.memory = []
        
        # TD3 Hyperparameters
        self.policy_noise = 3.0  # Noise added to target policy during training
        self.noise_clip = 15.0   # Range to clip target policy noise
        self.policy_freq = 2     # Update target networks every step (Continuous correction)
        self.expl_noise = 5    # Initial exploration noise (std dev)
        
        # Locations
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200)
        
        # Multiple Targets Support
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        return state
    
    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []

        angles = [-45, -30, -15, 0, 15, 30, 45]

        for a in angles:
            rad = math.radians(self.car_angle + a)
            hit_dist = SENSOR_DIST
            hit_point = None

            for d in range(5, SENSOR_DIST + 1, 5):
                sx = self.car_pos.x() + math.cos(rad) * d
                sy = self.car_pos.y() + math.sin(rad) * d

                if not (0 <= sx < self.w and 0 <= sy < self.h):
                    hit_dist = d
                    hit_point = QPointF(sx, sy)
                    break

                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0 / 255.0

                if brightness < 0.8:  # wall
                    hit_dist = d
                    hit_point = QPointF(sx, sy)
                    break

            if hit_point is None:
                hit_point = QPointF(
                    self.car_pos.x() + math.cos(rad) * SENSOR_DIST,
                    self.car_pos.y() + math.sin(rad) * SENSOR_DIST
                )

            self.sensor_coords.append(hit_point)

            # Normalize: 1.0 = very close obstacle, 0.0 = no obstacle
            sensor_vals.append(1.0 - (hit_dist / SENSOR_DIST))

        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.hypot(dx, dy)

        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360

        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0

        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist


    def step(self, action):
        turn = float(action)
        turn = np.clip(turn, -5.0, 5.0)  # physical steering limit per step
        self.car_angle += turn

        
        rad = math.radians(self.car_angle)
        
        new_x = self.car_pos.x() + math.cos(rad) * SPEED
        new_y = self.car_pos.y() + math.sin(rad) * SPEED
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        sensors = next_state[:7]
        
        reward = -0.1
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.8:
            reward = -100
            done = True
            self.alive = False
        elif dist < 20: 
            reward = 100
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            # Reward for moving closer to target
            if self.prev_dist is not None:
                reward += (self.prev_dist - dist) * 2.0
            
            # Small penalty for alignment to encourage facing target
            # next_state[7] is normalized angle diff [-1, 1]
            # We want to reward when angle diff is close to 0, and PENALIZE when facing away
            # (0.5 - abs(x)) gives range [-0.5, 0.5]. Multiplied by 2.0 gives [-1.0, 1.0]
            #reward += (0.5 - abs(next_state[7])) * 2.0
            angle_norm = next_state[7]   # already in [-1, 1]
            reward += max(0.0, 1.0 - abs(angle_norm)) * 1.5
            reward -= abs(angle_norm) * 0.5 
            
            self.prev_dist = dist
            
        self.score += reward
        
        if self.score < -500:
            done = True
            reward -= 50
            
        return next_state, reward, done

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        return action[0]

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def optimize(self):
        if len(self.replay_buffer.storage) < BATCH_SIZE: return 0
        
        # Sample replay buffer
        state, next_state, action, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action).reshape(-1, 1) # Ensure shape (Batch, 1)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        
        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        
        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * GAMMA * target_Q).detach()
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.steps % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
        return critic_loss.item()
    
    def store_experience(self, experience):
        # experience: (state, action, rew, next_s, done)
        self.replay_buffer.add(experience)
    
    def finalize_episode(self, episode_reward):
        # TD3 doesn't typically use the prioritized episode buffer logic in the same way as the original code,
        # but we can keep track of stats.
        self.episode_scores.append(episode_reward)
        
        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0

# ==========================================
# 4. CUSTOM WIDGETS (VISUALS)
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: max_val += 1
        
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(C_ACCENT, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                avg_pen = QPen(QColor(255, 215, 0), 3)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))
        
        legend_x = 10
        legend_y = 15
        
        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")
        
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")

class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        if self.is_detecting:
            color = C_SENSOR_ON
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active):
        self.is_active = active
        self.update()
    
    def set_color(self, color):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: self.growing = True
            
            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0,0), r, r)
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0,0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0,0), 6, 6)
        
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 5. APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav: ASSIGNMENT - Fix the Parameters!")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        lbl_title = QLabel("CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)
        
        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)

        vbox.addSpacing(15)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.val_eps = QLabel("1.00")
        self.val_eps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Expl. Noise:"), 0,0)
        sf_layout.addWidget(self.val_eps, 0,1)
        
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Last Reward:"), 1,0)
        sf_layout.addWidget(self.val_rew, 1,1)
        
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Logic
        self.setup_map("city_map.png") 
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded.")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: 
            self.full_reset()
            self.setup_map(f)

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt) 
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map -> TARGET(S)\nRight-click when done")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                target_idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)
                num_targets = len(self.brain.targets)
                
                target_item = TargetItem(color, is_active, num_targets)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish setup")
                self.log(f"Target #{num_targets} added at ({pt.x():.0f}, {pt.y():.0f})")
            
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        self.scene.removeItem(self.car_item)
        for target_item in self.target_items:
            self.scene.removeItem(target_item)
        self.target_items = []
        self.brain.targets = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        
        for s in self.sensor_items: 
            if s.scene() == self.scene: self.scene.removeItem(s)
        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("--- RESET ---")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        if self.setup_state != 2: return

        state, current_dist = self.brain.get_state()
        action = 0
        
        prev_target_idx = self.brain.current_target_idx
        
        if self.brain.steps < 1000: # Warmup steps
             action = random.uniform(-self.brain.max_action, self.brain.max_action)
        else:
            action = self.brain.select_action(state)
            if self.brain.expl_noise != 0:
                noise = np.random.normal(0, self.brain.expl_noise)
                action = (action + noise)
                action = max(-self.brain.max_action, min(self.brain.max_action, action))
                
                # Decay exploration noise
                if self.brain.expl_noise > 0.1:
                    self.brain.expl_noise *= 0.99995
        
        next_s, rew, done = self.brain.step(action)
        
        if self.brain.steps % 50 == 0:
            # Debug log
            angle_diff = state[7] * 180.0 # Denormalize
            
            dist_diff = 0.0
            if self.brain.prev_dist is not None:
                dist_diff = current_dist - self.brain.prev_dist
                
            self.log(f"Step {self.brain.steps}: AngleDiff={angle_diff:.1f}, Action={action:.2f}, Dist={current_dist:.1f}, DistDiff={dist_diff:.2f}")
        
        # Store experience
        self.brain.store_experience((state, next_s, action, rew, done))
        self.brain.optimize()
        
        if self.brain.current_target_idx != prev_target_idx:
            target_num = self.brain.current_target_idx + 1
            total = len(self.brain.targets)
            self.log(f"<font color='#88C0D0'>🎯 Target {prev_target_idx + 1} reached! Moving to target {target_num}/{total}</font>")
        
        # TD3 updates target networks inside optimize()
        # for target_param, policy_param in zip(self.brain.target_net.parameters(), 
        #                                        self.brain.policy_net.parameters()):
        #     target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
        
        self.brain.steps += 1
        
        if done:
            self.brain.finalize_episode(self.brain.score)
            
            should_reset_position = False
            
            if self.brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='#BF616A'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting to origin...</b></font>")
                self.log(f"<font color='#88C0D0'>💡 Tip: Adjust hyperparameters, simplify map, or increase exploration (noise)</font>")
                self.brain.consecutive_crashes = 0
                should_reset_position = True
            
            if not self.brain.alive:
                txt = f"CRASH ({self.brain.consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES})"
                col = "#BF616A"
            else:
                if self.brain.targets_reached == len(self.brain.targets) - 1:
                    txt = f"ALL {len(self.brain.targets)} TARGETS COMPLETED!"
                    col = "#A3BE8C"
                else:
                    txt = "GOAL"
                    col = "#A3BE8C"
                should_reset_position = True
            
            buffer_size = len(self.brain.replay_buffer.storage)
            max_size = self.brain.replay_buffer.max_size
            
            self.log(f"<font color='{col}'>{txt} (Scr: {self.brain.score:.0f}) | "
                    f"Buffer: {buffer_size}/{max_size} | "
                    f"Noise: {self.brain.expl_noise:.3f}</font>")
            
            self.chart.update_chart(self.brain.score)
            
            if should_reset_position:
                self.brain.reset()
            else:
                self.brain.score = 0
                self.brain.alive = True
                self.brain.current_target_idx = 0
                self.brain.targets_reached = 0
                if len(self.brain.targets) > 0:
                    self.brain.target_pos = self.brain.targets[0]
                _, dist = self.brain.get_state()
                self.brain.prev_dist = dist

        self.update_visuals()
        self.val_eps.setText(f"{self.brain.expl_noise:.3f}")
        self.val_rew.setText(f"{self.brain.score:.0f}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target_item.set_active(is_active)
        
        self.scene.update() 
        
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            s_val = self.brain.get_state()[0][i]
            self.sensor_items[i].set_detecting(s_val > 0.5)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec())