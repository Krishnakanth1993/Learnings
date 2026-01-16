import numpy as np
import time
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os

class GridWorld:
    def __init__(self, size=4, gamma=1.0, theta=1e-4):
        """
        Initialize GridWorld environment
        
        Args:
            size: Grid dimension (NxN)
            gamma: Discount factor
            theta: Convergence threshold
        """
        self.size = size
        self.n_states = size * size
        self.gamma = gamma
        self.theta = theta
        
        # Terminal state is bottom-right corner
        self.terminal_state = self.n_states - 1
        
        # Initialize value function to 0 for all states
        self.V = np.zeros(self.n_states)
        
        # Define actions: up, down, left, right
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        
        # Reward is -1 for each move, 0 for terminal state
        self.rewards = np.full(self.n_states, -1.0)
        self.rewards[self.terminal_state] = 0.0
        
        # Track iteration history for visualization
        self.history = []
        
    def get_next_state(self, state, action):
        """
        Get next state given current state and action
        Handles grid boundaries
        
        Args:
            state: Current state (0 to n_states-1)
            action: Action to take ('up', 'down', 'left', 'right')
            
        Returns:
            next_state: Resulting state after action
        """
        # Convert state to (row, col)
        row = state // self.size
        col = state % self.size
        
        # Apply action
        if action == 'up':
            row = max(0, row - 1)
        elif action == 'down':
            row = min(self.size - 1, row + 1)
        elif action == 'left':
            col = max(0, col - 1)
        elif action == 'right':
            col = min(self.size - 1, col + 1)
        
        # Convert back to state
        next_state = row * self.size + col
        return next_state
    
    def value_iteration(self, max_iterations=1000, delay=0.1):
        """
        Perform value iteration using Bellman equation
        
        Args:
            max_iterations: Maximum number of iterations
            delay: Delay between iterations for visualization (seconds)
            
        Returns:
            V: Final value function
            iterations: Number of iterations until convergence
        """
        print(f"Starting Value Iteration for {self.size}x{self.size} GridWorld")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        print(f"Terminal State: {self.terminal_state}")
        print("-" * 60)
        
        iteration = 0
        
        while iteration < max_iterations:
            # Track maximum change in value function
            delta = 0.0
            
            # Create a copy of current value function
            V_new = self.V.copy()
            
            # For each state (excluding terminal state)
            for s in range(self.n_states):
                if s == self.terminal_state:
                    continue
                
                # Compute new value using Bellman equation
                # V(s) = sum over actions [P(a) * sum over s' [P(s'|s,a) * (R + gamma * V(s'))]]
                # Since policy is uniform random, P(a) = 1/n_actions for all actions
                
                v = 0.0
                for action in self.actions:
                    # Get next state
                    next_state = self.get_next_state(s, action)
                    
                    # Transition probability is 1.0 (deterministic given action)
                    # Reward for taking action from state s
                    reward = self.rewards[s]
                    
                    # Expected value for this action
                    action_value = reward + self.gamma * self.V[next_state]
                    
                    # Add to total (weighted by uniform policy probability)
                    v += (1.0 / self.n_actions) * action_value
                
                # Update new value
                V_new[s] = v
                
                # Track maximum change
                delta = max(delta, abs(V_new[s] - self.V[s]))
            
            # Update value function
            self.V = V_new.copy()
            
            # Store history for visualization
            self.history.append({
                'iteration': iteration,
                'V': self.V.copy().tolist(),
                'delta': float(delta)
            })
            
            # Print progress
            print(f"Iteration {iteration:4d} | Delta: {delta:.6f} | Max V: {self.V.max():.4f} | Min V: {self.V.min():.4f}")
            
            iteration += 1
            
            # Add delay for visualization
            time.sleep(delay)
            
            # Check convergence
            if delta < self.theta:
                print("-" * 60)
                print(f"Converged after {iteration} iterations!")
                break
        
        if iteration == max_iterations:
            print("-" * 60)
            print(f"Reached maximum iterations ({max_iterations})")
        
        return self.V, iteration
    
    def extract_policy(self):
        """
        Extract optimal policy from converged value function
        For each state, compute Q-values for all actions and determine best action(s)
        
        Returns:
            policy: Dictionary with state -> action probabilities
            action_values: Dictionary with state -> action Q-values
        """
        policy = {}
        action_values = {}
        
        for s in range(self.n_states):
            if s == self.terminal_state:
                # Terminal state has no actions
                policy[s] = {action: 0.0 for action in self.actions}
                action_values[s] = {action: 0.0 for action in self.actions}
                continue
            
            # Compute Q-value for each action
            q_values = {}
            for action in self.actions:
                next_state = self.get_next_state(s, action)
                reward = self.rewards[s]
                q_values[action] = reward + self.gamma * self.V[next_state]
            
            action_values[s] = q_values
            
            # Find best action(s) - there may be ties
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if abs(q - max_q) < 1e-6]
            
            # Create policy distribution (uniform over best actions)
            policy_dist = {action: 0.0 for action in self.actions}
            for action in best_actions:
                policy_dist[action] = 1.0 / len(best_actions)
            
            policy[s] = policy_dist
        
        return policy, action_values
    
    def print_policy(self, policy):
        """Print the optimal policy as a grid with arrows"""
        # Use ASCII characters for console compatibility
        action_symbols = {
            'up': 'U',
            'down': 'D',
            'left': 'L',
            'right': 'R'
        }
        
        print("\nOptimal Policy:")
        print("=" * 60)
        for row in range(self.size):
            row_policies = []
            for col in range(self.size):
                state = row * self.size + col
                if state == self.terminal_state:
                    row_policies.append("  GOAL  ")
                else:
                    # Get best actions
                    best_actions = [action for action, prob in policy[state].items() if prob > 0]
                    symbols = ''.join([action_symbols[action] for action in best_actions])
                    row_policies.append(f"  {symbols:^4}  ")
            print(" | ".join(row_policies))
            if row < self.size - 1:
                print("-" * 60)
        print("=" * 60)
    
    def print_value_function(self):
        """Print the value function as a grid"""
        print("\nFinal Value Function:")
        print("=" * 60)
        for row in range(self.size):
            row_values = []
            for col in range(self.size):
                state = row * self.size + col
                row_values.append(f"{self.V[state]:7.3f}")
            print(" | ".join(row_values))
            if row < self.size - 1:
                print("-" * 60)
        print("=" * 60)
    
    def save_history(self, filename='gridworld_history.json'):
        """Save iteration history to JSON file"""
        # Extract final policy
        policy, action_values = self.extract_policy()
        
        # Convert policy to serializable format
        policy_data = {}
        action_values_data = {}
        
        for state in range(self.n_states):
            policy_data[str(state)] = policy[state]
            action_values_data[str(state)] = action_values[state]
        
        data = {
            'size': self.size,
            'gamma': self.gamma,
            'theta': self.theta,
            'terminal_state': self.terminal_state,
            'history': self.history,
            'final_policy': policy_data,
            'action_values': action_values_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"\nHistory saved to {filename}")


def main():
    # Create GridWorld
    grid = GridWorld(size=4, gamma=1.0, theta=1e-4)
    
    # Run value iteration with delay for visualization
    V_final, iterations = grid.value_iteration(max_iterations=1000, delay=0.05)
    
    # Print final value function
    grid.print_value_function()
    
    # Extract and print optimal policy
    policy, action_values = grid.extract_policy()
    grid.print_policy(policy)
    
    # Save history for web visualization
    grid.save_history('gridworld_history.json')
    
    print("\n" + "=" * 60)
    print("Value iteration complete!")
    print(f"Total iterations: {iterations}")
    print("=" * 60)



if __name__ == "__main__":
    main()
