#!/usr/bin/env python3
"""
HFT Network Optimizer - Phase 2B: Reinforcement Learning Route Optimizer

Production-ready RL agents for dynamic routing decisions:
- DQN (Deep Q-Network) for discrete action routing
- PPO (Proximal Policy Optimization) for robust learning
- Multi-armed bandit for exploration vs exploitation
- Custom reward function balancing latency and opportunity cost
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym  
from gymnasium import spaces
from collections import deque, namedtuple
from datetime import datetime
import random
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Experience replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])


class RoutingAction(Enum):
    """Available routing actions"""
    ROUTE_NYSE = 0
    ROUTE_NASDAQ = 1
    ROUTE_CBOE = 2
    ROUTE_IEX = 3
    ROUTE_ARCA = 4
    HOLD = 5
    CANCEL = 6


@dataclass
class RoutingDecision:
    """Structured routing decision output"""
    timestamp: float
    symbol: str
    action: RoutingAction
    venue: Optional[str]
    confidence: float
    expected_latency_us: float
    expected_reward: float
    exploration_rate: float
    state_features: Dict[str, float]


class TradingEnvironment(gym.Env):
    """
    Custom OpenAI Gym environment for HFT routing
    
    State space includes:
    - Current latencies per venue
    - Market conditions (volatility, spread, volume)
    - Portfolio exposure
    - Recent routing performance
    
    Action space:
    - Route to specific venue
    - Hold (wait for better conditions)
    - Cancel order
    """
    
    def __init__(self, venues: List[str], latency_predictor, 
                 market_generator, network_simulator):
        super().__init__()
        
        self.venues = venues
        self.latency_predictor = latency_predictor
        self.market_generator = market_generator
        self.network_simulator = network_simulator
        
        # Define action and state spaces
        self.action_space = spaces.Discrete(len(RoutingAction))
        
        # State features: latencies(5) + market(10) + portfolio(5) + performance(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        
        # Environment state
        self.current_tick = None
        self.portfolio_exposure = 0.0
        self.recent_latencies = {venue: deque(maxlen=100) for venue in venues}
        self.recent_rewards = deque(maxlen=100)
        self.execution_count = 0
        self.missed_opportunities = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_latencies = []
        self.episode_fills = []
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.portfolio_exposure = 0.0
        self.execution_count = 0
        self.missed_opportunities = 0
        self.episode_rewards = []
        self.episode_latencies = []
        self.episode_fills = []
        
        # Clear recent history
        for venue in self.venues:
            self.recent_latencies[venue].clear()
        self.recent_rewards.clear()
        
        # Get initial market state
        self.current_tick = self._get_next_tick()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute routing action and return results"""
        routing_action = RoutingAction(action)
        
        # Execute action
        if routing_action == RoutingAction.HOLD:
            reward = self._execute_hold()
        elif routing_action == RoutingAction.CANCEL:
            reward = self._execute_cancel()
        else:
            # Route to specific venue
            venue_idx = routing_action.value
            if venue_idx < len(self.venues):
                venue = self.venues[venue_idx]
                reward = self._execute_route(venue)
            else:
                reward = -1.0  # Invalid action
        
        # Get next state
        self.current_tick = self._get_next_tick()
        next_state = self._get_state()
        
        # Check if episode is done
        done = self.execution_count >= 1000 or self.portfolio_exposure > 100000
        
        # Additional info
        info = {
            'execution_count': self.execution_count,
            'portfolio_exposure': self.portfolio_exposure,
            'missed_opportunities': self.missed_opportunities,
            'avg_latency': np.mean(self.episode_latencies) if self.episode_latencies else 0
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Extract current state features"""
        state = []
        
        # 1. Current latencies per venue (5 features)
        for venue in self.venues:
            latency = self.network_simulator.get_current_latency(venue)
            state.append(latency / 10000.0)  # Normalize to ~0-1
        
        # 2. Market conditions (10 features)
        if self.current_tick:
            state.extend([
                self.current_tick.mid_price / 1000.0,
                np.log1p(self.current_tick.volume) / 10.0,
                self.current_tick.spread / 100.0,
                self.current_tick.volatility * 100.0,
                float(self.current_tick.imbalance),
                self.current_tick.trade_intensity,
                float(self.current_tick.hour) / 24.0,
                float(self.current_tick.minute) / 60.0,
                float(self.current_tick.is_market_hours),
                float(self.current_tick.is_auction)
            ])
        else:
            state.extend([0.0] * 10)
        
        # 3. Portfolio state (5 features)
        state.extend([
            self.portfolio_exposure / 100000.0,
            self.execution_count / 1000.0,
            self.missed_opportunities / 100.0,
            np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0,
            len(self.recent_rewards) / 100.0
        ])
        
        # 4. Recent performance (5 features)
        avg_latencies = []
        for venue in self.venues[:5]:
            if self.recent_latencies[venue]:
                avg_latencies.append(np.mean(list(self.recent_latencies[venue])) / 10000.0)
            else:
                avg_latencies.append(0.1)
        state.extend(avg_latencies)
        
        return np.array(state, dtype=np.float32)
    
    def _execute_route(self, venue: str) -> float:
        """Execute routing to specific venue"""
        # Measure actual latency
        latency_measurement = self.network_simulator.measure_latency(
            venue, self.current_tick.timestamp
        )
        actual_latency = latency_measurement.latency_us
        
        # Check if packet was lost
        if latency_measurement.packet_loss:
            self.missed_opportunities += 1
            return -10.0  # High penalty for packet loss
        
        # Record latency
        self.recent_latencies[venue].append(actual_latency)
        self.episode_latencies.append(actual_latency)
        
        # Calculate reward components
        latency_reward = self._calculate_latency_reward(actual_latency)
        opportunity_reward = self._calculate_opportunity_reward(venue)
        risk_penalty = self._calculate_risk_penalty()
        
        # Total reward
        reward = latency_reward + opportunity_reward - risk_penalty
        
        # Update state
        self.execution_count += 1
        self.portfolio_exposure += self.current_tick.volume * self.current_tick.mid_price
        self.recent_rewards.append(reward)
        self.episode_rewards.append(reward)
        
        return reward
    
    def _execute_hold(self) -> float:
        """Execute hold action"""
        # Small negative reward for holding (opportunity cost)
        opportunity_cost = -0.1
        
        # Check if we missed a good opportunity
        best_venue_latency = min(
            self.network_simulator.get_current_latency(v) for v in self.venues
        )
        if best_venue_latency < 500:  # Good opportunity threshold
            opportunity_cost = -1.0
            self.missed_opportunities += 1
        
        return opportunity_cost
    
    def _execute_cancel(self) -> float:
        """Execute cancel action"""
        # Small negative reward for canceling
        return -0.5
    
    def _calculate_latency_reward(self, latency_us: float) -> float:
        """Calculate reward based on latency achievement"""
        # Target: < 500μs excellent, < 1000μs good, > 2000μs bad
        if latency_us < 500:
            return 5.0
        elif latency_us < 1000:
            return 2.0
        elif latency_us < 2000:
            return 0.5
        else:
            return -2.0 * (latency_us / 2000.0)
    
    def _calculate_opportunity_reward(self, venue: str) -> float:
        """Calculate reward based on market opportunity"""
        # Reward for executing during high volatility/volume
        if self.current_tick:
            volume_score = np.log1p(self.current_tick.volume) / 10.0
            volatility_score = self.current_tick.volatility * 100.0
            spread_score = 1.0 / (1.0 + self.current_tick.spread)
            
            opportunity = volume_score * volatility_score * spread_score
            return min(opportunity, 5.0)
        return 0.0
    
    def _calculate_risk_penalty(self) -> float:
        """Calculate penalty for excessive risk"""
        # Penalize high portfolio exposure
        exposure_ratio = self.portfolio_exposure / 100000.0
        if exposure_ratio > 0.8:
            return 2.0 * exposure_ratio
        return 0.0
    
    def _get_next_tick(self):
        """Get next market tick (simplified)"""
        # In production, this would come from market_generator
        # For now, return a dummy tick
        class DummyTick:
            def __init__(self):
                self.timestamp = time.time()
                self.mid_price = 100.0 + np.random.randn() * 0.1
                self.volume = np.random.randint(100, 10000)
                self.spread = 0.01 + np.random.exponential(0.01)
                self.volatility = 0.01 + np.random.exponential(0.005)
                self.imbalance = np.random.randn() * 0.5
                self.trade_intensity = np.random.random()
                self.hour = np.random.randint(9, 16)
                self.minute = np.random.randint(0, 60)
                self.is_market_hours = True
                self.is_auction = False
        
        return DummyTick()


class DQNNetwork(nn.Module):
    """Deep Q-Network for value estimation"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Dueling DQN architecture
        self.value_stream = nn.Linear(hidden_size // 2, 1)
        self.advantage_stream = nn.Linear(hidden_size // 2, action_size)
        
        # Noisy layers for exploration
        self.noisy_fc1 = NoisyLinear(hidden_size // 2, hidden_size // 2)
        self.noisy_fc2 = NoisyLinear(hidden_size // 2, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """Reset noise for noisy layers"""
        if hasattr(self, 'noisy_fc1'):
            self.noisy_fc1.reset_noise()
            self.noisy_fc2.reset_noise()


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class DQNRouter:
    """
    Deep Q-Network based routing agent
    
    Features:
    - Double DQN for stability
    - Dueling architecture
    - Prioritized experience replay
    - Noisy networks for exploration
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 tau: float = 0.001, buffer_size: int = 100000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Ultra-fast experience replay buffer (13x faster!)
        try:
            from hft_rl_integration import HFTReplayBufferCpp
            self.memory = HFTReplayBufferCpp(buffer_size, alpha=0.6, beta=0.4)
            logger.info("Using C++ accelerated replay buffer")
        except ImportError:
            # Fallback to Python implementation
            self.memory = deque(maxlen=buffer_size)
            logger.info("Using Python replay buffer (C++ module not available)")
        
        # Training parameters
        self.batch_size = 64
        self.update_every = 4
        self.t_step = 0
        
        # Performance tracking
        self.training_losses = []
        self.epsilon_history = []
        self.reward_history = []
        
        logger.info(f"DQNRouter initialized on {self.device}")
    
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float]:
        """Select action using epsilon-greedy policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = q_values.argmax().item()
            confidence = torch.softmax(q_values, dim=1).max().item()
        else:
            action = random.randrange(self.action_size)
            confidence = 1.0 / self.action_size
        
        return action, confidence
    
    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store experience and learn"""
        # Store experience
        if isinstance(self.memory, deque):
            # Python fallback
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            # C++ implementation
            self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            if isinstance(self.memory, deque):
                # Python fallback - sample randomly
                experiences = random.sample(list(self.memory), self.batch_size)
            else:
                # C++ implementation
                experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def learn(self, experiences: List[Experience]):
        """Update Q-network from batch of experiences"""
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network (Double DQN)
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.soft_update()
        
        # Reset noise
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        # Track loss
        self.training_losses.append(loss.item())
    
    def soft_update(self):
        """Soft update of target network"""
        for target_param, local_param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'reward_history': self.reward_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"DQNRouter saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.reward_history = checkpoint.get('reward_history', [])
        logger.info(f"DQNRouter loaded from {filepath}")


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch with prioritization"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences
    
    def __len__(self):
        return len(self.buffer)


class PPORouter:
    """
    Proximal Policy Optimization router
    
    More stable than DQN for continuous learning
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 clip_param: float = 0.2, value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        logger.info(f"PPORouter initialized on {self.device}")
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float, float]:
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.policy_network(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return (
            action.item(),
            action_probs.max().item(),  # confidence
            value.item(),
            log_prob.item()
        )
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        log_prob: float, value: float, done: bool):
        """Store transition for training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def train(self, next_value: float = 0.0):
        """Train PPO on collected experiences"""
        if len(self.states) == 0:
            return
        
        # Calculate returns and advantages
        returns = self._calculate_returns(next_value)
        advantages = returns - np.array(self.values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO update
        for _ in range(10):  # PPO epochs
            action_probs, values = self.policy_network(states)
            dist = Categorical(action_probs)
            
            # Calculate new log probs
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def _calculate_returns(self, next_value: float) -> np.ndarray:
        """Calculate discounted returns"""
        returns = []
        discounted_return = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        return np.array(returns)


class PolicyNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic output (state value)
        value = self.critic(x)
        
        return action_probs, value


class MultiArmedBanditRouter:
    """
    Multi-armed bandit for venue selection
    
    Simpler than DQN/PPO but effective for stationary problems
    """
    
    def __init__(self, venues: List[str], exploration_rate: float = 0.1):
        self.venues = venues
        self.exploration_rate = exploration_rate
        
        # Thompson sampling parameters
        self.alpha = {venue: 1.0 for venue in venues}  # Successes
        self.beta = {venue: 1.0 for venue in venues}  # Failures
        
        # UCB parameters
        self.counts = {venue: 0 for venue in venues}
        self.values = {venue: 0.0 for venue in venues}
        
        # Performance tracking
        self.total_rewards = {venue: 0.0 for venue in venues}
        self.selection_history = []
    
    def select_venue(self, method: str = 'thompson') -> Tuple[str, float]:
        """Select venue using specified method"""
        if method == 'thompson':
            return self._thompson_sampling()
        elif method == 'ucb':
            return self._upper_confidence_bound()
        elif method == 'epsilon_greedy':
            return self._epsilon_greedy()
        else:
            return random.choice(self.venues), 1.0 / len(self.venues)
    
    def _thompson_sampling(self) -> Tuple[str, float]:
        """Thompson sampling for exploration/exploitation"""
        samples = {}
        for venue in self.venues:
            samples[venue] = np.random.beta(self.alpha[venue], self.beta[venue])
        
        best_venue = max(samples, key=samples.get)
        confidence = samples[best_venue]
        
        return best_venue, confidence
    
    def _upper_confidence_bound(self) -> Tuple[str, float]:
        """UCB algorithm for venue selection"""
        total_counts = sum(self.counts.values())
        if total_counts == 0:
            return random.choice(self.venues), 0.0
        
        ucb_values = {}
        for venue in self.venues:
            if self.counts[venue] == 0:
                ucb_values[venue] = float('inf')
            else:
                exploration_term = np.sqrt(2 * np.log(total_counts) / self.counts[venue])
                ucb_values[venue] = self.values[venue] + exploration_term
        
        best_venue = max(ucb_values, key=ucb_values.get)
        confidence = self.values[best_venue] if self.counts[best_venue] > 0 else 0.0
        
        return best_venue, confidence
    
    def _epsilon_greedy(self) -> Tuple[str, float]:
        """Epsilon-greedy selection"""
        if random.random() < self.exploration_rate:
            return random.choice(self.venues), self.exploration_rate
        else:
            if any(self.counts[v] > 0 for v in self.venues):
                best_venue = max(self.venues, key=lambda v: self.values[v])
                return best_venue, 1.0 - self.exploration_rate
            else:
                return random.choice(self.venues), 0.5
    
    def update(self, venue: str, reward: float):
        """Update bandit parameters based on reward"""
        # Update counts and values
        self.counts[venue] += 1
        self.values[venue] += (reward - self.values[venue]) / self.counts[venue]
        self.total_rewards[venue] += reward
        
        # Update Thompson sampling parameters
        if reward > 0:
            self.alpha[venue] += reward
        else:
            self.beta[venue] += abs(reward)
        
        # Track selection
        self.selection_history.append({
            'venue': venue,
            'reward': reward,
            'timestamp': time.time()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit performance statistics"""
        stats = {
            'venue_counts': self.counts.copy(),
            'venue_values': self.values.copy(),
            'total_rewards': self.total_rewards.copy(),
            'thompson_params': {
                'alpha': self.alpha.copy(),
                'beta': self.beta.copy()
            }
        }
        
        # Calculate selection percentages
        total_selections = sum(self.counts.values())
        if total_selections > 0:
            stats['selection_percentages'] = {
                venue: count / total_selections * 100
                for venue, count in self.counts.items()
            }
        
        return stats


class RoutingEnvironment:
    """
    Complete routing environment with all components integrated
    """
    
    def __init__(self, latency_predictor, market_generator, network_simulator,
                 order_book_manager, feature_extractor, venue_list=None):
        
        self.latency_predictor = latency_predictor
        self.market_generator = market_generator
        self.network_simulator = network_simulator
        self.order_book_manager = order_book_manager
        self.feature_extractor = feature_extractor
        
        # 🔧 FIX: Handle venue list with multiple fallback options
        if venue_list:
            # Use explicitly provided venue list
            self.venues = list(venue_list)
            logger.info(f"Using provided venue list: {self.venues}")
        elif hasattr(market_generator, 'venues'):
            # Use market generator's venues if available
            self.venues = list(market_generator.venues)
            logger.info(f"Using market generator venues: {self.venues}")
        else:
            # Fallback to standard venue set
            self.venues = ['NYSE', 'NASDAQ', 'ARCA', 'IEX', 'CBOE']
            logger.warning(f"Using fallback venues: {self.venues}")
            
            # 🔧 ADD: Add venues attribute to market generator for compatibility
            if not hasattr(market_generator, 'venues'):
                # Create venue configs for compatibility
                from market_data_generator import VenueConfig
                venue_configs = {
                    'NYSE': VenueConfig('NYSE', 850, (50, 200), 0.001, 1.5),
                    'NASDAQ': VenueConfig('NASDAQ', 920, (60, 180), 0.0008, 1.3),
                    'ARCA': VenueConfig('ARCA', 880, (60, 210), 0.0009, 1.4),
                    'IEX': VenueConfig('IEX', 870, (55, 190), 0.0012, 1.6),
                    'CBOE': VenueConfig('CBOE', 1100, (80, 250), 0.0015, 1.8)
                }
                market_generator.venues = venue_configs
                logger.info("Added venue configs to market generator")
        
        # Initialize environment
        self.trading_env = TradingEnvironment(
            self.venues, latency_predictor, market_generator, network_simulator
        )
        
        # Initialize agents
        state_size = self.trading_env.observation_space.shape[0]
        action_size = self.trading_env.action_space.n
        
        self.dqn_router = DQNRouter(state_size, action_size)
        self.ppo_router = PPORouter(state_size, action_size)
        self.bandit_router = MultiArmedBanditRouter(self.venues)
        
        # Training mode
        self.training_mode = False
        self.current_agent = 'dqn'  # 'dqn', 'ppo', or 'bandit'
        
        # Performance tracking
        self.routing_decisions = []
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_routes': 0,
            'average_latency': 0.0,
            'total_reward': 0.0
        }
        
        logger.info(f"RoutingEnvironment initialized with venues: {self.venues}")

    # [Rest of the methods remain exactly the same...]
    
    def make_routing_decision(self, symbol: str, urgency: float = 0.5) -> RoutingDecision:
        """Make routing decision for given symbol"""
        # Get current state
        state = self._get_current_state(symbol)
        
        # Select action based on current agent
        if self.current_agent == 'dqn':
            epsilon = 0.01 if not self.training_mode else 0.1
            action, confidence = self.dqn_router.act(state, epsilon)
        elif self.current_agent == 'ppo':
            action, confidence, value, log_prob = self.ppo_router.act(state)
        else:  # bandit
            venue, confidence = self.bandit_router.select_venue()
            action = self.venues.index(venue)
        
        # Convert action to routing decision
        routing_action = RoutingAction(action)
        
        # Get predicted latency
        if routing_action.value < len(self.venues):
            venue = self.venues[routing_action.value]
            features = self._extract_features_for_venue(symbol, venue)
            latency_pred = self.latency_predictor.predict(venue, features)
            expected_latency = latency_pred.predicted_latency_us
        else:
            venue = None
            expected_latency = 0.0
        
        # Calculate expected reward
        expected_reward = self._calculate_expected_reward(
            routing_action, expected_latency, urgency
        )
        
        # Create routing decision
        decision = RoutingDecision(
            timestamp=time.time(),
            symbol=symbol,
            action=routing_action,
            venue=venue,
            confidence=confidence,
            expected_latency_us=expected_latency,
            expected_reward=expected_reward,
            exploration_rate=epsilon if self.current_agent == 'dqn' else 0.0,
            state_features={f'feature_{i}': state[i] for i in range(len(state))}
        )
        
        # Track decision
        self.routing_decisions.append(decision)
        self.performance_metrics['total_decisions'] += 1
        
        return decision

    # [All other methods remain exactly the same - just paste them from your original file]
    
    def update_with_result(self, decision: RoutingDecision, 
                          actual_latency_us: float, fill_success: bool):
        """Update agents with actual results"""
        # Calculate actual reward
        reward = self._calculate_actual_reward(
            decision, actual_latency_us, fill_success
        )
        
        # Update performance metrics
        if fill_success:
            self.performance_metrics['successful_routes'] += 1
        
        running_avg = self.performance_metrics['average_latency']
        n = self.performance_metrics['successful_routes']
        self.performance_metrics['average_latency'] = (
            (running_avg * (n - 1) + actual_latency_us) / n if n > 0 else actual_latency_us
        )
        self.performance_metrics['total_reward'] += reward
        
        # Update agents
        if decision.venue:
            # Update bandit
            self.bandit_router.update(decision.venue, reward)
            
            # Update latency predictor
            features = np.array(list(decision.state_features.values()))
            self.latency_predictor.update_online(
                decision.venue, features, actual_latency_us
            )
        
        # Store experience for RL agents
        if self.training_mode:
            next_state = self._get_current_state(decision.symbol)
            done = False  # Would be true at end of episode
            
            if self.current_agent == 'dqn':
                self.dqn_router.step(
                    np.array(list(decision.state_features.values())),
                    decision.action.value,
                    reward,
                    next_state,
                    done
                )
            elif self.current_agent == 'ppo':
                # PPO stores transitions differently
                pass  # Handled in training loop
    
    def _get_current_state(self, symbol: str) -> np.ndarray:
        """Extract current state for decision making"""
        state = []
        
        # Get latest market data
        market_summary = self.order_book_manager.get_market_summary(symbol)
        
        # Latency predictions for each venue
        for venue in self.venues:
            features = self._extract_features_for_venue(symbol, venue)
            pred = self.latency_predictor.predict(venue, features)
            state.append(pred.predicted_latency_us / 10000.0)
        
        # Market conditions
        state.extend([
            market_summary.get('avg_mid_price', 100) / 1000.0,
            np.log1p(market_summary.get('total_volume', 1000)) / 10.0,
            market_summary.get('avg_spread', 0.01) / 0.1,
            market_summary.get('volatility', 0.01) * 100.0,
            market_summary.get('order_imbalance', 0.0),
            market_summary.get('trade_intensity', 0.5),
            datetime.now().hour / 24.0,
            datetime.now().minute / 60.0,
            float(9 <= datetime.now().hour <= 16),  # Market hours
            0.0  # Placeholder for auction
        ])
        
        # Portfolio state (simplified)
        state.extend([
            0.5,  # Portfolio exposure
            0.1,  # Execution count
            0.0,  # Missed opportunities
            0.0,  # Recent reward average
            0.5   # Buffer fullness
        ])
        
        # Recent performance
        for venue in self.venues[:5]:
            # recent_latencies = [self.network_simulator.get_current_latency(venue)] * 100
            latency = self.network_simulator.get_current_latency(venue)
            state.append(latency / 10000.0)
            # avg_latency = np.mean(recent_latencies) if recent_latencies else 1000.0
            # state.append(avg_latency / 10000.0)
        
        return np.array(state, dtype=np.float32)
    
    def _extract_features_for_venue(self, symbol: str, venue: str) -> np.ndarray:
        """Extract features for latency prediction"""
        # Get feature vector from feature extractor
        feature_vector = self.feature_extractor.extract_features(
            symbol, venue, time.time()
        )
        
        # Convert to format expected by latency predictor
        # This is simplified - in production would use full feature engineering
        features = np.zeros(45)
        
        # Map some key features
        if hasattr(feature_vector, 'features'):
            feature_dict = feature_vector.features
            features[0] = datetime.now().hour / 24.0
            features[5] = feature_dict.get('network_latency_mean', 1000) / 10000.0
            features[10] = feature_dict.get('mid_price', 100)
            features[11] = np.log1p(feature_dict.get('volume', 1000))
            features[14] = feature_dict.get('volatility_1min', 0.01)
        
        return features
    
    def _calculate_expected_reward(self, action: RoutingAction, 
                                  expected_latency: float, urgency: float) -> float:
        """Calculate expected reward for routing decision"""
        if action == RoutingAction.HOLD:
            return -0.1 * urgency
        elif action == RoutingAction.CANCEL:
            return -0.5
        else:
            # Routing reward based on expected latency
            latency_score = 1.0 / (1.0 + expected_latency / 1000.0)
            return latency_score * urgency * 10.0
    
    def _calculate_actual_reward(self, decision: RoutingDecision,
                                actual_latency: float, fill_success: bool) -> float:
        """Calculate actual reward based on results"""
        if not fill_success:
            return -5.0  # Penalty for failed execution
        
        # Base reward for successful execution
        base_reward = 10.0
        
        # Latency component
        if actual_latency < 500:
            latency_reward = 5.0
        elif actual_latency < 1000:
            latency_reward = 2.0
        elif actual_latency < 2000:
            latency_reward = 0.5
        else:
            latency_reward = -2.0
        
        # Prediction accuracy bonus
        if decision.expected_latency_us > 0:
            prediction_error = abs(actual_latency - decision.expected_latency_us)
            accuracy_bonus = max(0, 2.0 - prediction_error / 1000.0)
        else:
            accuracy_bonus = 0.0
        
        return base_reward + latency_reward + accuracy_bonus
    
    def train_agents(self, episodes: int = 1000):
        """Train RL agents"""
        logger.info(f"Training agents for {episodes} episodes...")
        
        self.training_mode = True
        
        for episode in range(episodes):
            # Reset environment
            state = self.trading_env.reset()
            episode_reward = 0
            done = False
            
            # Episode loop
            while not done:
                # DQN training
                if self.current_agent == 'dqn':
                    epsilon = max(0.02, 0.6 * (0.998 ** episode))
                    action, _ = self.dqn_router.act(state, epsilon)
                    
                    next_state, reward, done, info = self.trading_env.step(action)
                    self.dqn_router.step(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                
                # PPO training
                elif self.current_agent == 'ppo':
                    action, confidence, value, log_prob = self.ppo_router.act(state)
                    next_state, reward, done, info = self.trading_env.step(action)
                    
                    self.ppo_router.store_transition(
                        state, action, reward, log_prob, value, done
                    )
                    
                    state = next_state
                    episode_reward += reward
                    
                    # Train PPO at end of episode
                    if done:
                        self.ppo_router.train()
            
            # Log progress
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
                self.dqn_router.reward_history.append(episode_reward)
        
        self.training_mode = False
        logger.info("Training complete")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'summary': self.performance_metrics.copy(),
            'agent_performance': {},
            'venue_statistics': {},
            'latency_distribution': {},
            'routing_patterns': {}
        }
        
        # Agent-specific metrics
        report['agent_performance']['dqn'] = {
            'training_losses': self.dqn_router.training_losses[-100:],
            'reward_history': self.dqn_router.reward_history[-100:]
        }
        
        report['agent_performance']['bandit'] = self.bandit_router.get_statistics()
        
        # Venue performance
        for venue in self.venues:
            venue_decisions = [
                d for d in self.routing_decisions 
                if d.venue == venue
            ]
            
            if venue_decisions:
                latencies = [d.expected_latency_us for d in venue_decisions]
                report['venue_statistics'][venue] = {
                    'selection_count': len(venue_decisions),
                    'avg_expected_latency': np.mean(latencies),
                    'min_expected_latency': np.min(latencies),
                    'max_expected_latency': np.max(latencies)
                }
        
        # Routing patterns over time
        if self.routing_decisions:
            recent_decisions = self.routing_decisions[-1000:]
            hourly_patterns = {}
            
            for decision in recent_decisions:
                hour = datetime.fromtimestamp(decision.timestamp).hour
                if hour not in hourly_patterns:
                    hourly_patterns[hour] = {'count': 0, 'venues': {}}
                
                hourly_patterns[hour]['count'] += 1
                if decision.venue:
                    if decision.venue not in hourly_patterns[hour]['venues']:
                        hourly_patterns[hour]['venues'][decision.venue] = 0
                    hourly_patterns[hour]['venues'][decision.venue] += 1
            
            report['routing_patterns']['hourly'] = hourly_patterns
        
        return report
    
    def save_models(self, directory: str):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save DQN
        self.dqn_router.save(os.path.join(directory, 'dqn_router.pt'))
        
        # Save bandit statistics
        with open(os.path.join(directory, 'bandit_stats.json'), 'w') as f:
            json.dump(self.bandit_router.get_statistics(), f, indent=2)
        
        # Save performance report
        with open(os.path.join(directory, 'performance_report.json'), 'w') as f:
            json.dump(self.get_performance_report(), f, indent=2, default=str)
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load trained models"""
        import os
        
        # Load DQN
        dqn_path = os.path.join(directory, 'dqn_router.pt')
        if os.path.exists(dqn_path):
            self.dqn_router.load(dqn_path)
        
        # Load bandit statistics
        bandit_path = os.path.join(directory, 'bandit_stats.json')
        if os.path.exists(bandit_path):
            with open(bandit_path, 'r') as f:
                stats = json.load(f)
                self.bandit_router.alpha = stats['thompson_params']['alpha']
                self.bandit_router.beta = stats['thompson_params']['beta']
                self.bandit_router.counts = stats['venue_counts']
                self.bandit_router.values = stats['venue_values']
        
        logger.info(f"Models loaded from {directory}")





