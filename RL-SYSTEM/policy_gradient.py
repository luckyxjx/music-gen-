"""
Policy Gradient (REINFORCE) training loop for RL fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

from reward_function import RewardFunction


class Baseline(nn.Module):
    """
    Value function baseline for variance reduction
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict expected reward for a state
        
        Args:
            state: (batch, d_model) state representation
        Returns:
            value: (batch, 1) predicted value
        """
        return self.network(state)


class PolicyGradientTrainer:
    """
    REINFORCE algorithm with baseline for fine-tuning music generator
    """
    def __init__(
        self,
        generator,
        tokenizer,
        reward_function: RewardFunction,
        baseline: Baseline,
        policy_lr: float = 1e-5,
        baseline_lr: float = 1e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Args:
            generator: Pre-trained music generation model
            tokenizer: MIDI tokenizer
            reward_function: Reward function for evaluation
            baseline: Baseline value function
            policy_lr: Learning rate for policy (generator)
            baseline_lr: Learning rate for baseline
            gamma: Discount factor
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computation
        """
        self.generator = generator.to(device)
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.baseline = baseline.to(device)
        self.device = device
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=policy_lr
        )
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(),
            lr=baseline_lr
        )
        
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.previous_samples = []
        
    def generate_episode(
        self,
        emotion: int,
        duration_minutes: float = 2.0,
        temperature: float = 1.0,
        max_tokens: int = 512
    ) -> Tuple[torch.Tensor, List[float], List[torch.Tensor]]:
        """
        Generate a music sequence (episode) and collect log probabilities
        
        Args:
            emotion: Target emotion index
            duration_minutes: Target duration
            temperature: Sampling temperature
            max_tokens: Maximum sequence length
        Returns:
            tokens: (seq_len,) generated token sequence
            log_probs: List of log probabilities for each token
            states: List of hidden states for baseline
        """
        self.generator.eval()
        
        # Start token
        tokens = [self.tokenizer.bos_token_id]
        log_probs = []
        states = []
        
        # Emotion embedding
        emotion_tensor = torch.tensor([emotion], device=self.device)
        
        with torch.set_grad_enabled(True):
            for _ in range(max_tokens):
                # Prepare input
                input_ids = torch.tensor([tokens], device=self.device)
                
                # Forward pass
                outputs = self.generator(
                    input_ids,
                    emotion=emotion_tensor,
                    return_hidden_states=True
                )
                
                logits = outputs['logits'][0, -1, :]  # (vocab_size,)
                hidden_state = outputs['hidden_states'][-1][0, -1, :]  # (d_model,)
                
                # Sample next token
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Store log probability and state
                log_prob = torch.log(probs[next_token] + 1e-10)
                log_probs.append(log_prob)
                states.append(hidden_state)
                
                tokens.append(next_token)
                
                # Stop at EOS token
                if next_token == self.tokenizer.eos_token_id:
                    break
        
        tokens_tensor = torch.tensor(tokens, device=self.device)
        return tokens_tensor, log_probs, states
    
    def compute_advantages(
        self,
        rewards: List[float],
        states: List[torch.Tensor]
    ) -> List[float]:
        """
        Compute advantages using baseline subtraction
        
        Args:
            rewards: List of rewards for each timestep
            states: List of hidden states for baseline prediction
        Returns:
            advantages: List of advantage values
        """
        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Predict baselines
        states_tensor = torch.stack(states)  # (seq_len, d_model)
        with torch.no_grad():
            baselines = self.baseline(states_tensor).squeeze(-1)  # (seq_len,)
        
        # Compute advantages
        returns_tensor = torch.tensor(returns, device=self.device)
        advantages = (returns_tensor - baselines).tolist()
        
        return advantages
    
    def update_policy(
        self,
        log_probs: List[torch.Tensor],
        advantages: List[float]
    ) -> float:
        """
        Update policy using REINFORCE gradient
        
        Args:
            log_probs: List of log probabilities
            advantages: List of advantage values
        Returns:
            policy_loss: Policy loss value
        """
        self.generator.train()
        self.policy_optimizer.zero_grad()
        
        # Compute policy gradient loss
        policy_loss = 0
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss -= log_prob * advantage
        
        policy_loss = policy_loss / len(log_probs)
        
        # Backward pass
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.max_grad_norm
        )
        
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def update_baseline(
        self,
        states: List[torch.Tensor],
        returns: List[float]
    ) -> float:
        """
        Update baseline value function
        
        Args:
            states: List of hidden states
            returns: List of actual returns
        Returns:
            baseline_loss: MSE loss value
        """
        self.baseline.train()
        self.baseline_optimizer.zero_grad()
        
        # Prepare data
        states_tensor = torch.stack(states)  # (seq_len, d_model)
        returns_tensor = torch.tensor(returns, device=self.device).unsqueeze(-1)  # (seq_len, 1)
        
        # Predict values
        predicted_values = self.baseline(states_tensor)  # (seq_len, 1)
        
        # MSE loss
        baseline_loss = F.mse_loss(predicted_values, returns_tensor)
        
        # Backward pass
        baseline_loss.backward()
        self.baseline_optimizer.step()
        
        return baseline_loss.item()
    
    def train_episode(
        self,
        emotion: int,
        duration_minutes: float = 2.0,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Train on a single episode
        
        Args:
            emotion: Target emotion
            duration_minutes: Target duration
            temperature: Sampling temperature
        Returns:
            metrics: Dictionary of training metrics
        """
        # Generate episode
        tokens, log_probs, states = self.generate_episode(
            emotion, duration_minutes, temperature
        )
        
        # Compute reward
        tokens_batch = tokens.unsqueeze(0)  # (1, seq_len)
        total_reward, reward_components = self.reward_function.compute_reward(
            tokens_batch,
            emotion,
            self.previous_samples
        )
        
        # Store sample for diversity computation
        self.previous_samples.append(tokens)
        if len(self.previous_samples) > 50:
            self.previous_samples = self.previous_samples[-50:]
        
        # Assign reward to all timesteps (sparse reward)
        rewards = [total_reward.item()] * len(log_probs)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, states)
        
        # Update policy
        policy_loss = self.update_policy(log_probs, advantages)
        
        # Update baseline
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        baseline_loss = self.update_baseline(states, returns)
        
        # Track statistics
        self.episode_rewards.append(total_reward.item())
        self.episode_lengths.append(len(tokens))
        
        metrics = {
            'policy_loss': policy_loss,
            'baseline_loss': baseline_loss,
            'total_reward': total_reward.item(),
            'emotion_reward': reward_components['emotion'].item(),
            'coherence_reward': reward_components['coherence'].item(),
            'diversity_reward': reward_components['diversity'].item(),
            'episode_length': len(tokens)
        }
        
        return metrics
    
    def train(
        self,
        num_episodes: int,
        emotions: List[int],
        save_dir: str = 'RL-SYSTEM/checkpoints',
        log_interval: int = 10,
        save_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train for multiple episodes
        
        Args:
            num_episodes: Number of episodes to train
            emotions: List of emotions to sample from
            save_dir: Directory for saving checkpoints
            log_interval: Logging frequency
            save_interval: Checkpoint saving frequency
        Returns:
            history: Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = {
            'policy_loss': [],
            'baseline_loss': [],
            'total_reward': [],
            'emotion_reward': [],
            'coherence_reward': [],
            'diversity_reward': []
        }
        
        best_avg_reward = float('-inf')
        
        for episode in range(num_episodes):
            # Sample random emotion
            emotion = np.random.choice(emotions)
            
            # Train episode
            metrics = self.train_episode(emotion)
            
            # Log metrics
            for key in history.keys():
                history[key].append(metrics[key])
            
            # Print progress
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                print(f"Episode {episode+1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.4f}, "
                      f"Policy Loss: {metrics['policy_loss']:.4f}, "
                      f"Baseline Loss: {metrics['baseline_loss']:.4f}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-save_interval:])
                
                checkpoint = {
                    'episode': episode + 1,
                    'generator_state': self.generator.state_dict(),
                    'baseline_state': self.baseline.state_dict(),
                    'policy_optimizer_state': self.policy_optimizer.state_dict(),
                    'baseline_optimizer_state': self.baseline_optimizer.state_dict(),
                    'avg_reward': avg_reward,
                    'history': history
                }
                
                torch.save(checkpoint, save_path / f'checkpoint_ep{episode+1}.pt')
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(checkpoint, save_path / 'best_model.pt')
                    print(f"  → Saved best model (avg reward: {avg_reward:.4f})")
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.baseline.load_state_dict(checkpoint['baseline_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state'])
        
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        print(f"Average reward: {checkpoint['avg_reward']:.4f}")
        
        return checkpoint['history']
