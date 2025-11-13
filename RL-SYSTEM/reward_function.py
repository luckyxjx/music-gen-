"""
Reward function for RL-based emotion alignment
Implements multi-component rewards: emotion accuracy, musical quality, diversity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class EmotionClassifier(nn.Module):
    """
    Emotion classifier for computing emotion alignment rewards
    """
    def __init__(self, vocab_size: int, d_model: int = 256, num_emotions: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) token indices
        Returns:
            logits: (batch, num_emotions) emotion logits
        """
        x = self.embedding(tokens)  # (batch, seq_len, d_model)
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: (4, batch, d_model)
        
        # Concatenate final forward and backward hidden states
        h_forward = h_n[-2]  # (batch, d_model)
        h_backward = h_n[-1]  # (batch, d_model)
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (batch, d_model*2)
        
        logits = self.classifier(h_concat)  # (batch, num_emotions)
        return logits


class RewardFunction:
    """
    Multi-component reward function for RL fine-tuning
    """
    def __init__(
        self,
        emotion_classifier: EmotionClassifier,
        emotion_weight: float = 0.6,
        coherence_weight: float = 0.25,
        diversity_weight: float = 0.15,
        device: str = 'cpu'
    ):
        """
        Args:
            emotion_classifier: Trained emotion classifier
            emotion_weight: Weight for emotion alignment reward
            coherence_weight: Weight for musical coherence reward
            diversity_weight: Weight for diversity reward
            device: Device for computation
        """
        self.emotion_classifier = emotion_classifier.to(device)
        self.emotion_classifier.eval()
        
        self.emotion_weight = emotion_weight
        self.coherence_weight = coherence_weight
        self.diversity_weight = diversity_weight
        self.device = device
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []
        
    def compute_emotion_reward(
        self,
        tokens: torch.Tensor,
        target_emotion: int
    ) -> torch.Tensor:
        """
        Compute emotion alignment reward using classifier confidence
        
        Args:
            tokens: (batch, seq_len) generated token sequences
            target_emotion: Target emotion index (0-5)
        Returns:
            reward: (batch,) emotion alignment scores
        """
        with torch.no_grad():
            logits = self.emotion_classifier(tokens)  # (batch, num_emotions)
            probs = torch.softmax(logits, dim=-1)  # (batch, num_emotions)
            
            # Reward is the confidence in target emotion
            emotion_reward = probs[:, target_emotion]  # (batch,)
            
        return emotion_reward
    
    def compute_coherence_reward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute musical coherence reward
        Measures pitch range consistency and rhythm stability
        
        Args:
            tokens: (batch, seq_len) generated token sequences
        Returns:
            reward: (batch,) coherence scores
        """
        batch_size = tokens.shape[0]
        coherence_scores = []
        
        for i in range(batch_size):
            token_seq = tokens[i].cpu().numpy()
            
            # Pitch range coherence (penalize extreme jumps)
            pitch_tokens = token_seq[token_seq < 128]  # Assuming first 128 tokens are pitches
            if len(pitch_tokens) > 1:
                pitch_diffs = np.abs(np.diff(pitch_tokens))
                avg_jump = np.mean(pitch_diffs)
                # Reward smaller jumps (more coherent)
                pitch_score = np.exp(-avg_jump / 12.0)  # Normalize by octave
            else:
                pitch_score = 0.5
            
            # Rhythm consistency (measure token repetition patterns)
            token_counts = Counter(token_seq)
            entropy = -sum((count / len(token_seq)) * np.log2(count / len(token_seq) + 1e-10)
                          for count in token_counts.values())
            max_entropy = np.log2(len(token_seq))
            rhythm_score = entropy / (max_entropy + 1e-10)  # Normalized entropy
            
            # Combined coherence score
            coherence = 0.6 * pitch_score + 0.4 * rhythm_score
            coherence_scores.append(coherence)
        
        return torch.tensor(coherence_scores, device=self.device, dtype=torch.float32)
    
    def compute_diversity_reward(
        self,
        tokens: torch.Tensor,
        previous_samples: List[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diversity reward to encourage varied outputs
        
        Args:
            tokens: (batch, seq_len) generated token sequences
            previous_samples: List of previous generated sequences for comparison
        Returns:
            reward: (batch,) diversity scores
        """
        batch_size = tokens.shape[0]
        diversity_scores = []
        
        for i in range(batch_size):
            token_seq = tokens[i].cpu().numpy()
            
            # Token entropy (higher is more diverse)
            token_counts = Counter(token_seq)
            entropy = -sum((count / len(token_seq)) * np.log2(count / len(token_seq) + 1e-10)
                          for count in token_counts.values())
            max_entropy = np.log2(len(token_seq))
            entropy_score = entropy / (max_entropy + 1e-10)
            
            # Uniqueness compared to previous samples
            if previous_samples and len(previous_samples) > 0:
                similarities = []
                for prev_sample in previous_samples[-10:]:  # Compare with last 10
                    prev_seq = prev_sample.cpu().numpy()
                    # Jaccard similarity
                    intersection = len(set(token_seq) & set(prev_seq))
                    union = len(set(token_seq) | set(prev_seq))
                    similarity = intersection / (union + 1e-10)
                    similarities.append(similarity)
                uniqueness_score = 1.0 - np.mean(similarities)
            else:
                uniqueness_score = 1.0
            
            # Combined diversity score
            diversity = 0.7 * entropy_score + 0.3 * uniqueness_score
            diversity_scores.append(diversity)
        
        return torch.tensor(diversity_scores, device=self.device, dtype=torch.float32)
    
    def compute_reward(
        self,
        tokens: torch.Tensor,
        target_emotion: int,
        previous_samples: List[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total reward as weighted sum of components
        
        Args:
            tokens: (batch, seq_len) generated token sequences
            target_emotion: Target emotion index
            previous_samples: Previous generated samples for diversity
        Returns:
            total_reward: (batch,) total reward scores
            components: Dict of individual reward components
        """
        # Compute individual rewards
        emotion_reward = self.compute_emotion_reward(tokens, target_emotion)
        coherence_reward = self.compute_coherence_reward(tokens)
        diversity_reward = self.compute_diversity_reward(tokens, previous_samples)
        
        # Weighted combination
        total_reward = (
            self.emotion_weight * emotion_reward +
            self.coherence_weight * coherence_reward +
            self.diversity_weight * diversity_reward
        )
        
        # Normalize rewards
        total_reward = self.normalize_rewards(total_reward)
        
        components = {
            'emotion': emotion_reward,
            'coherence': coherence_reward,
            'diversity': diversity_reward,
            'total': total_reward
        }
        
        return total_reward, components
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards to zero mean and unit variance
        
        Args:
            rewards: (batch,) raw reward scores
        Returns:
            normalized_rewards: (batch,) normalized scores
        """
        # Update running statistics
        self.reward_history.extend(rewards.cpu().tolist())
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        if len(self.reward_history) > 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8
        
        # Normalize
        normalized = (rewards - self.reward_mean) / self.reward_std
        return normalized
    
    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics for logging"""
        return {
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'history_size': len(self.reward_history)
        }


def train_emotion_classifier(
    model: EmotionClassifier,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> EmotionClassifier:
    """
    Train the emotion classifier on labeled data
    
    Args:
        model: EmotionClassifier instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device for training
    Returns:
        trained_model: Trained classifier
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for tokens, emotions in train_loader:
            tokens = tokens.to(device)
            emotions = emotions.to(device)
            
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, emotions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == emotions).sum().item()
            train_total += emotions.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for tokens, emotions in val_loader:
                tokens = tokens.to(device)
                emotions = emotions.to(device)
                
                logits = model(tokens)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == emotions).sum().item()
                val_total += emotions.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'RL-SYSTEM/emotion_classifier_best.pt')
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model
