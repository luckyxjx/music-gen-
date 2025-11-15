"""
RL evaluation and tracking for emotion alignment fine-tuning
Tracks reward progression, emotion accuracy, and musical quality metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt


class RLEvaluator:
    """Track and evaluate RL fine-tuning improvements"""
    
    def __init__(self, save_dir: str = "rl_evaluation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics_history = {
            'episode': [],
            'total_reward': [],
            'emotion_reward': [],
            'coherence_reward': [],
            'diversity_reward': [],
            'emotion_accuracy': [],
            'avg_episode_length': [],
            'policy_loss': [],
            'baseline_loss': []
        }
        
        # Pre-RL baseline metrics
        self.pre_rl_metrics = None
        
    def set_pre_rl_baseline(self, metrics: Dict[str, float]):
        """Store pre-RL performance metrics for comparison"""
        self.pre_rl_metrics = metrics.copy()
        
        # Save to file
        with open(self.save_dir / 'pre_rl_baseline.json', 'w') as f:
            json.dump(self.pre_rl_metrics, f, indent=2)
        
        print(f"Pre-RL baseline set: {metrics}")
    
    def update(self, episode: int, metrics: Dict[str, float]):
        """Update metrics for current episode"""
        self.metrics_history['episode'].append(episode)
        
        for key in ['total_reward', 'emotion_reward', 'coherence_reward', 
                    'diversity_reward', 'policy_loss', 'baseline_loss']:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
        
        # Optional metrics
        if 'emotion_accuracy' in metrics:
            self.metrics_history['emotion_accuracy'].append(metrics['emotion_accuracy'])
        if 'avg_episode_length' in metrics:
            self.metrics_history['avg_episode_length'].append(metrics['avg_episode_length'])
    
    def compute_moving_average(self, values: List[float], window: int = 10) -> List[float]:
        """Compute moving average for smoothing"""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(values[start_idx:i+1]))
        
        return smoothed
    
    def get_improvement_stats(self) -> Dict[str, float]:
        """Calculate improvement statistics compared to pre-RL baseline"""
        if not self.pre_rl_metrics or not self.metrics_history['total_reward']:
            return {}
        
        # Get recent performance (last 100 episodes)
        recent_window = min(100, len(self.metrics_history['total_reward']))
        recent_rewards = self.metrics_history['total_reward'][-recent_window:]
        recent_emotion_acc = self.metrics_history['emotion_accuracy'][-recent_window:] if self.metrics_history['emotion_accuracy'] else []
        
        stats = {
            'pre_rl_reward': self.pre_rl_metrics.get('total_reward', 0.0),
            'post_rl_reward': np.mean(recent_rewards),
            'reward_improvement': np.mean(recent_rewards) - self.pre_rl_metrics.get('total_reward', 0.0),
            'reward_improvement_pct': ((np.mean(recent_rewards) - self.pre_rl_metrics.get('total_reward', 0.0)) / 
                                       (abs(self.pre_rl_metrics.get('total_reward', 1.0)) + 1e-8) * 100)
        }
        
        if 'emotion_accuracy' in self.pre_rl_metrics and recent_emotion_acc:
            stats['pre_rl_emotion_acc'] = self.pre_rl_metrics['emotion_accuracy']
            stats['post_rl_emotion_acc'] = np.mean(recent_emotion_acc)
            stats['emotion_acc_improvement'] = np.mean(recent_emotion_acc) - self.pre_rl_metrics['emotion_accuracy']
        
        return stats
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Generate training progress plots"""
        if not self.metrics_history['episode']:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=16)
        
        episodes = self.metrics_history['episode']
        
        # Plot 1: Total Reward
        ax = axes[0, 0]
        rewards = self.metrics_history['total_reward']
        smoothed_rewards = self.compute_moving_average(rewards, window=10)
        ax.plot(episodes, rewards, alpha=0.3, label='Raw')
        ax.plot(episodes, smoothed_rewards, label='Smoothed (MA-10)')
        if self.pre_rl_metrics and 'total_reward' in self.pre_rl_metrics:
            ax.axhline(y=self.pre_rl_metrics['total_reward'], color='r', 
                      linestyle='--', label='Pre-RL Baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reward Components
        ax = axes[0, 1]
        if self.metrics_history['emotion_reward']:
            emotion_smooth = self.compute_moving_average(self.metrics_history['emotion_reward'])
            coherence_smooth = self.compute_moving_average(self.metrics_history['coherence_reward'])
            diversity_smooth = self.compute_moving_average(self.metrics_history['diversity_reward'])
            
            ax.plot(episodes, emotion_smooth, label='Emotion')
            ax.plot(episodes, coherence_smooth, label='Coherence')
            ax.plot(episodes, diversity_smooth, label='Diversity')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Component')
        ax.set_title('Reward Components (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Emotion Accuracy
        ax = axes[1, 0]
        if self.metrics_history['emotion_accuracy']:
            acc = self.metrics_history['emotion_accuracy']
            acc_smooth = self.compute_moving_average(acc)
            ax.plot(episodes[:len(acc)], acc, alpha=0.3, label='Raw')
            ax.plot(episodes[:len(acc_smooth)], acc_smooth, label='Smoothed')
            if self.pre_rl_metrics and 'emotion_accuracy' in self.pre_rl_metrics:
                ax.axhline(y=self.pre_rl_metrics['emotion_accuracy'], color='r',
                          linestyle='--', label='Pre-RL Baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Emotion Accuracy')
        ax.set_title('Emotion Classification Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss Values
        ax = axes[1, 1]
        if self.metrics_history['policy_loss']:
            policy_smooth = self.compute_moving_average(self.metrics_history['policy_loss'])
            baseline_smooth = self.compute_moving_average(self.metrics_history['baseline_loss'])
            
            ax.plot(episodes, policy_smooth, label='Policy Loss')
            ax.plot(episodes, baseline_smooth, label='Baseline Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / 'training_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress plot saved to {save_path}")
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if not self.metrics_history['episode']:
            return "No training data available"
        
        report_lines = [
            "=" * 60,
            "RL FINE-TUNING EVALUATION REPORT",
            "=" * 60,
            ""
        ]
        
        # Training summary
        total_episodes = len(self.metrics_history['episode'])
        report_lines.extend([
            f"Total Episodes: {total_episodes}",
            ""
        ])
        
        # Recent performance (last 100 episodes)
        window = min(100, total_episodes)
        recent_rewards = self.metrics_history['total_reward'][-window:]
        
        report_lines.extend([
            f"Recent Performance (last {window} episodes):",
            f"  Average Total Reward: {np.mean(recent_rewards):.4f} ± {np.std(recent_rewards):.4f}",
            f"  Max Reward: {np.max(recent_rewards):.4f}",
            f"  Min Reward: {np.min(recent_rewards):.4f}",
            ""
        ])
        
        # Reward components
        if self.metrics_history['emotion_reward']:
            recent_emotion = self.metrics_history['emotion_reward'][-window:]
            recent_coherence = self.metrics_history['coherence_reward'][-window:]
            recent_diversity = self.metrics_history['diversity_reward'][-window:]
            
            report_lines.extend([
                "Reward Components:",
                f"  Emotion Reward: {np.mean(recent_emotion):.4f}",
                f"  Coherence Reward: {np.mean(recent_coherence):.4f}",
                f"  Diversity Reward: {np.mean(recent_diversity):.4f}",
                ""
            ])
        
        # Emotion accuracy
        if self.metrics_history['emotion_accuracy']:
            recent_acc = self.metrics_history['emotion_accuracy'][-window:]
            report_lines.extend([
                f"Emotion Accuracy: {np.mean(recent_acc):.4f}",
                ""
            ])
        
        # Improvement over baseline
        if self.pre_rl_metrics:
            improvement_stats = self.get_improvement_stats()
            report_lines.extend([
                "Improvement vs Pre-RL Baseline:",
                f"  Pre-RL Reward: {improvement_stats.get('pre_rl_reward', 0):.4f}",
                f"  Post-RL Reward: {improvement_stats.get('post_rl_reward', 0):.4f}",
                f"  Absolute Improvement: {improvement_stats.get('reward_improvement', 0):.4f}",
                f"  Relative Improvement: {improvement_stats.get('reward_improvement_pct', 0):.2f}%",
                ""
            ])
            
            if 'emotion_acc_improvement' in improvement_stats:
                report_lines.extend([
                    f"  Pre-RL Emotion Acc: {improvement_stats['pre_rl_emotion_acc']:.4f}",
                    f"  Post-RL Emotion Acc: {improvement_stats['post_rl_emotion_acc']:.4f}",
                    f"  Accuracy Improvement: {improvement_stats['emotion_acc_improvement']:.4f}",
                    ""
                ])
        
        # Training stability
        if len(recent_rewards) > 10:
            # Compute trend (linear regression slope)
            x = np.arange(len(recent_rewards))
            slope = np.polyfit(x, recent_rewards, 1)[0]
            
            report_lines.extend([
                "Training Stability:",
                f"  Reward Std Dev: {np.std(recent_rewards):.4f}",
                f"  Trend (slope): {slope:.6f}",
                f"  Status: {'Improving' if slope > 0 else 'Declining'}",
                ""
            ])
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.save_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Evaluation report saved to {report_path}")
        
        return report_text
    
    def save_metrics(self):
        """Save metrics history to JSON"""
        metrics_path = self.save_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
    
    def load_metrics(self, metrics_path: str):
        """Load metrics history from JSON"""
        with open(metrics_path, 'r') as f:
            self.metrics_history = json.load(f)
        
        print(f"Metrics loaded from {metrics_path}")
    
    def compare_models(
        self,
        pre_rl_samples: List[torch.Tensor],
        post_rl_samples: List[torch.Tensor],
        target_emotions: List[int],
        emotion_classifier
    ) -> Dict[str, float]:
        """
        Compare pre-RL and post-RL model performance
        
        Args:
            pre_rl_samples: Generated samples from pre-RL model
            post_rl_samples: Generated samples from post-RL model
            target_emotions: Target emotion labels
            emotion_classifier: Trained emotion classifier
        
        Returns:
            comparison_stats: Dictionary with comparison metrics
        """
        emotion_classifier.eval()
        
        # Evaluate pre-RL samples
        pre_rl_correct = 0
        with torch.no_grad():
            for sample, target in zip(pre_rl_samples, target_emotions):
                logits = emotion_classifier(sample.unsqueeze(0))
                pred = torch.argmax(logits, dim=-1).item()
                if pred == target:
                    pre_rl_correct += 1
        
        pre_rl_acc = pre_rl_correct / len(pre_rl_samples)
        
        # Evaluate post-RL samples
        post_rl_correct = 0
        with torch.no_grad():
            for sample, target in zip(post_rl_samples, target_emotions):
                logits = emotion_classifier(sample.unsqueeze(0))
                pred = torch.argmax(logits, dim=-1).item()
                if pred == target:
                    post_rl_correct += 1
        
        post_rl_acc = post_rl_correct / len(post_rl_samples)
        
        comparison = {
            'pre_rl_accuracy': pre_rl_acc,
            'post_rl_accuracy': post_rl_acc,
            'accuracy_improvement': post_rl_acc - pre_rl_acc,
            'improvement_pct': ((post_rl_acc - pre_rl_acc) / (pre_rl_acc + 1e-8)) * 100,
            'num_samples': len(pre_rl_samples)
        }
        
        print("\nModel Comparison:")
        print(f"  Pre-RL Accuracy: {pre_rl_acc:.4f}")
        print(f"  Post-RL Accuracy: {post_rl_acc:.4f}")
        print(f"  Improvement: {comparison['accuracy_improvement']:.4f} ({comparison['improvement_pct']:.2f}%)")
        
        return comparison
