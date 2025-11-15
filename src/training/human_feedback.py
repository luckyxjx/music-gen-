"""
Human-in-the-loop feedback system for RL fine-tuning
Collects and integrates human ratings into reward function
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class HumanFeedbackCollector:
    """Collect and manage human feedback on generated samples"""
    
    def __init__(self, feedback_dir: str = "human_feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.feedback_dir / "feedback_data.jsonl"
        self.stats_file = self.feedback_dir / "feedback_stats.json"
        
        # Load existing feedback
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback from file"""
        if not self.feedback_file.exists():
            return []
        
        feedback = []
        with open(self.feedback_file, 'r') as f:
            for line in f:
                feedback.append(json.loads(line))
        return feedback
    
    def add_feedback(
        self,
        generation_id: str,
        emotion: str,
        emotion_accuracy: int,  # 1-5 scale
        musical_quality: int,   # 1-5 scale
        overall_rating: int,    # 1-5 scale
        comments: str = ""
    ) -> Dict:
        """
        Add human feedback for a generated sample
        
        Args:
            generation_id: ID of the generated sample
            emotion: Target emotion
            emotion_accuracy: How well emotion matches (1-5)
            musical_quality: Musical coherence quality (1-5)
            overall_rating: Overall impression (1-5)
            comments: Optional text comments
        
        Returns:
            Feedback entry
        """
        feedback_entry = {
            'generation_id': generation_id,
            'emotion': emotion,
            'emotion_accuracy': emotion_accuracy,
            'musical_quality': musical_quality,
            'overall_rating': overall_rating,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        
        # Append to file
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        # Add to memory
        self.feedback_data.append(feedback_entry)
        
        # Update statistics
        self._update_statistics()
        
        return feedback_entry
    
    def get_feedback_for_sample(self, generation_id: str) -> Optional[Dict]:
        """Get feedback for a specific sample"""
        for feedback in self.feedback_data:
            if feedback['generation_id'] == generation_id:
                return feedback
        return None
    
    def get_average_ratings(self) -> Dict[str, float]:
        """Get average ratings across all feedback"""
        if not self.feedback_data:
            return {
                'emotion_accuracy': 0.0,
                'musical_quality': 0.0,
                'overall_rating': 0.0,
                'count': 0
            }
        
        emotion_acc = [f['emotion_accuracy'] for f in self.feedback_data]
        musical_qual = [f['musical_quality'] for f in self.feedback_data]
        overall = [f['overall_rating'] for f in self.feedback_data]
        
        return {
            'emotion_accuracy': np.mean(emotion_acc),
            'musical_quality': np.mean(musical_qual),
            'overall_rating': np.mean(overall),
            'count': len(self.feedback_data)
        }
    
    def get_emotion_specific_ratings(self, emotion: str) -> Dict[str, float]:
        """Get average ratings for a specific emotion"""
        emotion_feedback = [f for f in self.feedback_data if f['emotion'] == emotion]
        
        if not emotion_feedback:
            return {
                'emotion_accuracy': 0.0,
                'musical_quality': 0.0,
                'overall_rating': 0.0,
                'count': 0
            }
        
        emotion_acc = [f['emotion_accuracy'] for f in emotion_feedback]
        musical_qual = [f['musical_quality'] for f in emotion_feedback]
        overall = [f['overall_rating'] for f in emotion_feedback]
        
        return {
            'emotion_accuracy': np.mean(emotion_acc),
            'musical_quality': np.mean(musical_qual),
            'overall_rating': np.mean(overall),
            'count': len(emotion_feedback)
        }
    
    def get_human_model_agreement(self) -> Dict[str, float]:
        """
        Calculate agreement between human ratings and model predictions
        Higher emotion_accuracy ratings indicate better agreement
        """
        if not self.feedback_data:
            return {'agreement_score': 0.0, 'count': 0}
        
        # Normalize emotion accuracy to 0-1 scale
        emotion_scores = [f['emotion_accuracy'] / 5.0 for f in self.feedback_data]
        
        return {
            'agreement_score': np.mean(emotion_scores),
            'std': np.std(emotion_scores),
            'count': len(emotion_scores)
        }
    
    def _update_statistics(self):
        """Update and save feedback statistics"""
        stats = {
            'total_feedback': len(self.feedback_data),
            'average_ratings': self.get_average_ratings(),
            'human_model_agreement': self.get_human_model_agreement(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Add per-emotion statistics
        emotions = set(f['emotion'] for f in self.feedback_data)
        stats['per_emotion'] = {
            emotion: self.get_emotion_specific_ratings(emotion)
            for emotion in emotions
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get current feedback statistics"""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {}
    
    def export_for_training(self) -> Dict[str, List]:
        """
        Export feedback in format suitable for RL training
        
        Returns:
            Dictionary with generation IDs and corresponding human rewards
        """
        generation_ids = []
        human_rewards = []
        
        for feedback in self.feedback_data:
            generation_ids.append(feedback['generation_id'])
            
            # Compute human reward as weighted average
            # Normalize to 0-1 scale
            emotion_score = feedback['emotion_accuracy'] / 5.0
            quality_score = feedback['musical_quality'] / 5.0
            overall_score = feedback['overall_rating'] / 5.0
            
            # Weighted combination (similar to RL reward function)
            human_reward = (
                0.5 * emotion_score +
                0.3 * quality_score +
                0.2 * overall_score
            )
            
            human_rewards.append(human_reward)
        
        return {
            'generation_ids': generation_ids,
            'human_rewards': human_rewards
        }


class ActiveLearning:
    """Select samples for human feedback using active learning"""
    
    def __init__(self, feedback_collector: HumanFeedbackCollector):
        self.feedback_collector = feedback_collector
    
    def select_samples_for_feedback(
        self,
        candidate_samples: List[Dict],
        n_samples: int = 10,
        strategy: str = 'uncertainty'
    ) -> List[Dict]:
        """
        Select most informative samples for human feedback
        
        Args:
            candidate_samples: List of generated samples with metadata
            n_samples: Number of samples to select
            strategy: Selection strategy ('uncertainty', 'diversity', 'random')
        
        Returns:
            Selected samples for feedback
        """
        if strategy == 'random':
            return self._random_selection(candidate_samples, n_samples)
        elif strategy == 'uncertainty':
            return self._uncertainty_selection(candidate_samples, n_samples)
        elif strategy == 'diversity':
            return self._diversity_selection(candidate_samples, n_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _random_selection(self, samples: List[Dict], n: int) -> List[Dict]:
        """Random sample selection"""
        import random
        return random.sample(samples, min(n, len(samples)))
    
    def _uncertainty_selection(self, samples: List[Dict], n: int) -> List[Dict]:
        """
        Select samples with highest uncertainty
        (e.g., low confidence in emotion classification)
        """
        # Sort by uncertainty (assuming 'confidence' field exists)
        sorted_samples = sorted(
            samples,
            key=lambda x: x.get('confidence', 0.5)
        )
        return sorted_samples[:n]
    
    def _diversity_selection(self, samples: List[Dict], n: int) -> List[Dict]:
        """
        Select diverse samples covering different emotions and characteristics
        """
        # Group by emotion
        emotion_groups = {}
        for sample in samples:
            emotion = sample.get('emotion', 'unknown')
            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append(sample)
        
        # Select samples from each emotion group
        selected = []
        samples_per_emotion = max(1, n // len(emotion_groups))
        
        for emotion, group in emotion_groups.items():
            selected.extend(group[:samples_per_emotion])
        
        # Fill remaining slots randomly
        if len(selected) < n:
            remaining = [s for s in samples if s not in selected]
            selected.extend(remaining[:n - len(selected)])
        
        return selected[:n]


def integrate_human_feedback_into_reward(
    model_reward: float,
    human_reward: Optional[float],
    alpha: float = 0.3
) -> float:
    """
    Integrate human feedback into model reward
    
    Args:
        model_reward: Reward from automated reward function
        human_reward: Human-provided reward (0-1 scale)
        alpha: Weight for human feedback (0-1)
    
    Returns:
        Combined reward
    """
    if human_reward is None:
        return model_reward
    
    # Weighted combination
    combined_reward = (1 - alpha) * model_reward + alpha * human_reward
    
    return combined_reward


# Example usage
if __name__ == "__main__":
    print("Human Feedback System")
    print("=" * 60)
    
    # Initialize collector
    collector = HumanFeedbackCollector()
    
    # Add sample feedback
    collector.add_feedback(
        generation_id="sample_001",
        emotion="joy",
        emotion_accuracy=4,
        musical_quality=5,
        overall_rating=4,
        comments="Great upbeat melody!"
    )
    
    # Get statistics
    stats = collector.get_statistics()
    print("\nFeedback Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Export for training
    training_data = collector.export_for_training()
    print(f"\nExported {len(training_data['generation_ids'])} samples for training")
