"""
Advanced emotion class balancing strategies
"""

import random
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np


class EmotionBalancer:
    """Handles emotion class balancing with multiple strategies"""
    
    def __init__(self, strategy: str = "oversample", seed: int = 42):
        """
        Initialize balancer
        
        Args:
            strategy: Balancing strategy - 'oversample', 'undersample', 'hybrid', or 'weighted'
            seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def balance(self, samples: List[Dict]) -> List[Dict]:
        """
        Balance emotion classes in dataset
        
        Args:
            samples: List of sample dictionaries with 'emotion' key
        
        Returns:
            Balanced list of samples
        """
        if self.strategy == "oversample":
            return self._oversample(samples)
        elif self.strategy == "undersample":
            return self._undersample(samples)
        elif self.strategy == "hybrid":
            return self._hybrid_sample(samples)
        elif self.strategy == "weighted":
            # For weighted strategy, return original samples with weights
            return self._add_sample_weights(samples)
        else:
            raise ValueError(f"Unknown balancing strategy: {self.strategy}")
    
    def _oversample(self, samples: List[Dict]) -> List[Dict]:
        """
        Oversample minority classes to match majority class
        
        This duplicates samples from minority classes with replacement
        """
        emotion_groups = self._group_by_emotion(samples)
        
        # Find max class size
        max_size = max(len(group) for group in emotion_groups.values())
        
        print(f"\nBalancing with oversampling (target size: {max_size} per class)")
        
        balanced_samples = []
        for emotion, group in emotion_groups.items():
            original_size = len(group)
            if original_size < max_size:
                # Oversample with replacement
                oversampled = random.choices(group, k=max_size)
                balanced_samples.extend(oversampled)
                print(f"  {emotion}: {original_size} → {max_size} (+{max_size - original_size})")
            else:
                balanced_samples.extend(group)
                print(f"  {emotion}: {original_size} (no change)")
        
        random.shuffle(balanced_samples)
        print(f"Total samples after balancing: {len(balanced_samples)}")
        
        return balanced_samples
    
    def _undersample(self, samples: List[Dict]) -> List[Dict]:
        """
        Undersample majority classes to match minority class
        
        This randomly removes samples from majority classes
        """
        emotion_groups = self._group_by_emotion(samples)
        
        # Find min class size
        min_size = min(len(group) for group in emotion_groups.values())
        
        print(f"\nBalancing with undersampling (target size: {min_size} per class)")
        
        balanced_samples = []
        for emotion, group in emotion_groups.items():
            original_size = len(group)
            if original_size > min_size:
                # Undersample without replacement
                undersampled = random.sample(group, k=min_size)
                balanced_samples.extend(undersampled)
                print(f"  {emotion}: {original_size} → {min_size} (-{original_size - min_size})")
            else:
                balanced_samples.extend(group)
                print(f"  {emotion}: {original_size} (no change)")
        
        random.shuffle(balanced_samples)
        print(f"Total samples after balancing: {len(balanced_samples)}")
        
        return balanced_samples
    
    def _hybrid_sample(self, samples: List[Dict]) -> List[Dict]:
        """
        Hybrid approach: oversample small classes, undersample large classes
        
        Target is the median class size
        """
        emotion_groups = self._group_by_emotion(samples)
        
        # Find median class size as target
        sizes = [len(group) for group in emotion_groups.values()]
        target_size = int(np.median(sizes))
        
        print(f"\nBalancing with hybrid sampling (target size: {target_size} per class)")
        
        balanced_samples = []
        for emotion, group in emotion_groups.items():
            original_size = len(group)
            
            if original_size < target_size:
                # Oversample
                oversampled = random.choices(group, k=target_size)
                balanced_samples.extend(oversampled)
                print(f"  {emotion}: {original_size} → {target_size} (+{target_size - original_size})")
            elif original_size > target_size:
                # Undersample
                undersampled = random.sample(group, k=target_size)
                balanced_samples.extend(undersampled)
                print(f"  {emotion}: {original_size} → {target_size} (-{original_size - target_size})")
            else:
                balanced_samples.extend(group)
                print(f"  {emotion}: {original_size} (no change)")
        
        random.shuffle(balanced_samples)
        print(f"Total samples after balancing: {len(balanced_samples)}")
        
        return balanced_samples
    
    def _add_sample_weights(self, samples: List[Dict]) -> List[Dict]:
        """
        Add sample weights for weighted sampling (no actual balancing)
        
        Weights are inversely proportional to class frequency
        """
        emotion_groups = self._group_by_emotion(samples)
        
        # Calculate weights
        total_samples = len(samples)
        emotion_weights = {}
        
        for emotion, group in emotion_groups.items():
            # Weight = total_samples / (num_classes * class_size)
            weight = total_samples / (len(emotion_groups) * len(group))
            emotion_weights[emotion] = weight
        
        print(f"\nAdding sample weights (no resampling):")
        for emotion, weight in emotion_weights.items():
            print(f"  {emotion}: weight = {weight:.3f}")
        
        # Add weights to samples
        weighted_samples = []
        for sample in samples:
            sample_copy = sample.copy()
            sample_copy["weight"] = emotion_weights[sample["emotion"]]
            weighted_samples.append(sample_copy)
        
        return weighted_samples
    
    def _group_by_emotion(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by emotion"""
        emotion_groups = {}
        for sample in samples:
            emotion = sample["emotion"]
            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append(sample)
        return emotion_groups
    
    def get_class_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Get emotion class distribution"""
        emotions = [s["emotion"] for s in samples]
        return dict(Counter(emotions))
    
    def print_distribution(self, samples: List[Dict], title: str = "Distribution"):
        """Print emotion distribution"""
        distribution = self.get_class_distribution(samples)
        total = len(samples)
        
        print(f"\n{title}:")
        print(f"Total samples: {total}")
        for emotion, count in sorted(distribution.items()):
            percentage = (count / total) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")


class StratifiedSplitter:
    """Stratified train/val/test splitting"""
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize splitter
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        random.seed(seed)
    
    def split(
        self,
        samples: List[Dict],
        stratify_key: str = "emotion"
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Perform stratified split
        
        Args:
            samples: List of samples
            stratify_key: Key to stratify on (default: 'emotion')
        
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Group by stratification key
        groups = {}
        for sample in samples:
            key = sample[stratify_key]
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        print(f"\nPerforming stratified split ({self.train_ratio:.0%}/{self.val_ratio:.0%}/{self.test_ratio:.0%}):")
        
        # Split each group proportionally
        for key, group in groups.items():
            random.shuffle(group)
            n = len(group)
            
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            train_group = group[:train_end]
            val_group = group[train_end:val_end]
            test_group = group[val_end:]
            
            train_samples.extend(train_group)
            val_samples.extend(val_group)
            test_samples.extend(test_group)
            
            print(f"  {key}: {len(train_group)} train, {len(val_group)} val, {len(test_group)} test")
        
        # Shuffle final splits
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        print(f"\nTotal: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_samples, val_samples, test_samples
    
    def save_split_indices(
        self,
        samples: List[Dict],
        output_path: str,
        stratify_key: str = "emotion"
    ):
        """
        Save split indices to JSON for reproducibility
        
        Args:
            samples: List of samples
            output_path: Path to save indices
            stratify_key: Key to stratify on
        """
        import json
        from pathlib import Path
        
        train, val, test = self.split(samples, stratify_key)
        
        # Create index mapping
        sample_to_idx = {sample["path"]: idx for idx, sample in enumerate(samples)}
        
        split_indices = {
            "train": [sample_to_idx[s["path"]] for s in train],
            "val": [sample_to_idx[s["path"]] for s in val],
            "test": [sample_to_idx[s["path"]] for s in test],
            "config": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "seed": self.seed,
                "stratify_key": stratify_key
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(split_indices, f, indent=2)
        
        print(f"\nSaved split indices to {output_path}")


def analyze_emotion_distribution(samples: List[Dict]) -> Dict:
    """
    Analyze emotion distribution and return statistics
    
    Args:
        samples: List of samples with 'emotion' key
    
    Returns:
        Dictionary with distribution statistics
    """
    emotions = [s["emotion"] for s in samples]
    distribution = Counter(emotions)
    
    total = len(samples)
    counts = list(distribution.values())
    
    stats = {
        "total_samples": total,
        "num_classes": len(distribution),
        "distribution": dict(distribution),
        "min_class_size": min(counts),
        "max_class_size": max(counts),
        "mean_class_size": np.mean(counts),
        "median_class_size": np.median(counts),
        "std_class_size": np.std(counts),
        "imbalance_ratio": max(counts) / min(counts) if min(counts) > 0 else float('inf')
    }
    
    return stats


def print_distribution_analysis(stats: Dict):
    """Print formatted distribution analysis"""
    print("\n" + "="*60)
    print("EMOTION DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    
    print(f"\nClass sizes:")
    for emotion, count in sorted(stats['distribution'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    print(f"\nStatistics:")
    print(f"  Min class size: {stats['min_class_size']}")
    print(f"  Max class size: {stats['max_class_size']}")
    print(f"  Mean class size: {stats['mean_class_size']:.1f}")
    print(f"  Median class size: {stats['median_class_size']:.1f}")
    print(f"  Std deviation: {stats['std_class_size']:.1f}")
    print(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
    
    if stats['imbalance_ratio'] > 2.0:
        print("\n⚠️  High class imbalance detected! Consider balancing.")
    elif stats['imbalance_ratio'] > 1.5:
        print("\n⚠️  Moderate class imbalance detected.")
    else:
        print("\n✓ Classes are relatively balanced.")
    
    print("="*60)
