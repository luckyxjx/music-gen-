"""
Evaluation metrics for music generation
"""

import torch
import numpy as np
from typing import Dict, List
import pretty_midi


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss"""
    return np.exp(loss)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> float:
    """Compute token-level accuracy"""
    predictions = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


class MusicMetrics:
    """Musical coherence metrics"""
    
    @staticmethod
    def pitch_range(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """Compute pitch range statistics"""
        pitches = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                pitches.extend([note.pitch for note in instrument.notes])
        
        if not pitches:
            return {'min': 0, 'max': 0, 'range': 0, 'mean': 0}
        
        return {
            'min': float(np.min(pitches)),
            'max': float(np.max(pitches)),
            'range': float(np.max(pitches) - np.min(pitches)),
            'mean': float(np.mean(pitches))
        }
    
    @staticmethod
    def rhythm_consistency(midi: pretty_midi.PrettyMIDI) -> float:
        """Measure rhythm consistency (lower is more consistent)"""
        note_durations = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    duration = note.end - note.start
                    note_durations.append(duration)
        
        if len(note_durations) < 2:
            return 0.0
        
        # Standard deviation of note durations
        return float(np.std(note_durations))
    
    @staticmethod
    def note_density(midi: pretty_midi.PrettyMIDI) -> float:
        """Compute notes per second"""
        total_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
        duration = midi.get_end_time()
        
        if duration == 0:
            return 0.0
        
        return total_notes / duration


class EmotionValidator:
    """Validate emotion consistency of generated samples"""
    
    def __init__(self, emotion_categories: List[str]):
        self.emotion_categories = emotion_categories
        self.num_emotions = len(emotion_categories)
    
    def validate_emotion_accuracy(
        self,
        generated_samples: List[pretty_midi.PrettyMIDI],
        target_emotions: List[int]
    ) -> Dict[str, float]:
        """
        Validate emotion consistency using simple heuristics
        
        Args:
            generated_samples: List of generated MIDI files
            target_emotions: List of target emotion indices
        
        Returns:
            Dictionary with accuracy metrics
        """
        if len(generated_samples) != len(target_emotions):
            raise ValueError("Mismatch between samples and emotions")
        
        correct = 0
        total = len(generated_samples)
        
        for midi, target_emotion in zip(generated_samples, target_emotions):
            predicted_emotion = self._classify_emotion(midi)
            if predicted_emotion == target_emotion:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'emotion_accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def _classify_emotion(self, midi: pretty_midi.PrettyMIDI) -> int:
        """
        Simple rule-based emotion classification
        Based on tempo, pitch range, and note density
        """
        # Extract features
        tempo = self._estimate_tempo(midi)
        pitch_stats = MusicMetrics.pitch_range(midi)
        density = MusicMetrics.note_density(midi)
        
        # Simple heuristic classification
        # joy: fast tempo, high pitch, high density
        # sadness: slow tempo, low pitch, low density
        # anger: very fast tempo, wide range, high density
        # calm: moderate tempo, narrow range, low density
        # surprise: varied tempo, wide range
        # fear: moderate-fast tempo, dissonant
        
        if tempo > 140:
            return self.emotion_categories.index('anger') if 'anger' in self.emotion_categories else 0
        elif tempo < 80:
            return self.emotion_categories.index('sadness') if 'sadness' in self.emotion_categories else 0
        elif tempo > 110 and density > 5:
            return self.emotion_categories.index('joy') if 'joy' in self.emotion_categories else 0
        else:
            return self.emotion_categories.index('calm') if 'calm' in self.emotion_categories else 0
    
    def _estimate_tempo(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Estimate tempo from MIDI"""
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            return float(tempo_changes[1][0])
        return 120.0  # Default tempo


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'emotion_accuracy': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> float:
        """Get latest metric value"""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return 0.0
    
    def get_best(self, key: str, mode: str = 'min') -> float:
        """Get best metric value"""
        if key not in self.metrics or not self.metrics[key]:
            return float('inf') if mode == 'min' else float('-inf')
        
        if mode == 'min':
            return min(self.metrics[key])
        else:
            return max(self.metrics[key])
    
    def to_dict(self) -> Dict:
        """Export metrics as dictionary"""
        return self.metrics.copy()
