"""
Enhanced dataset module with augmentation, normalization, and multi-source support
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json

from src.config import DataConfig, QUADRANT_TO_EMOTION
from src.dataset_loaders import get_dataset_loader, EMOPIALoader
from src.data_balancing import (
    EmotionBalancer,
    StratifiedSplitter,
    analyze_emotion_distribution,
    print_distribution_analysis
)


class MIDIAugmenter:
    """Handles MIDI data augmentation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.pitch_shift_range = config.pitch_shift_range
        self.tempo_variation = config.tempo_variation
    
    def pitch_shift(self, midi: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
        """Shift all notes by semitones"""
        shifted_midi = pretty_midi.PrettyMIDI()
        
        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            for note in instrument.notes:
                new_pitch = np.clip(note.pitch + semitones, 0, 127)
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=int(new_pitch),
                    start=note.start,
                    end=note.end
                )
                new_instrument.notes.append(new_note)
            
            shifted_midi.instruments.append(new_instrument)
        
        return shifted_midi
    
    def tempo_change(self, midi: pretty_midi.PrettyMIDI, factor: float) -> pretty_midi.PrettyMIDI:
        """Change tempo by multiplying all times by factor"""
        scaled_midi = pretty_midi.PrettyMIDI()
        
        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            for note in instrument.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start / factor,
                    end=note.end / factor
                )
                new_instrument.notes.append(new_note)
            
            scaled_midi.instruments.append(new_instrument)
        
        return scaled_midi
    
    def velocity_variation(self, midi: pretty_midi.PrettyMIDI, scale: float) -> pretty_midi.PrettyMIDI:
        """Vary note velocities for dynamic diversity"""
        varied_midi = pretty_midi.PrettyMIDI()
        
        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            for note in instrument.notes:
                new_velocity = int(np.clip(note.velocity * scale, 1, 127))
                new_note = pretty_midi.Note(
                    velocity=new_velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end
                )
                new_instrument.notes.append(new_note)
            
            varied_midi.instruments.append(new_instrument)
        
        return varied_midi
    
    def augment(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Apply random augmentation"""
        if not self.config.apply_augmentation:
            return midi
        
        # Random pitch shift (±5 semitones)
        if random.random() < 0.5:
            semitones = random.randint(-self.pitch_shift_range, self.pitch_shift_range)
            midi = self.pitch_shift(midi, semitones)
        
        # Random tempo variation (±10%)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.tempo_variation, self.tempo_variation)
            midi = self.tempo_change(midi, factor)
        
        # Random velocity variation (±20%)
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            midi = self.velocity_variation(midi, scale)
        
        return midi


class MIDINormalizer:
    """Handles MIDI normalization"""
    
    # Emotion-specific BPM ranges
    EMOTION_BPM = {
        "joy": (120, 140),
        "sadness": (60, 80),
        "anger": (140, 180),
        "calm": (70, 90),
        "surprise": (110, 130),
        "fear": (100, 120)
    }
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.target_bpm = config.target_bpm
    
    def normalize_tempo(self, midi: pretty_midi.PrettyMIDI, emotion: Optional[str] = None) -> pretty_midi.PrettyMIDI:
        """Normalize tempo to target BPM or emotion-specific range"""
        if not self.config.normalize_tempo:
            return midi
        
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) == 0:
            return midi
        
        current_bpm = tempo_changes[1][0]
        
        # Use emotion-specific BPM if provided
        if emotion and emotion in self.EMOTION_BPM:
            bpm_min, bpm_max = self.EMOTION_BPM[emotion]
            target_bpm = np.clip(current_bpm, bpm_min, bpm_max)
        else:
            target_bpm = self.target_bpm
        
        if abs(current_bpm - target_bpm) < 1:
            return midi
        
        factor = current_bpm / target_bpm
        
        # Scale all note times
        normalized_midi = pretty_midi.PrettyMIDI()
        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            for note in instrument.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start * factor,
                    end=note.end * factor
                )
                new_instrument.notes.append(new_note)
            
            normalized_midi.instruments.append(new_instrument)
        
        return normalized_midi
    
    def normalize_key(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Transpose to C major or A minor"""
        if not self.config.normalize_key:
            return midi
        
        pitch_classes = np.zeros(12)
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch_classes[note.pitch % 12] += 1
        
        if pitch_classes.sum() == 0:
            return midi
        
        tonic = np.argmax(pitch_classes)
        
        # Transpose to C (0) or A (9)
        if tonic != 0 and tonic != 9:
            semitones = -tonic
            augmenter = MIDIAugmenter(self.config)
            return augmenter.pitch_shift(midi, semitones)
        
        return midi
    
    def normalize_time_signature(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Normalize to 4/4 time signature"""
        if not self.config.normalize_time_signature:
            return midi
        
        # Note: pretty_midi doesn't directly support time signature changes
        # This is a placeholder for future implementation
        return midi


class EMOPIADataset(Dataset):
    """Enhanced EMOPIA dataset with augmentation and normalization"""
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        tokenizer=None,
        max_seq_len: int = 512
    ):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.augmenter = MIDIAugmenter(config)
        self.normalizer = MIDINormalizer(config)
        
        # Load data
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_emotion_distribution()
    
    def _load_dataset(self) -> List[Dict]:
        """Load and process dataset from multiple sources"""
        samples = []
        
        # Load EMOPIA dataset
        print("Loading primary EMOPIA dataset...")
        emopia_loader = EMOPIALoader(self.config.dataset_path)
        emopia_samples = emopia_loader.load()
        samples.extend(emopia_samples)
        print(f"Loaded {len(emopia_samples)} samples from EMOPIA")
        
        # Load additional datasets if specified
        for dataset_info in self.config.additional_datasets:
            if isinstance(dataset_info, str):
                # Simple path string
                dataset_path = dataset_info
                dataset_type = None
            elif isinstance(dataset_info, dict):
                # Dict with path and type
                dataset_path = dataset_info.get("path")
                dataset_type = dataset_info.get("type")
            else:
                print(f"Warning: Invalid dataset info format: {dataset_info}")
                continue
            
            print(f"Loading additional dataset from {dataset_path}...")
            loader = get_dataset_loader(dataset_path, dataset_type)
            additional_samples = loader.load()
            samples.extend(additional_samples)
            print(f"Loaded {len(additional_samples)} samples from {dataset_path}")
        
        print(f"\nTotal samples before processing: {len(samples)}")
        
        # Analyze distribution before balancing
        stats = analyze_emotion_distribution(samples)
        print_distribution_analysis(stats)
        
        # Perform stratified split first (before balancing)
        if self.config.use_stratified_split:
            splitter = StratifiedSplitter(
                train_ratio=self.config.train_split,
                val_ratio=self.config.val_split,
                test_ratio=self.config.test_split,
                seed=self.config.seed
            )
            train_samples, val_samples, test_samples = splitter.split(samples)
            
            # Save split indices if requested
            if self.config.save_split_indices and self.split == "train":
                splitter.save_split_indices(samples, self.config.split_indices_path)
            
            # Select appropriate split
            if self.split == "train":
                samples = train_samples
            elif self.split == "val":
                samples = val_samples
            elif self.split == "test":
                samples = test_samples
        else:
            # Use simple random split
            samples = self._split_dataset(samples)
        
        # Balance emotion classes (only for training set)
        if self.split == "train" and self.config.balance_emotions:
            balancer = EmotionBalancer(
                strategy=self.config.balancing_strategy,
                seed=self.config.seed
            )
            samples = balancer.balance(samples)
        
        return samples
    


    
    def _split_dataset(self, samples: List[Dict]) -> List[Dict]:
        """Split dataset into train/val/test"""
        random.seed(self.config.seed)
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))
        
        if self.split == "train":
            return samples[:train_end]
        elif self.split == "val":
            return samples[train_end:val_end]
        elif self.split == "test":
            return samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _print_emotion_distribution(self):
        """Print emotion distribution"""
        emotions = [s["emotion"] for s in self.samples]
        emotion_counts = pd.Series(emotions).value_counts()
        print(f"\nEmotion distribution ({self.split}):")
        print(emotion_counts)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        sample = self.samples[idx]
        
        try:
            # Load MIDI
            midi = pretty_midi.PrettyMIDI(sample["path"])
            
            # Apply normalization with emotion context
            emotion = sample["emotion"]
            midi = self.normalizer.normalize_tempo(midi, emotion)
            midi = self.normalizer.normalize_key(midi)
            midi = self.normalizer.normalize_time_signature(midi)
            
            # Apply augmentation (only for training)
            if self.split == "train":
                midi = self.augmenter.augment(midi)
            
            # Tokenize if tokenizer provided
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(midi)
                tokens = tokens[:self.max_seq_len]
            else:
                tokens = []
            
            # Map emotion to index
            emotion_idx = self.config.emotion_categories.index(sample["emotion"])
            
            return {
                "tokens": torch.LongTensor(tokens) if tokens else torch.LongTensor([]),
                "emotion": emotion_idx,
                "emotion_name": sample["emotion"],
                "path": sample["path"]
            }
        
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return empty sample
            return {
                "tokens": torch.LongTensor([]),
                "emotion": 0,
                "emotion_name": "calm",
                "path": ""
            }


def collate_fn(batch: List[Dict], pad_value: int = 0) -> Dict:
    """Collate function for DataLoader"""
    # Filter out empty samples
    batch = [b for b in batch if len(b["tokens"]) > 0]
    
    if len(batch) == 0:
        return {
            "tokens": torch.LongTensor([]),
            "emotions": torch.LongTensor([]),
            "lengths": torch.LongTensor([])
        }
    
    # Get max length
    max_len = max(len(b["tokens"]) for b in batch)
    
    # Pad sequences
    padded_tokens = []
    emotions = []
    lengths = []
    
    for b in batch:
        tokens = b["tokens"]
        length = len(tokens)
        
        # Pad
        if length < max_len:
            padding = torch.full((max_len - length,), pad_value, dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        padded_tokens.append(tokens)
        emotions.append(b["emotion"])
        lengths.append(length)
    
    return {
        "tokens": torch.stack(padded_tokens),
        "emotions": torch.LongTensor(emotions),
        "lengths": torch.LongTensor(lengths)
    }
