"""
Dataset loaders for various emotion-labeled MIDI datasets
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

from src.config import QUADRANT_TO_EMOTION


class DatasetLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
    
    def load(self) -> List[Dict]:
        """Load dataset and return list of samples"""
        raise NotImplementedError


class EMOPIALoader(DatasetLoader):
    """Loader for EMOPIA dataset"""
    
    def load(self) -> List[Dict]:
        samples = []
        labels_path = self.dataset_path / "label.csv"
        
        if not labels_path.exists():
            print(f"Warning: EMOPIA labels not found at {labels_path}")
            return samples
        
        df = pd.read_csv(labels_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading EMOPIA"):
            midi_path = self.dataset_path / "midis" / f"{row['ID']}.mid"
            
            if not midi_path.exists():
                continue
            
            quadrant = int(row['4Q'])
            emotion = QUADRANT_TO_EMOTION.get(quadrant, "calm")
            
            samples.append({
                "path": str(midi_path),
                "emotion": emotion,
                "quadrant": quadrant,
                "source": "emopia"
            })
        
        return samples


class LakhMIDILoader(DatasetLoader):
    """
    Loader for Lakh MIDI dataset with emotion annotations
    Expected format: metadata.json with emotion labels
    """
    
    EMOTION_MAPPING = {
        "happy": "joy",
        "joyful": "joy",
        "excited": "joy",
        "sad": "sadness",
        "melancholic": "sadness",
        "depressed": "sadness",
        "angry": "anger",
        "aggressive": "anger",
        "furious": "anger",
        "calm": "calm",
        "peaceful": "calm",
        "relaxed": "calm",
        "surprised": "surprise",
        "shocked": "surprise",
        "scared": "fear",
        "fearful": "fear",
        "anxious": "fear"
    }
    
    def load(self) -> List[Dict]:
        samples = []
        metadata_path = self.dataset_path / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Warning: Lakh MIDI metadata not found at {metadata_path}")
            return samples
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for item in tqdm(metadata, desc="Loading Lakh MIDI"):
            midi_path = self.dataset_path / item.get("path", "")
            
            if not midi_path.exists():
                continue
            
            # Map emotion label
            raw_emotion = item.get("emotion", "").lower()
            emotion = self.EMOTION_MAPPING.get(raw_emotion, "calm")
            
            samples.append({
                "path": str(midi_path),
                "emotion": emotion,
                "quadrant": None,
                "source": "lakh_midi"
            })
        
        return samples


class MAESTROLoader(DatasetLoader):
    """
    Loader for MAESTRO dataset with emotion annotations
    Expected format: CSV file with emotion labels
    """
    
    EMOTION_MAPPING = {
        "happy": "joy",
        "joyful": "joy",
        "sad": "sadness",
        "melancholic": "sadness",
        "angry": "anger",
        "calm": "calm",
        "peaceful": "calm",
        "surprised": "surprise",
        "fearful": "fear"
    }
    
    def load(self) -> List[Dict]:
        samples = []
        metadata_path = self.dataset_path / "maestro_emotions.csv"
        
        if not metadata_path.exists():
            print(f"Warning: MAESTRO emotion labels not found at {metadata_path}")
            return samples
        
        df = pd.read_csv(metadata_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading MAESTRO"):
            midi_path = self.dataset_path / row.get("midi_filename", "")
            
            if not midi_path.exists():
                continue
            
            # Map emotion
            raw_emotion = str(row.get("emotion", "")).lower()
            emotion = self.EMOTION_MAPPING.get(raw_emotion, "calm")
            
            samples.append({
                "path": str(midi_path),
                "emotion": emotion,
                "quadrant": None,
                "source": "maestro"
            })
        
        return samples


class GenericJSONLoader(DatasetLoader):
    """
    Generic loader for JSON-formatted datasets
    Expected format: 
    [
        {"path": "relative/path/to/file.mid", "emotion": "joy"},
        ...
    ]
    """
    
    EMOTION_MAPPING = {
        "happy": "joy",
        "joyful": "joy",
        "excited": "joy",
        "sad": "sadness",
        "melancholic": "sadness",
        "depressed": "sadness",
        "angry": "anger",
        "aggressive": "anger",
        "calm": "calm",
        "peaceful": "calm",
        "relaxed": "calm",
        "surprised": "surprise",
        "scared": "fear",
        "fearful": "fear"
    }
    
    def load(self) -> List[Dict]:
        samples = []
        metadata_path = self.dataset_path / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Warning: Generic dataset metadata not found at {metadata_path}")
            return samples
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for item in tqdm(metadata, desc=f"Loading {self.dataset_path.name}"):
            midi_path = self.dataset_path / item.get("path", "")
            
            if not midi_path.exists():
                continue
            
            # Map emotion
            raw_emotion = item.get("emotion", "").lower()
            emotion = self.EMOTION_MAPPING.get(raw_emotion, raw_emotion)
            
            # Validate emotion is in our categories
            valid_emotions = ["joy", "sadness", "anger", "calm", "surprise", "fear"]
            if emotion not in valid_emotions:
                emotion = "calm"  # Default fallback
            
            samples.append({
                "path": str(midi_path),
                "emotion": emotion,
                "quadrant": None,
                "source": self.dataset_path.name
            })
        
        return samples


def get_dataset_loader(dataset_path: str, dataset_type: Optional[str] = None) -> DatasetLoader:
    """
    Factory function to get appropriate dataset loader
    
    Args:
        dataset_path: Path to dataset
        dataset_type: Type of dataset (emopia, lakh, maestro, json). 
                     If None, will auto-detect based on directory structure
    
    Returns:
        DatasetLoader instance
    """
    path = Path(dataset_path)
    
    if dataset_type:
        dataset_type = dataset_type.lower()
        if dataset_type == "emopia":
            return EMOPIALoader(dataset_path)
        elif dataset_type == "lakh" or dataset_type == "lakh_midi":
            return LakhMIDILoader(dataset_path)
        elif dataset_type == "maestro":
            return MAESTROLoader(dataset_path)
        elif dataset_type == "json":
            return GenericJSONLoader(dataset_path)
    
    # Auto-detect based on files present
    if (path / "label.csv").exists() and (path / "midis").exists():
        return EMOPIALoader(dataset_path)
    elif (path / "maestro_emotions.csv").exists():
        return MAESTROLoader(dataset_path)
    elif (path / "metadata.json").exists():
        # Check if it's Lakh or generic
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            if metadata and "lakh" in str(path).lower():
                return LakhMIDILoader(dataset_path)
            else:
                return GenericJSONLoader(dataset_path)
    
    # Default to generic JSON loader
    print(f"Warning: Could not auto-detect dataset type for {dataset_path}, using GenericJSONLoader")
    return GenericJSONLoader(dataset_path)
