"""
Configuration management for the music generation system
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import yaml
import json


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_path: str = "./EMOPIA_1.0"
    # Can be list of paths (str) or list of dicts with {"path": "...", "type": "..."}
    # Supported types: emopia, lakh, maestro, json
    additional_datasets: List = field(default_factory=list)
    emotion_categories: List[str] = field(default_factory=lambda: [
        "joy", "sadness", "anger", "calm", "surprise", "fear"
    ])
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    
    # Augmentation
    pitch_shift_range: int = 5  # ±5 semitones
    tempo_variation: float = 0.1  # ±10%
    apply_augmentation: bool = True
    
    # Normalization
    normalize_tempo: bool = True
    normalize_key: bool = True
    normalize_time_signature: bool = True
    target_bpm: int = 120
    
    # Class balancing
    balance_emotions: bool = True
    balancing_strategy: str = "oversample"  # oversample, undersample, hybrid, weighted
    use_stratified_split: bool = True
    save_split_indices: bool = True
    split_indices_path: str = "./data/split_indices.json"


@dataclass
class TokenizerConfig:
    """Tokenization configuration"""
    max_shift_steps: int = 32
    velocity_buckets: List[int] = field(default_factory=lambda: [0, 40, 80, 128])
    max_events: int = 512  # Increased for longer sequences
    quantization: str = "16th"  # 16th note resolution
    
    # Multi-instrument support
    instruments: List[str] = field(default_factory=lambda: ["melody", "harmony", "drums"])
    instrument_programs: Dict[str, int] = field(default_factory=lambda: {
        "melody": 0,  # Acoustic Grand Piano
        "harmony": 0,  # Acoustic Grand Piano
        "drums": 128  # Drum kit
    })


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "transformer"  # or "lstm"
    
    # Transformer settings
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    
    # LSTM settings
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 3
    
    # Emotion conditioning
    emotion_emb_dim: int = 64
    num_emotions: int = 6
    use_emotion_conditioning: bool = True
    
    # Duration control
    use_duration_control: bool = True
    duration_emb_dim: int = 32
    
    # Multi-instrument
    use_multi_instrument: bool = False
    instrument_emb_dim: int = 32


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    gradient_clip: float = 1.0
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, linear, exponential
    warmup_epochs: int = 5
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    keep_best_n: int = 3
    
    # Logging
    log_dir: str = "./logs"
    use_wandb: bool = False
    wandb_project: str = "emopia-music-gen"
    log_every_n_steps: int = 100
    
    # Validation
    validate_every_n_epochs: int = 1
    generate_samples_during_val: bool = True
    num_val_samples: int = 4


@dataclass
class GenerationConfig:
    """Generation configuration"""
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    
    # Duration control
    default_duration_minutes: float = 2.0
    min_duration_minutes: float = 0.5
    max_duration_minutes: float = 5.0
    
    # Sampling
    seed_length: int = 32
    use_seed: bool = True
    
    # Output
    output_dir: str = "./generated"
    export_formats: List[str] = field(default_factory=lambda: ["midi", "wav", "mp3"])
    
    # Audio synthesis
    soundfont_path: Optional[str] = None
    sample_rate: int = 44100
    mp3_bitrate: str = "192k"


@dataclass
class SystemConfig:
    """Complete system configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Device
    device: str = "auto"  # auto, cuda, mps, cpu
    
    @classmethod
    def from_yaml(cls, path: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'SystemConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


# Emotion presets
EMOTION_PRESETS = {
    "joy": {
        "tempo_range": (120, 140),
        "key_preference": "major",
        "velocity_range": (80, 120)
    },
    "sadness": {
        "tempo_range": (60, 80),
        "key_preference": "minor",
        "velocity_range": (40, 70)
    },
    "anger": {
        "tempo_range": (140, 180),
        "key_preference": "minor",
        "velocity_range": (100, 127)
    },
    "calm": {
        "tempo_range": (70, 90),
        "key_preference": "major",
        "velocity_range": (50, 80)
    },
    "surprise": {
        "tempo_range": (110, 130),
        "key_preference": "major",
        "velocity_range": (70, 100)
    },
    "fear": {
        "tempo_range": (100, 120),
        "key_preference": "minor",
        "velocity_range": (60, 90)
    }
}


# Quadrant to emotion mapping
QUADRANT_TO_EMOTION = {
    1: "joy",      # Q1: High Valence, High Arousal
    2: "anger",    # Q2: Low Valence, High Arousal
    3: "sadness",  # Q3: Low Valence, Low Arousal
    4: "calm"      # Q4: High Valence, Low Arousal
}


def get_default_config() -> SystemConfig:
    """Get default system configuration"""
    return SystemConfig()


def create_config_template(path: str, format: str = "yaml"):
    """Create a configuration template file"""
    config = get_default_config()
    if format == "yaml":
        config.to_yaml(path)
    elif format == "json":
        config.to_json(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
