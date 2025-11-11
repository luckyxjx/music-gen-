"""
Music Transformer model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.config import ModelConfig


class PositionalEncoding(nn.Module):
    """Relative positional encoding for music sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return x + self.pe[:x.size(1)]


class TransformerMusicModel(nn.Module):
    """Music Transformer with emotion conditioning"""
    
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.use_emotion = config.use_emotion_conditioning
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        
        # Emotion embedding
        self.use_duration = config.use_duration_control
        emb_size = config.d_model
        
        if self.use_emotion:
            self.emotion_embedding = nn.Embedding(config.num_emotions, config.emotion_emb_dim)
            emb_size += config.emotion_emb_dim
        
        # Duration embedding
        if self.use_duration:
            self.duration_embedding = nn.Linear(1, config.duration_emb_dim)
            emb_size += config.duration_emb_dim
        
        # Project concatenated embeddings back to d_model
        if self.use_emotion or self.use_duration:
            self.input_proj = nn.Linear(emb_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, seq_len] token indices
            emotion: [batch] emotion indices (optional)
            duration: [batch, 1] target duration in minutes (optional)
            mask: Optional attention mask
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens
        token_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add emotion conditioning
        if self.use_emotion and emotion is not None:
            emotion_emb = self.emotion_embedding(emotion)  # [batch, emotion_dim]
            emotion_emb = emotion_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
            token_emb = torch.cat([token_emb, emotion_emb], dim=-1)
        
        # Add duration conditioning
        if self.use_duration and duration is not None:
            duration_emb = self.duration_embedding(duration)  # [batch, duration_dim]
            duration_emb = duration_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
            token_emb = torch.cat([token_emb, duration_emb], dim=-1)
        
        # Project if we added conditioning
        if self.use_emotion or self.use_duration:
            x = self.input_proj(token_emb)
        else:
            x = token_emb
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Generate causal mask if not provided
        if mask is None:
            mask = self.generate_causal_mask(seq_len).to(x.device)
        
        # Transformer decoder (using memory=x for self-attention)
        x = self.transformer(x, x, tgt_mask=mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits


class LSTMMusicModel(nn.Module):
    """LSTM baseline model with emotion conditioning"""
    
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.use_emotion = config.use_emotion_conditioning
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        # Emotion and duration embeddings
        self.use_duration = config.use_duration_control
        lstm_input_size = config.d_model
        
        if self.use_emotion:
            self.emotion_embedding = nn.Embedding(config.num_emotions, config.emotion_emb_dim)
            lstm_input_size += config.emotion_emb_dim
        
        if self.use_duration:
            self.duration_embedding = nn.Linear(1, config.duration_emb_dim)
            lstm_input_size += config.duration_emb_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.lstm_hidden_size, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, seq_len] token indices
            emotion: [batch] emotion indices (optional)
            duration: [batch, 1] target duration in minutes (optional)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens
        x = self.embedding(x)
        
        # Add emotion conditioning
        if self.use_emotion and emotion is not None:
            emotion_emb = self.emotion_embedding(emotion)
            emotion_emb = emotion_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
            x = torch.cat([x, emotion_emb], dim=-1)
        
        # Add duration conditioning
        if self.use_duration and duration is not None:
            duration_emb = self.duration_embedding(duration)
            duration_emb = duration_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
            x = torch.cat([x, duration_emb], dim=-1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits


def interpolate_emotions(
    emotion1: torch.Tensor,
    emotion2: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Linearly interpolate between two emotion embeddings
    
    Args:
        emotion1: First emotion embedding [batch, emb_dim]
        emotion2: Second emotion embedding [batch, emb_dim]
        alpha: Interpolation factor (0 = emotion1, 1 = emotion2)
    
    Returns:
        Interpolated embedding [batch, emb_dim]
    """
    return (1 - alpha) * emotion1 + alpha * emotion2


def create_emotion_transition(
    model: nn.Module,
    emotion_start: int,
    emotion_end: int,
    num_steps: int
) -> list[torch.Tensor]:
    """
    Create smooth emotion transition embeddings
    
    Args:
        model: Model with emotion_embedding layer
        emotion_start: Starting emotion index
        emotion_end: Ending emotion index
        num_steps: Number of interpolation steps
    
    Returns:
        List of interpolated emotion embeddings
    """
    if not hasattr(model, 'emotion_embedding'):
        raise ValueError("Model does not have emotion_embedding layer")
    
    # Get emotion embeddings
    start_emb = model.emotion_embedding(torch.tensor([emotion_start]))
    end_emb = model.emotion_embedding(torch.tensor([emotion_end]))
    
    # Create interpolation steps
    transitions = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        interp_emb = interpolate_emotions(start_emb, end_emb, alpha)
        transitions.append(interp_emb)
    
    return transitions


def create_model(config: ModelConfig, vocab_size: int) -> nn.Module:
    """Factory function to create model based on config"""
    if config.model_type == "transformer":
        return TransformerMusicModel(config, vocab_size)
    elif config.model_type == "lstm":
        return LSTMMusicModel(config, vocab_size)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
