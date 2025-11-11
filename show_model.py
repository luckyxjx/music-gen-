#!/usr/bin/env python3
"""
Quick script to visualize the model architecture
"""

import torch
from src.config import ModelConfig, TokenizerConfig
from src.model import create_model
from src.tokenizer import MIDITokenizer


def main():
    print("\n" + "="*60)
    print("EMOPIA MUSIC GENERATION - MODEL ARCHITECTURE")
    print("="*60)
    
    # Create tokenizer to get vocab size
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    
    print(f"\nTokenizer:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Sample tokens: {list(tokenizer.vocab.keys())[:10]}")
    
    # Create model
    model_config = ModelConfig(
        model_type="transformer",
        d_model=512,
        n_layers=6,
        n_heads=8,
        use_emotion_conditioning=True,
        use_duration_control=True
    )
    
    model = create_model(model_config, tokenizer.vocab_size)
    
    print(f"\nModel Architecture:")
    print(f"  Type: {model_config.model_type.upper()}")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Hidden size: {model_config.d_model}")
    print(f"  Attention heads: {model_config.n_heads}")
    print(f"  Feed-forward size: {model_config.d_ff}")
    print(f"  Dropout: {model_config.dropout}")
    
    print(f"\nConditioning:")
    print(f"  Emotion conditioning: {'✓' if model_config.use_emotion_conditioning else '✗'}")
    print(f"  Emotion embedding dim: {model_config.emotion_emb_dim}")
    print(f"  Duration control: {'✓' if model_config.use_duration_control else '✗'}")
    print(f"  Duration embedding dim: {model_config.duration_emb_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory (FP32): ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Memory (FP16): ~{total_params * 2 / 1024 / 1024:.1f} MB")
    
    # Show model structure
    print(f"\nModel Structure:")
    print(model)
    
    # Test forward pass
    print(f"\n" + "="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    batch_size = 2
    seq_len = 64
    
    # Create dummy inputs
    tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    emotions = torch.randint(0, 6, (batch_size,))
    duration = torch.rand(batch_size, 1) * 3.0
    
    print(f"\nInput:")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Emotions: {emotions.tolist()}")
    print(f"  Duration (minutes): {duration.squeeze().tolist()}")
    
    # Forward pass
    with torch.no_grad():
        output = model(tokens, emotion=emotions, duration=duration)
    
    print(f"\nOutput:")
    print(f"  Logits shape: {output.shape}")
    print(f"  Logits range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Show predictions
    predictions = output.argmax(dim=-1)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[0, :10].tolist()}")
    
    print(f"\n✓ Forward pass successful!")
    
    print(f"\n" + "="*60)
    print("MODEL READY FOR TRAINING")
    print("="*60)


if __name__ == "__main__":
    main()
