#!/usr/bin/env python3
"""
RL Fine-tuning script for music generation
"""

import sys
import torch
from pathlib import Path

# Add RL-SYSTEM to path
sys.path.append('RL-SYSTEM')

from src.config import ModelConfig, TokenizerConfig
from src.model import create_model
from src.tokenizer import MIDITokenizer
from reward_function import EmotionClassifier, RewardFunction
from policy_gradient import PolicyGradientTrainer, Baseline


def main():
    print("\n" + "="*60)
    print("RL FINE-TUNING FOR MUSIC GENERATION")
    print("="*60)
    
    # Find best checkpoint
    checkpoint_dir = Path("../../checkpoints")
    checkpoints = list(checkpoint_dir.glob("best_epoch_*.pt"))
    
    if not checkpoints:
        print("\n✗ No checkpoints found!")
        print("Train the model first with train.py")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Loading checkpoint: {latest_checkpoint}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Create tokenizer
    print("\n[1/5] Creating tokenizer...")
    tokenizer = MIDITokenizer(TokenizerConfig())
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Create and load generator model
    print("\n[2/5] Loading generator model...")
    model_config = ModelConfig(
        model_type="transformer",
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512,
        use_emotion_conditioning=True,
        emotion_emb_dim=64,
        num_emotions=6,
        use_duration_control=True,
        duration_emb_dim=32
    )
    
    generator = create_model(model_config, tokenizer.vocab_size)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator = generator.to(device)
    print(f"✓ Generator loaded from epoch {checkpoint['epoch']}")
    
    # Create emotion classifier
    print("\n[3/5] Creating emotion classifier...")
    emotion_classifier = EmotionClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_emotions=6
    ).to(device)
    
    # Try to load pretrained classifier
    classifier_path = Path("RL-SYSTEM/emotion_classifier_best.pt")
    if classifier_path.exists():
        emotion_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print(f"✓ Loaded pretrained emotion classifier")
    else:
        print(f"⚠️  No pretrained classifier found, using random initialization")
        print(f"   For better results, train classifier first")
    
    # Create reward function
    print("\n[4/5] Setting up reward function...")
    reward_function = RewardFunction(
        emotion_classifier=emotion_classifier,
        emotion_weight=0.6,
        coherence_weight=0.25,
        diversity_weight=0.15,
        device=device
    )
    print(f"✓ Reward function initialized")
    
    # Create baseline
    baseline = Baseline(d_model=512).to(device)
    
    # Create RL trainer
    print("\n[5/5] Creating RL trainer...")
    rl_trainer = PolicyGradientTrainer(
        generator=generator,
        tokenizer=tokenizer,
        reward_function=reward_function,
        baseline=baseline,
        policy_lr=1e-5,
        baseline_lr=1e-4,
        gamma=0.99,
        device=device
    )
    print(f"✓ RL trainer ready")
    
    # Training configuration
    print("\n" + "="*60)
    print("RL TRAINING CONFIGURATION")
    print("="*60)
    print(f"Episodes: 1000")
    print(f"Policy LR: 1e-5")
    print(f"Baseline LR: 1e-4")
    print(f"Emotions: [joy, sadness, anger, calm, surprise, fear]")
    print(f"Reward weights: emotion=0.6, coherence=0.25, diversity=0.15")
    print(f"Device: {device}")
    
    # Start RL training
    print("\n" + "="*60)
    print("STARTING RL FINE-TUNING")
    print("="*60)
    print("\nThis will improve emotion accuracy and musical quality.")
    print("Press Ctrl+C to stop at any time.\n")
    
    try:
        history = rl_trainer.train(
            num_episodes=1000,
            emotions=[0, 1, 2, 3, 4, 5],
            save_dir='RL-SYSTEM/checkpoints',
            log_interval=10,
            save_interval=100
        )
        
        print("\n" + "="*60)
        print("RL FINE-TUNING COMPLETE!")
        print("="*60)
        print(f"\n✓ Completed 1000 episodes")
        print(f"✓ Model saved in: RL-SYSTEM/checkpoints/")
        print("\nNext steps:")
        print("1. Test improved model: python test_trained_model.py")
        print("2. Compare with baseline")
        print("3. Collect human feedback")
        
    except KeyboardInterrupt:
        print("\n\nRL training interrupted by user.")
        print("Progress saved in: RL-SYSTEM/checkpoints/")
    
    except Exception as e:
        print(f"\n✗ RL training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
