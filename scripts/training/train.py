#!/usr/bin/env python3
"""
Train the music generation model
"""

import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, ModelConfig, TrainingConfig, TokenizerConfig
from src.dataset import EMOPIADataset, collate_fn
from src.tokenizer import MIDITokenizer
from src.model import create_model
from src.training.trainer import Trainer
from src.training.logger import ExperimentLogger


def main():
    print("\n" + "="*60)
    print("EMOPIA MUSIC GENERATION - TRAINING")
    print("="*60)
    
    # 1. Configure dataset
    print("\n[1/6] Configuring dataset...")
    data_config = DataConfig(
        dataset_path="../../datasets/EMOPIA_1.0",
        balance_emotions=True,
        balancing_strategy="oversample",
        use_stratified_split=True,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=42
    )
    print("✓ Dataset config ready")
    
    # 2. Create tokenizer
    print("\n[2/6] Creating tokenizer...")
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    print(f"✓ Tokenizer created (vocab size: {tokenizer.vocab_size})")
    
    # 3. Load datasets
    print("\n[3/6] Loading datasets...")
    train_dataset = EMOPIADataset(
        data_config,
        split="train",
        tokenizer=tokenizer,
        max_seq_len=512
    )
    
    val_dataset = EMOPIADataset(
        data_config,
        split="val",
        tokenizer=tokenizer,
        max_seq_len=512
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # 4. Create model
    print("\n[4/6] Creating model...")
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
    
    model = create_model(model_config, tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created ({total_params:,} parameters)")
    
    # 5. Create data loaders
    print("\n[5/6] Creating data loaders...")
    training_config = TrainingConfig(
        batch_size=8,  # Adjust based on your GPU memory
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=25,
        gradient_clip=1.0,
        use_lr_scheduler=True,
        checkpoint_dir="../../checkpoints",
        save_every_n_epochs=5,
        keep_best_n=3,
        log_dir="../../logs",
        validate_every_n_epochs=1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for macOS MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # 6. Create trainer
    print("\n[6/6] Setting up trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    # Create logger
    logger = ExperimentLogger(
        log_dir=training_config.log_dir,
        experiment_name="music_generation"
    )
    
    logger.log_config({
        'model': model_config.__dict__,
        'training': training_config.__dict__,
        'data': data_config.__dict__,
        'device': device
    })
    
    print(f"✓ Trainer ready")
    print(f"✓ Logs will be saved to: {logger.get_experiment_dir()}")
    
    # Training info
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Epochs: {training_config.num_epochs}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Device: {device}")
    print(f"Model parameters: {total_params:,}")
    print(f"Estimated time per epoch: ~{len(train_loader) * 0.5:.0f} seconds")
    print(f"Total estimated time: ~{training_config.num_epochs * len(train_loader) * 0.5 / 60:.0f} minutes")
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print("\nPress Ctrl+C to stop training at any time.")
    print("Training will resume from the last checkpoint.\n")
    
    try:
        trainer.train()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\n✓ Best model saved in: {training_config.checkpoint_dir}")
        print(f"✓ Logs saved in: {logger.get_experiment_dir()}")
        print("\nNext steps:")
        print("1. Generate music: python generate_music.py")
        print("2. Use the trained model for text-to-music generation")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Checkpoints saved in: {training_config.checkpoint_dir}")
        print("Resume training by running this script again.")
    
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
