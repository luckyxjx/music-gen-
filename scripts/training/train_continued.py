#!/usr/bin/env python3
"""
Continue training from checkpoint with improvements
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.config import DataConfig, ModelConfig, TrainingConfig, TokenizerConfig
from src.dataset import EMOPIADataset, collate_fn
from src.tokenizer import MIDITokenizer
from src.model import create_model
from src.training.trainer import Trainer
from src.training.logger import ExperimentLogger


def main():
    print("\n" + "="*60)
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("="*60)
    
    # Find best checkpoint
    checkpoint_dir = Path("../../checkpoints")
    checkpoints = list(checkpoint_dir.glob("best_epoch_*.pt"))
    
    if not checkpoints:
        print("\n✗ No checkpoints found!")
        print("Run train.py first to create initial checkpoint")
        return
    
    # Get latest best checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Found checkpoint: {latest_checkpoint}")
    
    # Load checkpoint to get epoch info
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    print(f"  Resuming from epoch {start_epoch}")
    print(f"  Previous train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Previous val loss: {checkpoint['val_loss']:.4f}")
    
    # Configure dataset
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
    
    # Create tokenizer
    print("\n[2/6] Creating tokenizer...")
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    print(f"✓ Tokenizer created (vocab size: {tokenizer.vocab_size})")
    
    # Load datasets
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
    
    # Create model with SAME config as training
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
    
    # Create data loaders
    print("\n[5/6] Creating data loaders...")
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=5e-5,  # Lower LR for continued training
        weight_decay=0.01,
        num_epochs=150,  # Train to epoch 150
        gradient_clip=1.0,
        use_lr_scheduler=True,
        checkpoint_dir="../../checkpoints",
        save_every_n_epochs=5,
        keep_best_n=5,  # Keep more checkpoints
        log_dir="../../logs",
        validate_every_n_epochs=1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
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
    
    # Create trainer
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
        experiment_name=f"continued_training_from_epoch_{start_epoch}"
    )
    
    logger.log_config({
        'model': model_config.__dict__,
        'training': training_config.__dict__,
        'data': data_config.__dict__,
        'device': device,
        'resumed_from': str(latest_checkpoint),
        'start_epoch': start_epoch
    })
    
    print(f"✓ Trainer ready")
    
    # Training info
    print("\n" + "="*60)
    print("CONTINUED TRAINING CONFIGURATION")
    print("="*60)
    print(f"Resuming from: Epoch {start_epoch}")
    print(f"Training to: Epoch {training_config.num_epochs}")
    print(f"Remaining epochs: {training_config.num_epochs - start_epoch + 1}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate} (reduced for stability)")
    print(f"Device: {device}")
    print(f"Estimated time: ~{(training_config.num_epochs - start_epoch + 1) * len(train_loader) * 0.5 / 60:.0f} minutes")
    
    # Start training
    print("\n" + "="*60)
    print("RESUMING TRAINING")
    print("="*60)
    print("\nPress Ctrl+C to stop training at any time.")
    print("Progress will be saved automatically.\n")
    
    try:
        trainer.train(resume_from=str(latest_checkpoint))
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\n✓ Trained from epoch {start_epoch} to {training_config.num_epochs}")
        print(f"✓ Best model saved in: {training_config.checkpoint_dir}")
        print(f"✓ Logs saved in: {logger.get_experiment_dir()}")
        print("\nNext steps:")
        print("1. Test generation: python test_trained_model.py")
        print("2. Start API: python api.py")
        print("3. RL fine-tuning: python rl_finetune.py")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Progress saved in: {training_config.checkpoint_dir}")
        print("Resume by running this script again.")
    
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
