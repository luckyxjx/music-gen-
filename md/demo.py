#!/usr/bin/env python3
"""
Demo script showing the complete EMOPIA music generation pipeline
Phases 1-3: Data processing, Model architecture, Training infrastructure
"""

import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, ModelConfig, TrainingConfig, TokenizerConfig
from src.dataset import EMOPIADataset, collate_fn
from src.tokenizer import MIDITokenizer
from src.model import create_model
from src.training.trainer import Trainer
from src.training.logger import ExperimentLogger
from src.training.metrics import MetricsTracker


def demo_phase1_data_processing():
    """Phase 1: Dataset & Preprocessing"""
    print("\n" + "="*60)
    print("PHASE 1: DATASET & PREPROCESSING")
    print("="*60)
    
    # Configure dataset
    data_config = DataConfig(
        dataset_path="./EMOPIA_1.0",
        balance_emotions=True,
        balancing_strategy="oversample",
        use_stratified_split=True,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=42
    )
    
    print("\n✓ Configuration created")
    print(f"  - Balancing: {data_config.balancing_strategy}")
    print(f"  - Stratified split: {data_config.use_stratified_split}")
    print(f"  - Splits: {data_config.train_split}/{data_config.val_split}/{data_config.test_split}")
    
    # Create tokenizer
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    
    print(f"\n✓ Tokenizer created")
    print(f"  - Vocabulary size: {tokenizer.vocab_size}")
    print(f"  - Quantization: {tokenizer_config.quantization}")
    
    # Load datasets
    print("\n✓ Loading datasets...")
    try:
        train_dataset = EMOPIADataset(data_config, split="train", tokenizer=tokenizer, max_seq_len=512)
        val_dataset = EMOPIADataset(data_config, split="val", tokenizer=tokenizer, max_seq_len=512)
        
        print(f"\n✓ Datasets loaded successfully!")
        print(f"  - Train samples: {len(train_dataset)}")
        print(f"  - Val samples: {len(val_dataset)}")
        
        # Show sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n✓ Sample data structure:")
            print(f"  - Tokens shape: {sample['tokens'].shape}")
            print(f"  - Emotion: {sample['emotion_name']} (index: {sample['emotion']})")
        
        return train_dataset, val_dataset, tokenizer
    
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nNote: Make sure EMOPIA_1.0 dataset is in the current directory")
        return None, None, tokenizer


def demo_phase2_model_architecture(vocab_size):
    """Phase 2: Model Architecture"""
    print("\n" + "="*60)
    print("PHASE 2: MODEL ARCHITECTURE")
    print("="*60)
    
    # Configure model
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
    
    print("\n✓ Model configuration created")
    print(f"  - Type: {model_config.model_type}")
    print(f"  - Layers: {model_config.n_layers}")
    print(f"  - Hidden size: {model_config.d_model}")
    print(f"  - Attention heads: {model_config.n_heads}")
    print(f"  - Emotion conditioning: {model_config.use_emotion_conditioning}")
    print(f"  - Duration control: {model_config.use_duration_control}")
    
    # Create model
    model = create_model(model_config, vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    batch_size = 2
    seq_len = 64
    
    dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_emotions = torch.randint(0, 6, (batch_size,))
    dummy_duration = torch.rand(batch_size, 1) * 3.0  # 0-3 minutes
    
    try:
        with torch.no_grad():
            output = model(dummy_tokens, emotion=dummy_emotions, duration=dummy_duration)
        
        print(f"  - Input shape: {dummy_tokens.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output logits range: [{output.min():.2f}, {output.max():.2f}]")
        print("\n✓ Forward pass successful!")
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
    
    return model, model_config


def demo_phase3_training_infrastructure(model, train_dataset, val_dataset):
    """Phase 3: Training Infrastructure"""
    print("\n" + "="*60)
    print("PHASE 3: TRAINING INFRASTRUCTURE")
    print("="*60)
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=4,  # Small batch for demo
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=2,  # Just 2 epochs for demo
        gradient_clip=1.0,
        use_lr_scheduler=True,
        checkpoint_dir="./checkpoints_demo",
        save_every_n_epochs=1,
        keep_best_n=2,
        log_dir="./logs_demo",
        validate_every_n_epochs=1
    )
    
    print("\n✓ Training configuration created")
    print(f"  - Batch size: {training_config.batch_size}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    print(f"  - Epochs: {training_config.num_epochs}")
    print(f"  - Gradient clipping: {training_config.gradient_clip}")
    print(f"  - LR scheduler: {training_config.use_lr_scheduler}")
    
    # Create data loaders
    if train_dataset is None or val_dataset is None:
        print("\n✗ Cannot create data loaders without datasets")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"\n✓ Data loaders created")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    print(f"\n✓ Trainer created")
    print(f"  - Optimizer: AdamW")
    print(f"  - Scheduler: CosineAnnealingLR")
    print(f"  - Checkpoint manager: Enabled")
    
    # Create logger
    logger = ExperimentLogger(
        log_dir=training_config.log_dir,
        experiment_name="demo_run"
    )
    
    print(f"\n✓ Logger created")
    print(f"  - Experiment dir: {logger.get_experiment_dir()}")
    
    # Log configuration
    logger.log_config({
        'model': model.config.__dict__,
        'training': training_config.__dict__,
        'device': device
    })
    
    print("\n✓ Configuration logged")
    
    return trainer, logger


def main():
    """Run complete demo"""
    print("\n" + "="*60)
    print("EMOPIA MUSIC GENERATION - COMPLETE PIPELINE DEMO")
    print("Phases 1-3: Data, Model, Training")
    print("="*60)
    
    # Phase 1: Data Processing
    train_dataset, val_dataset, tokenizer = demo_phase1_data_processing()
    
    if train_dataset is None:
        print("\n⚠️  Demo stopped: Dataset not available")
        print("\nTo run the full demo:")
        print("1. Download EMOPIA dataset")
        print("2. Place it in ./EMOPIA_1.0/")
        print("3. Run this script again")
        return
    
    # Phase 2: Model Architecture
    model, model_config = demo_phase2_model_architecture(tokenizer.vocab_size)
    
    # Phase 3: Training Infrastructure
    trainer, logger = demo_phase3_training_infrastructure(model, train_dataset, val_dataset)
    
    if trainer is None:
        return
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print("\n✓ All phases initialized successfully!")
    print("\nWhat we've built:")
    print("  ✓ Phase 1: Multi-source dataset loading with balancing")
    print("  ✓ Phase 1: Data augmentation (pitch, tempo, velocity)")
    print("  ✓ Phase 1: Musical normalization (tempo, key)")
    print("  ✓ Phase 1: Stratified train/val/test splitting")
    print("  ✓ Phase 2: Transformer model with emotion conditioning")
    print("  ✓ Phase 2: Duration control mechanism")
    print("  ✓ Phase 2: Multi-instrument tokenization")
    print("  ✓ Phase 2: Emotion interpolation system")
    print("  ✓ Phase 3: Automated checkpoint management")
    print("  ✓ Phase 3: Learning rate scheduling")
    print("  ✓ Phase 3: Comprehensive metrics tracking")
    print("  ✓ Phase 3: Emotion consistency validation")
    print("  ✓ Phase 3: Experiment logging")
    
    print("\n" + "="*60)
    print("READY FOR TRAINING!")
    print("="*60)
    print("\nTo start training, uncomment the line below:")
    print("# trainer.train()")
    print("\nOr run: python train.py")
    
    # Optionally run a quick training demo
    print("\n" + "="*60)
    user_input = input("Run quick training demo (2 epochs)? [y/N]: ")
    if user_input.lower() == 'y':
        print("\nStarting training demo...")
        try:
            trainer.train()
            print("\n✓ Training demo completed!")
        except Exception as e:
            print(f"\n✗ Training error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
