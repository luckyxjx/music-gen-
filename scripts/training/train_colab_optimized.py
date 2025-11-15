#!/usr/bin/env python3
"""
OPTIMIZED TRAINING FOR COLAB T4 GPU - BEST QUALITY SETTINGS
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os

from src.config import DataConfig, ModelConfig, TrainingConfig, TokenizerConfig
from src.dataset import EMOPIADataset, collate_fn
from src.tokenizer import MIDITokenizer
from src.model import create_model
from src.training.trainer import Trainer
from src.training.logger import ExperimentLogger


def main():
    print("\n" + "="*60)
    print("COLAB T4 OPTIMIZED TRAINING - BEST QUALITY")
    print("="*60)
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available!")
        print("Make sure you're using GPU runtime in Colab:")
        print("Runtime → Change runtime type → GPU (T4)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Find checkpoint to resume from
    checkpoint_dir = Path("../../checkpoints")
    checkpoint_path = None
    start_epoch = 0
    
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("best_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            print(f"\n✓ Found checkpoint: {checkpoint_path}")
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Previous loss: {checkpoint['val_loss']:.4f}")
    
    # OPTIMIZED DATA CONFIG FOR BEST QUALITY
    print("\n[1/6] Configuring dataset (BEST QUALITY)...")
    data_config = DataConfig(
        dataset_path="../../datasets/EMOPIA_1.0",
        balance_emotions=True,
        balancing_strategy="oversample",  # Oversample for balanced classes
        use_stratified_split=True,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        # AGGRESSIVE AUGMENTATION for better generalization
        pitch_shift_range=7,  # ±7 semitones (full octave range)
        tempo_variation=0.15,  # ±15% tempo variation
        apply_augmentation=True,
        # NORMALIZATION for consistency
        normalize_tempo=True,
        normalize_key=True,
        normalize_time_signature=True,
        target_bpm=120
    )
    print("✓ Data config: Aggressive augmentation, full normalization")
    
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
        max_seq_len=1024  # LONGER sequences for better music
    )
    
    val_dataset = EMOPIADataset(
        data_config,
        split="val",
        tokenizer=tokenizer,
        max_seq_len=1024
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # OPTIMIZED MODEL CONFIG FOR BEST QUALITY
    print("\n[4/6] Creating model (BEST QUALITY)...")
    model_config = ModelConfig(
        model_type="transformer",
        # LARGER MODEL for better capacity
        d_model=512,  # Keep at 512 for T4 memory
        n_layers=8,  # MORE layers for better learning
        n_heads=8,
        d_ff=2048,
        dropout=0.15,  # HIGHER dropout for better generalization
        max_seq_len=1024,  # LONGER sequences
        # EMOTION CONDITIONING
        use_emotion_conditioning=True,
        emotion_emb_dim=64,
        num_emotions=6,
        # DURATION CONTROL
        use_duration_control=True,
        duration_emb_dim=32
    )
    
    model = create_model(model_config, tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created ({total_params:,} parameters)")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Hidden size: {model_config.d_model}")
    print(f"  Max sequence: {model_config.max_seq_len}")
    
    # OPTIMIZED TRAINING CONFIG FOR T4 GPU
    print("\n[5/6] Creating data loaders (T4 OPTIMIZED)...")
    training_config = TrainingConfig(
        # T4 GPU OPTIMIZED BATCH SIZE
        batch_size=12,  # Optimal for T4 with 512 d_model
        # LEARNING RATE SCHEDULE
        learning_rate=3e-5 if start_epoch > 0 else 1e-4,  # Lower if resuming
        weight_decay=0.01,
        # TRAIN TO 150 EPOCHS for best quality
        num_epochs=150,
        gradient_clip=1.0,
        # ADVANCED LR SCHEDULING
        use_lr_scheduler=True,
        scheduler_type="cosine",
        warmup_epochs=10,  # Longer warmup
        # CHECKPOINTING
        checkpoint_dir="../../checkpoints",
        save_every_n_epochs=5,
        keep_best_n=10,  # Keep more checkpoints
        # LOGGING
        log_dir="../../logs",
        use_wandb=False,
        log_every_n_steps=50,
        # VALIDATION
        validate_every_n_epochs=1,
        generate_samples_during_val=True,
        num_val_samples=6
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Colab can handle 2 workers
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Batch size: {training_config.batch_size} (T4 optimized)")
    
    # Create trainer
    print("\n[6/6] Setting up trainer...")
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
        experiment_name=f"colab_t4_optimized_epoch_{start_epoch}"
    )
    
    logger.log_config({
        'model': model_config.__dict__,
        'training': training_config.__dict__,
        'data': data_config.__dict__,
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'start_epoch': start_epoch,
        'optimization': 'T4 GPU, Best Quality Settings'
    })
    
    print(f"✓ Trainer ready")
    
    # Training info
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION - BEST QUALITY")
    print("="*60)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Target epoch: {training_config.num_epochs}")
    print(f"Remaining epochs: {training_config.num_epochs - start_epoch}")
    print(f"Batch size: {training_config.batch_size} (T4 optimized)")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Model size: {total_params:,} parameters")
    print(f"Sequence length: {model_config.max_seq_len} tokens")
    print(f"Augmentation: Aggressive (±7 semitones, ±15% tempo)")
    print(f"Dropout: {model_config.dropout} (high for generalization)")
    
    # Time estimate
    time_per_epoch = len(train_loader) * 0.8  # ~0.8 sec per batch on T4
    total_time = (training_config.num_epochs - start_epoch) * time_per_epoch / 60
    print(f"\nEstimated time:")
    print(f"  Per epoch: ~{time_per_epoch/60:.1f} minutes")
    print(f"  Total: ~{total_time/60:.1f} hours")
    print(f"  To epoch 50: ~{(50 - start_epoch) * time_per_epoch / 60:.1f} hours")
    print(f"  To epoch 100: ~{(100 - start_epoch) * time_per_epoch / 60:.1f} hours")
    
    # Colab tips
    print("\n" + "="*60)
    print("COLAB TIPS")
    print("="*60)
    print("1. Keep this tab open to prevent disconnection")
    print("2. Download checkpoints periodically:")
    print("   from google.colab import files")
    print("   files.download('checkpoints/best_epoch_X.pt')")
    print("3. Monitor GPU usage:")
    print("   !nvidia-smi")
    print("4. If disconnected, just run again - it will resume!")
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print("\nPress Ctrl+C to stop (progress will be saved)\n")
    
    try:
        trainer.train(resume_from=str(checkpoint_path) if checkpoint_path else None)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\n✓ Trained to epoch {training_config.num_epochs}")
        print(f"✓ Best model saved in: {training_config.checkpoint_dir}")
        print(f"✓ Logs saved in: {logger.get_experiment_dir()}")
        
        # Download instructions
        print("\n" + "="*60)
        print("DOWNLOAD YOUR MODEL")
        print("="*60)
        print("\nRun this in a Colab cell:")
        print("```python")
        print("from google.colab import files")
        print("import glob")
        print("for f in glob.glob('checkpoints/best_*.pt'):")
        print("    files.download(f)")
        print("```")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Progress saved in: {training_config.checkpoint_dir}")
        print("Run this script again to resume.")
    
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
