"""
Training pipeline with checkpoint management and RL evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Optional, Dict
from tqdm import tqdm

from src.config import TrainingConfig
from src.training.rl_evaluator import RLEvaluator


class CheckpointManager:
    """Manages model checkpoints"""
    
    def __init__(self, checkpoint_dir: str, keep_best_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.best_checkpoints = []  # List of (loss, path) tuples
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        train_loss: float,
        val_loss: float,
        config: TrainingConfig,
        is_best: bool = False
    ) -> str:
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config.__dict__
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / f'best_epoch_{epoch}_loss_{val_loss:.4f}.pt'
            self._update_best_checkpoints(val_loss, path)
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'is_best': is_best
        }
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(path)
    
    def _update_best_checkpoints(self, val_loss: float, path: Path):
        """Track and cleanup best checkpoints"""
        self.best_checkpoints.append((val_loss, path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        
        # Remove old checkpoints beyond keep_best_n
        if len(self.best_checkpoints) > self.keep_best_n:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                old_path.with_suffix('.json').unlink(missing_ok=True)
    
    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict:
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        if not self.best_checkpoints:
            best_files = list(self.checkpoint_dir.glob('best_epoch_*.pt'))
            if not best_files:
                return None
            return str(best_files[0])
        
        _, path = self.best_checkpoints[0]
        return str(path)


class Trainer:
    """Training orchestration with RL evaluation support"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = 'cpu',
        enable_rl_eval: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.keep_best_n
        )
        
        # RL Evaluator (optional)
        self.rl_evaluator = RLEvaluator() if enable_rl_eval else None
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            tokens = batch['tokens'].to(self.device)
            emotions = batch['emotions'].to(self.device)
            
            # Create dummy duration (random between 1-3 minutes)
            batch_size = tokens.size(0)
            duration = torch.rand(batch_size, 1).to(self.device) * 2 + 1  # 1-3 minutes
            
            # Forward pass
            logits = self.model(tokens[:, :-1], emotion=emotions, duration=duration)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=0  # Assuming 0 is PAD token
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                tokens = batch['tokens'].to(self.device)
                emotions = batch['emotions'].to(self.device)
                
                # Create dummy duration
                batch_size = tokens.size(0)
                duration = torch.rand(batch_size, 1).to(self.device) * 2 + 1
                
                logits = self.model(tokens[:, :-1], emotion=emotions, duration=duration)
                
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    ignore_index=0
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        # Resume from checkpoint if specified
        if resume_from:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler
            )
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch']}")
        
        # Training loop
        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % self.config.validate_every_n_epochs == 0:
                val_loss = self.validate()
                print(f"Val loss: {val_loss:.4f}")
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best model! Val loss: {val_loss:.4f}")
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    train_loss,
                    val_loss,
                    self.config,
                    is_best=is_best
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    train_loss,
                    0.0,  # No val loss for periodic saves
                    self.config,
                    is_best=False
                )
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Generate RL evaluation report if enabled
        if self.rl_evaluator:
            self.rl_evaluator.save_metrics()
            self.rl_evaluator.plot_training_progress()
            report = self.rl_evaluator.generate_report()
            print("\n" + report)
