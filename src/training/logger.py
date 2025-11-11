"""
Experiment tracking and logging
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class ExperimentLogger:
    """Simple experiment logger"""
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.config_file = self.experiment_dir / "config.json"
        
        self.step = 0
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if step is None:
            step = self.step
            self.step += 1
        
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_text(self, text: str, filename: str = "log.txt"):
        """Log text to file"""
        log_file = self.experiment_dir / filename
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")
    
    def save_artifact(self, data: Any, filename: str):
        """Save artifact (JSON serializable)"""
        artifact_path = self.experiment_dir / filename
        with open(artifact_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path"""
        return str(self.experiment_dir)


class WandbLogger:
    """Weights & Biases logger wrapper"""
    
    def __init__(self, project: str, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        self.wandb = None
        
        if enabled:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(project=project, config=config)
            except ImportError:
                print("Warning: wandb not installed. Logging disabled.")
                self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb"""
        if self.enabled and self.wandb is not None:
            self.wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """Update config"""
        if self.enabled and self.wandb is not None:
            self.wandb.config.update(config)
    
    def finish(self):
        """Finish wandb run"""
        if self.enabled and self.wandb is not None:
            self.wandb.finish()


class MultiLogger:
    """Combine multiple loggers"""
    
    def __init__(self, loggers: list):
        self.loggers = loggers
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log config to all loggers"""
        for logger in self.loggers:
            logger.log_config(config)
    
    def log_text(self, text: str, filename: str = "log.txt"):
        """Log text (only for ExperimentLogger)"""
        for logger in self.loggers:
            if isinstance(logger, ExperimentLogger):
                logger.log_text(text, filename)
