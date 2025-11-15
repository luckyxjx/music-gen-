"""
Example usage of RL fine-tuning with evaluation tracking
Demonstrates Phase 5 Task 5.3: Track and evaluate RL improvements
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.training.rl_evaluator import RLEvaluator


def example_basic_evaluation():
    """Basic example of using RLEvaluator"""
    print("="*60)
    print("Example 1: Basic RL Evaluation Tracking")
    print("="*60)
    
    # Initialize evaluator
    evaluator = RLEvaluator(save_dir="examples/rl_eval_demo")
    
    # Set pre-RL baseline
    pre_rl_metrics = {
        'total_reward': 0.45,
        'emotion_accuracy': 0.72
    }
    evaluator.set_pre_rl_baseline(pre_rl_metrics)
    
    # Simulate training episodes
    print("\nSimulating RL training...")
    for episode in range(100):
        # Simulate improving metrics
        metrics = {
            'total_reward': 0.45 + (episode / 100) * 0.15,  # Improve from 0.45 to 0.60
            'emotion_reward': 0.50 + (episode / 100) * 0.10,
            'coherence_reward': 0.40 + (episode / 100) * 0.05,
            'diversity_reward': 0.45 + (episode / 100) * 0.08,
            'emotion_accuracy': 0.72 + (episode / 100) * 0.10,  # Improve from 0.72 to 0.82
            'policy_loss': 2.0 - (episode / 100) * 0.5,
            'baseline_loss': 1.5 - (episode / 100) * 0.3
        }
        
        evaluator.update(episode, metrics)
    
    # Generate report
    print("\n" + evaluator.generate_report())
    
    # Plot progress
    evaluator.plot_training_progress()
    
    # Get improvement stats
    stats = evaluator.get_improvement_stats()
    print("\nImprovement Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")


def example_model_comparison():
    """Example of comparing pre-RL and post-RL models"""
    print("\n" + "="*60)
    print("Example 2: Model Comparison")
    print("="*60)
    
    # This would use actual models in practice
    print("\nIn practice, you would:")
    print("1. Generate samples from pre-RL model")
    print("2. Train with RL")
    print("3. Generate samples from post-RL model")
    print("4. Compare emotion accuracy using evaluator.compare_models()")
    
    print("\nExample code:")
    print("""
    evaluator = RLEvaluator()
    
    # Generate samples
    pre_rl_samples = [model.generate(...) for _ in range(100)]
    # ... train with RL ...
    post_rl_samples = [model.generate(...) for _ in range(100)]
    
    # Compare
    comparison = evaluator.compare_models(
        pre_rl_samples, post_rl_samples,
        target_emotions, emotion_classifier
    )
    """)


def example_integration_with_training():
    """Example of integrating with existing training pipeline"""
    print("\n" + "="*60)
    print("Example 3: Integration with Training Pipeline")
    print("="*60)
    
    print("\nTo integrate RL evaluation with your training:")
    print("""
    from src.training.trainer import Trainer
    from src.training.rl_integration import run_rl_training_with_eval
    
    # Standard training
    trainer = Trainer(model, train_loader, val_loader, config, enable_rl_eval=True)
    trainer.train()
    
    # RL fine-tuning with evaluation
    rl_config = {
        'num_episodes': 1000,
        'policy_lr': 1e-5,
        'eval_interval': 50,
        'test_emotions': [0, 1, 2, 3, 4, 5]
    }
    
    evaluator = run_rl_training_with_eval(
        model, tokenizer, emotion_classifier, rl_config, device
    )
    
    # Evaluator automatically:
    # - Tracks reward progression
    # - Monitors emotion accuracy
    # - Generates plots
    # - Creates evaluation reports
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RL EVALUATION EXAMPLES - Phase 5 Task 5.3")
    print("="*70)
    
    # Run examples
    example_basic_evaluation()
    example_model_comparison()
    example_integration_with_training()
    
    print("\n" + "="*70)
    print("Examples complete! Check 'examples/rl_eval_demo' for outputs")
    print("="*70)
