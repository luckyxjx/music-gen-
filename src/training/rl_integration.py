"""
Integration utilities for RL fine-tuning with existing training pipeline
Minimal wrapper to connect RL components with standard training
"""

import torch
from typing import Dict, Optional
from pathlib import Path

from src.training.rl_evaluator import RLEvaluator


def setup_rl_training(
    generator,
    tokenizer,
    emotion_classifier,
    config: Dict,
    device: str = 'cpu'
):
    """
    Setup RL training components
    
    Args:
        generator: Pre-trained music generation model
        tokenizer: MIDI tokenizer
        emotion_classifier: Trained emotion classifier for rewards
        config: RL training configuration
        device: Device for training
    
    Returns:
        Tuple of (reward_function, policy_trainer, evaluator)
    """
    # Import RL components
    import sys
    sys.path.append('RL-SYSTEM')
    
    from reward_function import RewardFunction, EmotionClassifier
    from policy_gradient import PolicyGradientTrainer, Baseline
    
    # Initialize reward function
    reward_function = RewardFunction(
        emotion_classifier=emotion_classifier,
        emotion_weight=config.get('emotion_weight', 0.6),
        coherence_weight=config.get('coherence_weight', 0.25),
        diversity_weight=config.get('diversity_weight', 0.15),
        device=device
    )
    
    # Initialize baseline
    baseline = Baseline(d_model=config.get('d_model', 256))
    
    # Initialize policy gradient trainer
    policy_trainer = PolicyGradientTrainer(
        generator=generator,
        tokenizer=tokenizer,
        reward_function=reward_function,
        baseline=baseline,
        policy_lr=config.get('policy_lr', 1e-5),
        baseline_lr=config.get('baseline_lr', 1e-4),
        gamma=config.get('gamma', 0.99),
        device=device
    )
    
    # Initialize evaluator
    evaluator = RLEvaluator(save_dir=config.get('eval_dir', 'rl_evaluation'))
    
    return reward_function, policy_trainer, evaluator


def evaluate_pre_rl_baseline(
    generator,
    tokenizer,
    emotion_classifier,
    test_emotions: list,
    num_samples: int = 50,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate pre-RL model performance to establish baseline
    
    Args:
        generator: Pre-trained model (before RL)
        tokenizer: MIDI tokenizer
        emotion_classifier: Emotion classifier
        test_emotions: List of emotions to test
        num_samples: Number of samples per emotion
        device: Device
    
    Returns:
        baseline_metrics: Dictionary of baseline performance metrics
    """
    generator.eval()
    emotion_classifier.eval()
    
    total_correct = 0
    total_samples = 0
    rewards = []
    
    import sys
    sys.path.append('RL-SYSTEM')
    from reward_function import RewardFunction
    
    # Create reward function
    reward_fn = RewardFunction(emotion_classifier, device=device)
    
    with torch.no_grad():
        for emotion in test_emotions:
            for _ in range(num_samples):
                # Generate sample
                emotion_tensor = torch.tensor([emotion], device=device)
                
                # Simple generation (you may need to adjust based on your model)
                start_token = torch.tensor([[tokenizer.bos_token_id]], device=device)
                generated = generator.generate(
                    start_token,
                    emotion=emotion_tensor,
                    max_length=256,
                    temperature=1.0
                )
                
                # Compute reward
                reward, components = reward_fn.compute_reward(generated, emotion)
                rewards.append(reward.item())
                
                # Check emotion accuracy
                logits = emotion_classifier(generated)
                pred = torch.argmax(logits, dim=-1).item()
                if pred == emotion:
                    total_correct += 1
                total_samples += 1
    
    baseline_metrics = {
        'total_reward': float(torch.tensor(rewards).mean()),
        'reward_std': float(torch.tensor(rewards).std()),
        'emotion_accuracy': total_correct / total_samples,
        'num_samples': total_samples
    }
    
    return baseline_metrics


def run_rl_training_with_eval(
    generator,
    tokenizer,
    emotion_classifier,
    config: Dict,
    device: str = 'cpu'
):
    """
    Run complete RL training with evaluation tracking
    
    Args:
        generator: Pre-trained generator model
        tokenizer: MIDI tokenizer
        emotion_classifier: Emotion classifier
        config: Training configuration
        device: Device
    """
    print("Setting up RL training...")
    reward_fn, policy_trainer, evaluator = setup_rl_training(
        generator, tokenizer, emotion_classifier, config, device
    )
    
    # Evaluate pre-RL baseline
    print("\nEvaluating pre-RL baseline...")
    test_emotions = config.get('test_emotions', [0, 1, 2, 3, 4, 5])
    baseline_metrics = evaluate_pre_rl_baseline(
        generator, tokenizer, emotion_classifier,
        test_emotions, num_samples=config.get('baseline_samples', 20),
        device=device
    )
    
    evaluator.set_pre_rl_baseline(baseline_metrics)
    print(f"Pre-RL baseline: {baseline_metrics}")
    
    # Training loop with evaluation
    print("\nStarting RL training...")
    num_episodes = config.get('num_episodes', 1000)
    eval_interval = config.get('eval_interval', 50)
    
    for episode in range(num_episodes):
        # Sample emotion
        import numpy as np
        emotion = np.random.choice(test_emotions)
        
        # Train episode
        metrics = policy_trainer.train_episode(emotion)
        
        # Update evaluator
        evaluator.update(episode, metrics)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last {eval_interval}): {torch.tensor(evaluator.metrics_history['total_reward'][-eval_interval:]).mean():.4f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            
            # Generate plots
            evaluator.plot_training_progress()
    
    # Final evaluation
    print("\n" + "="*60)
    print("RL Training Complete!")
    print("="*60)
    
    evaluator.save_metrics()
    report = evaluator.generate_report()
    print(report)
    
    # Save final model
    save_path = Path(config.get('save_dir', 'rl_checkpoints'))
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'generator': generator.state_dict(),
        'baseline': policy_trainer.baseline.state_dict(),
        'metrics': evaluator.metrics_history,
        'config': config
    }, save_path / 'final_rl_model.pt')
    
    print(f"\nFinal model saved to {save_path / 'final_rl_model.pt'}")
    
    return evaluator


# Example usage
if __name__ == "__main__":
    print("RL Integration Module")
    print("Use this module to integrate RL fine-tuning with your training pipeline")
    print("\nExample:")
    print("""
    from src.training.rl_integration import run_rl_training_with_eval
    
    config = {
        'num_episodes': 1000,
        'policy_lr': 1e-5,
        'baseline_lr': 1e-4,
        'emotion_weight': 0.6,
        'coherence_weight': 0.25,
        'diversity_weight': 0.15,
        'eval_interval': 50,
        'test_emotions': [0, 1, 2, 3, 4, 5]
    }
    
    evaluator = run_rl_training_with_eval(
        generator, tokenizer, emotion_classifier, config, device
    )
    """)
