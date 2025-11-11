#!/usr/bin/env python3
"""
CLI tool for analyzing and balancing emotion distributions in datasets
"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig
from src.dataset_loaders import get_dataset_loader, EMOPIALoader
from src.data_balancing import (
    EmotionBalancer,
    StratifiedSplitter,
    analyze_emotion_distribution,
    print_distribution_analysis
)


def load_all_samples(config: DataConfig):
    """Load all samples from configured datasets"""
    samples = []
    
    # Load EMOPIA
    print("Loading EMOPIA dataset...")
    emopia_loader = EMOPIALoader(config.dataset_path)
    emopia_samples = emopia_loader.load()
    samples.extend(emopia_samples)
    print(f"Loaded {len(emopia_samples)} samples from EMOPIA")
    
    # Load additional datasets
    for dataset_info in config.additional_datasets:
        if isinstance(dataset_info, str):
            dataset_path = dataset_info
            dataset_type = None
        elif isinstance(dataset_info, dict):
            dataset_path = dataset_info.get("path")
            dataset_type = dataset_info.get("type")
        else:
            continue
        
        print(f"Loading {dataset_path}...")
        loader = get_dataset_loader(dataset_path, dataset_type)
        additional_samples = loader.load()
        samples.extend(additional_samples)
        print(f"Loaded {len(additional_samples)} samples")
    
    return samples


def analyze_command(args):
    """Analyze emotion distribution"""
    config = DataConfig(
        dataset_path=args.dataset_path,
        additional_datasets=args.additional or []
    )
    
    samples = load_all_samples(config)
    
    print(f"\nTotal samples loaded: {len(samples)}")
    
    # Analyze distribution
    stats = analyze_emotion_distribution(samples)
    print_distribution_analysis(stats)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


def balance_command(args):
    """Balance emotion classes and show results"""
    config = DataConfig(
        dataset_path=args.dataset_path,
        additional_datasets=args.additional or []
    )
    
    samples = load_all_samples(config)
    
    print("\n" + "="*60)
    print("BEFORE BALANCING")
    print("="*60)
    stats_before = analyze_emotion_distribution(samples)
    print_distribution_analysis(stats_before)
    
    # Apply balancing
    balancer = EmotionBalancer(strategy=args.strategy, seed=args.seed)
    balanced_samples = balancer.balance(samples)
    
    print("\n" + "="*60)
    print("AFTER BALANCING")
    print("="*60)
    stats_after = analyze_emotion_distribution(balanced_samples)
    print_distribution_analysis(stats_after)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Original samples: {len(samples)}")
    print(f"Balanced samples: {len(balanced_samples)}")
    print(f"Change: {len(balanced_samples) - len(samples):+d} samples")
    print(f"Imbalance ratio: {stats_before['imbalance_ratio']:.2f}:1 → {stats_after['imbalance_ratio']:.2f}:1")


def split_command(args):
    """Perform stratified split and analyze"""
    config = DataConfig(
        dataset_path=args.dataset_path,
        additional_datasets=args.additional or [],
        train_split=args.train_ratio,
        val_split=args.val_ratio,
        test_split=args.test_ratio
    )
    
    samples = load_all_samples(config)
    
    # Perform split
    splitter = StratifiedSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    train, val, test = splitter.split(samples)
    
    # Analyze each split
    for split_name, split_samples in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print("\n" + "="*60)
        print(f"{split_name} SPLIT")
        print("="*60)
        stats = analyze_emotion_distribution(split_samples)
        print_distribution_analysis(stats)
    
    # Save indices if requested
    if args.output:
        splitter.save_split_indices(samples, args.output)


def compare_strategies_command(args):
    """Compare different balancing strategies"""
    config = DataConfig(
        dataset_path=args.dataset_path,
        additional_datasets=args.additional or []
    )
    
    samples = load_all_samples(config)
    
    print("\n" + "="*60)
    print("ORIGINAL DISTRIBUTION")
    print("="*60)
    stats_original = analyze_emotion_distribution(samples)
    print_distribution_analysis(stats_original)
    
    strategies = ["oversample", "undersample", "hybrid"]
    
    for strategy in strategies:
        print("\n" + "="*60)
        print(f"STRATEGY: {strategy.upper()}")
        print("="*60)
        
        balancer = EmotionBalancer(strategy=strategy, seed=args.seed)
        balanced = balancer.balance(samples.copy())
        
        stats = analyze_emotion_distribution(balanced)
        print(f"\nTotal samples: {len(balanced)} (original: {len(samples)})")
        print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
        
        for emotion, count in sorted(stats['distribution'].items()):
            original_count = stats_original['distribution'].get(emotion, 0)
            change = count - original_count
            print(f"  {emotion}: {count} ({change:+d})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and balance emotion distributions in MIDI datasets"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze emotion distribution"
    )
    analyze_parser.add_argument(
        "dataset_path",
        help="Path to primary dataset (EMOPIA)"
    )
    analyze_parser.add_argument(
        "--additional",
        nargs="+",
        help="Additional dataset paths"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Save analysis to JSON file"
    )
    
    # Balance command
    balance_parser = subparsers.add_parser(
        "balance",
        help="Balance emotion classes and show results"
    )
    balance_parser.add_argument(
        "dataset_path",
        help="Path to primary dataset"
    )
    balance_parser.add_argument(
        "--additional",
        nargs="+",
        help="Additional dataset paths"
    )
    balance_parser.add_argument(
        "-s", "--strategy",
        choices=["oversample", "undersample", "hybrid"],
        default="oversample",
        help="Balancing strategy (default: oversample)"
    )
    balance_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Split command
    split_parser = subparsers.add_parser(
        "split",
        help="Perform stratified split and analyze"
    )
    split_parser.add_argument(
        "dataset_path",
        help="Path to primary dataset"
    )
    split_parser.add_argument(
        "--additional",
        nargs="+",
        help="Additional dataset paths"
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)"
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    split_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    split_parser.add_argument(
        "-o", "--output",
        help="Save split indices to JSON file"
    )
    
    # Compare strategies command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare different balancing strategies"
    )
    compare_parser.add_argument(
        "dataset_path",
        help="Path to primary dataset"
    )
    compare_parser.add_argument(
        "--additional",
        nargs="+",
        help="Additional dataset paths"
    )
    compare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_command(args)
    elif args.command == "balance":
        balance_command(args)
    elif args.command == "split":
        split_command(args)
    elif args.command == "compare":
        compare_strategies_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
