# Emotion Class Balancing Guide

This guide explains how to balance emotion classes in your MIDI datasets for better model training.

## Why Balance Emotion Classes?

Imbalanced datasets can lead to:
- **Biased models** that favor majority classes
- **Poor performance** on minority emotion classes
- **Reduced generalization** ability
- **Unfair predictions** across emotions

Balancing ensures each emotion category has equal representation during training.

## Quick Start

### 1. Analyze Your Dataset

Check the emotion distribution:

```bash
python scripts/analyze_balance.py analyze ./EMOPIA_1.0
```

Output shows:
- Total samples per emotion
- Percentage distribution
- Imbalance ratio (max/min class sizes)

### 2. Choose a Balancing Strategy

Three main strategies available:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Oversample** | Duplicate minority classes | When you have enough data and want maximum training samples |
| **Undersample** | Remove majority classes | When you have limited compute or want faster training |
| **Hybrid** | Balance to median size | Best compromise between data size and balance |

### 3. Configure Balancing

In your config file or code:

```python
from src.config import DataConfig

config = DataConfig(
    dataset_path="./EMOPIA_1.0",
    balance_emotions=True,
    balancing_strategy="oversample",  # or "undersample", "hybrid"
    use_stratified_split=True
)
```

## Balancing Strategies Explained

### Oversample Strategy

**How it works:** Randomly duplicates samples from minority classes until all classes have the same size as the largest class.

**Pros:**
- No data loss
- Maximum training samples
- Good for small datasets

**Cons:**
- Increased training time
- Risk of overfitting on duplicated samples
- Larger memory footprint

**Example:**
```
Before: joy=100, sadness=50, anger=75
After:  joy=100, sadness=100, anger=100
Total:  225 → 300 samples (+75)
```

### Undersample Strategy

**How it works:** Randomly removes samples from majority classes until all classes match the smallest class size.

**Pros:**
- Faster training
- Lower memory usage
- Reduces overfitting risk

**Cons:**
- Data loss
- May discard useful information
- Not ideal for small datasets

**Example:**
```
Before: joy=100, sadness=50, anger=75
After:  joy=50, sadness=50, anger=50
Total:  225 → 150 samples (-75)
```

### Hybrid Strategy

**How it works:** Balances to the median class size - oversamples small classes and undersamples large classes.

**Pros:**
- Best of both worlds
- Moderate data size
- Good balance/performance tradeoff

**Cons:**
- Still some data loss
- Still some duplication

**Example:**
```
Before: joy=100, sadness=50, anger=75
After:  joy=75, sadness=75, anger=75
Total:  225 → 225 samples (0)
```

## Stratified Splitting

Stratified splitting ensures each emotion is proportionally represented in train/val/test sets.

### Why Use Stratified Splits?

Without stratification:
```
Train: joy=70, sadness=5, anger=60  ❌ Imbalanced!
Val:   joy=15, sadness=30, anger=10
Test:  joy=15, sadness=15, anger=5
```

With stratification:
```
Train: joy=70, sadness=35, anger=52  ✓ Balanced!
Val:   joy=15, sadness=7, anger=11
Test:  joy=15, sadness=8, anger=12
```

### Enable Stratified Splitting

```python
config = DataConfig(
    use_stratified_split=True,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42
)
```

## CLI Tools

### Analyze Distribution

```bash
# Basic analysis
python scripts/analyze_balance.py analyze ./EMOPIA_1.0

# With additional datasets
python scripts/analyze_balance.py analyze ./EMOPIA_1.0 \
    --additional ./lakh_midi ./maestro

# Save to JSON
python scripts/analyze_balance.py analyze ./EMOPIA_1.0 -o analysis.json
```

### Compare Balancing Strategies

See how different strategies affect your data:

```bash
python scripts/analyze_balance.py compare ./EMOPIA_1.0
```

Shows side-by-side comparison of oversample, undersample, and hybrid strategies.

### Test Balancing

Preview balancing results:

```bash
# Test oversample strategy
python scripts/analyze_balance.py balance ./EMOPIA_1.0 -s oversample

# Test undersample strategy
python scripts/analyze_balance.py balance ./EMOPIA_1.0 -s undersample

# Test hybrid strategy
python scripts/analyze_balance.py balance ./EMOPIA_1.0 -s hybrid
```

### Analyze Splits

Check stratified split distribution:

```bash
python scripts/analyze_balance.py split ./EMOPIA_1.0 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    -o split_indices.json
```

## Programmatic Usage

### Basic Balancing

```python
from src.data_balancing import EmotionBalancer

# Load your samples
samples = [
    {"path": "song1.mid", "emotion": "joy"},
    {"path": "song2.mid", "emotion": "sadness"},
    # ...
]

# Balance
balancer = EmotionBalancer(strategy="oversample", seed=42)
balanced_samples = balancer.balance(samples)
```

### Stratified Splitting

```python
from src.data_balancing import StratifiedSplitter

splitter = StratifiedSplitter(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

train, val, test = splitter.split(samples, stratify_key="emotion")
```

### Analyze Distribution

```python
from src.data_balancing import analyze_emotion_distribution, print_distribution_analysis

stats = analyze_emotion_distribution(samples)
print_distribution_analysis(stats)

# Access statistics
print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
print(f"Min class size: {stats['min_class_size']}")
print(f"Max class size: {stats['max_class_size']}")
```

### Using with Dataset

```python
from src.dataset import EMOPIADataset
from src.config import DataConfig

config = DataConfig(
    dataset_path="./EMOPIA_1.0",
    balance_emotions=True,
    balancing_strategy="hybrid",
    use_stratified_split=True
)

# Automatically balanced and stratified
train_dataset = EMOPIADataset(config, split="train")
val_dataset = EMOPIADataset(config, split="val")
test_dataset = EMOPIADataset(config, split="test")
```

## Configuration Options

### DataConfig Parameters

```python
@dataclass
class DataConfig:
    # Balancing
    balance_emotions: bool = True
    balancing_strategy: str = "oversample"  # oversample, undersample, hybrid, weighted
    
    # Splitting
    use_stratified_split: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    
    # Save split indices for reproducibility
    save_split_indices: bool = True
    split_indices_path: str = "./data/split_indices.json"
```

### YAML Configuration

```yaml
data:
  dataset_path: "./EMOPIA_1.0"
  
  # Balancing
  balance_emotions: true
  balancing_strategy: "oversample"
  
  # Stratified splitting
  use_stratified_split: true
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  seed: 42
  
  # Reproducibility
  save_split_indices: true
  split_indices_path: "./data/split_indices.json"
```

## Best Practices

### 1. Always Analyze First

Before training, check your distribution:

```bash
python scripts/analyze_balance.py analyze ./EMOPIA_1.0
```

Look for:
- Imbalance ratio > 2.0 (high imbalance)
- Very small minority classes
- Missing emotion categories

### 2. Choose Strategy Based on Data Size

| Dataset Size | Recommended Strategy |
|--------------|---------------------|
| < 1000 samples | Oversample |
| 1000-10000 samples | Hybrid |
| > 10000 samples | Undersample or Hybrid |

### 3. Use Stratified Splits

Always enable stratified splitting to maintain emotion distribution across splits:

```python
config.use_stratified_split = True
```

### 4. Save Split Indices

For reproducibility, save split indices:

```python
config.save_split_indices = True
config.split_indices_path = "./data/split_indices.json"
```

### 5. Balance Only Training Set

The system automatically balances only the training set, keeping val/test sets unbalanced for realistic evaluation.

### 6. Monitor Training Metrics

After balancing, monitor per-class metrics:
- Accuracy per emotion
- Confusion matrix
- F1-score per class

### 7. Experiment with Strategies

Try different strategies and compare results:

```bash
python scripts/analyze_balance.py compare ./EMOPIA_1.0
```

## Advanced: Weighted Sampling

Instead of resampling, use weighted sampling during training:

```python
config = DataConfig(
    balance_emotions=True,
    balancing_strategy="weighted"
)

# Samples get weight attribute
dataset = EMOPIADataset(config, split="train")

# Use with WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler

weights = [sample["weight"] for sample in dataset.samples]
sampler = WeightedRandomSampler(weights, len(weights))

dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

## Troubleshooting

### "Imbalance ratio too high"

If ratio > 5.0:
1. Check for data collection issues
2. Consider collecting more minority class data
3. Use oversample strategy
4. Consider weighted sampling

### "Not enough samples after undersampling"

If undersample reduces data too much:
1. Switch to hybrid strategy
2. Use oversample instead
3. Collect more data for minority classes

### "Overfitting on duplicated samples"

If oversample causes overfitting:
1. Increase data augmentation
2. Use hybrid strategy
3. Add regularization to model
4. Use weighted sampling instead

### "Splits not stratified"

Ensure:
```python
config.use_stratified_split = True
```

Check split distribution:
```bash
python scripts/analyze_balance.py split ./EMOPIA_1.0
```

## Example Workflow

Complete workflow for balanced training:

```bash
# 1. Analyze original distribution
python scripts/analyze_balance.py analyze ./EMOPIA_1.0

# 2. Compare strategies
python scripts/analyze_balance.py compare ./EMOPIA_1.0

# 3. Test chosen strategy
python scripts/analyze_balance.py balance ./EMOPIA_1.0 -s hybrid

# 4. Create stratified splits
python scripts/analyze_balance.py split ./EMOPIA_1.0 \
    -o ./data/split_indices.json

# 5. Train with balanced data
python train.py --config configs/balanced_training.yaml
```

## References

- **Requirement 2.3**: Balance emotion classes across dataset
- **Requirement 2.6**: Implement proper train/val/test splitting
- See also: [Dataset Integration Guide](DATASET_INTEGRATION.md)
