# Complete Training Guide - Get Real Music Instead of Random Notes

## Why You're Getting Random Notes

Your model is currently **untrained** - it's generating random tokens because it hasn't learned from any music data yet. To get proper music, you need to train the model on a dataset of MIDI files.

---

## Quick Start (Fastest Way to Train)

### Option 1: Use EMOPIA Dataset (Recommended)

1. **Download EMOPIA Dataset**
   ```bash
   # Download from: https://zenodo.org/record/5090631
   # Or use wget:
   wget https://zenodo.org/record/5090631/files/EMOPIA_1.0.zip
   unzip EMOPIA_1.0.zip
   ```

2. **Verify Dataset Structure**
   ```
   EMOPIA_1.0/
   ├── Q1/  # Joy (High Valence, High Arousal)
   ├── Q2/  # Anger (Low Valence, High Arousal)
   ├── Q3/  # Sadness (Low Valence, Low Arousal)
   └── Q4/  # Calm (High Valence, Low Arousal)
   ```

3. **Start Training**
   ```bash
   python train.py
   ```

That's it! Training will start automatically.

---

## Option 2: Use Your Own MIDI Files

If you don't have EMOPIA, you can use any MIDI files:

### Step 1: Organize Your MIDI Files

Create this structure:
```
my_dataset/
├── joy/
│   ├── song1.mid
│   ├── song2.mid
│   └── ...
├── sadness/
│   ├── song1.mid
│   └── ...
├── anger/
│   └── ...
├── calm/
│   └── ...
├── surprise/
│   └── ...
└── fear/
    └── ...
```

### Step 2: Create a Simple Training Script

```python
# quick_train.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.config import DataConfig, ModelConfig, TrainingConfig, TokenizerConfig
from src.dataset import EMOPIADataset, collate_fn
from src.tokenizer import MIDITokenizer
from src.model import create_model
from src.training.trainer import Trainer

# Configure for your dataset
data_config = DataConfig(
    dataset_path="./my_dataset",  # Your folder
    balance_emotions=True,
    train_split=0.8,
    val_split=0.2,
    test_split=0.0
)

# Create tokenizer
tokenizer = MIDITokenizer(TokenizerConfig())

# Load datasets
train_dataset = EMOPIADataset(data_config, split="train", tokenizer=tokenizer, max_seq_len=512)
val_dataset = EMOPIADataset(data_config, split="val", tokenizer=tokenizer, max_seq_len=512)

# Create small model (faster training)
model_config = ModelConfig(
    d_model=256,      # Smaller model
    n_layers=4,       # Fewer layers
    n_heads=4,        # Fewer heads
    d_ff=1024,        # Smaller feedforward
    max_seq_len=512
)

model = create_model(model_config, tokenizer.vocab_size)

# Training config
training_config = TrainingConfig(
    batch_size=4,           # Small batch for faster iteration
    learning_rate=1e-4,
    num_epochs=10,          # Start with 10 epochs
    checkpoint_dir="./checkpoints",
    validate_every_n_epochs=1
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Train
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
trainer = Trainer(model, train_loader, val_loader, training_config, device)
trainer.train()
```

Run it:
```bash
python quick_train.py
```

---

## Option 3: Download Free MIDI Datasets

### Lakh MIDI Dataset (Large, 176k files)
```bash
# Download from: https://colinraffel.com/projects/lmd/
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
```

### MAESTRO Dataset (Classical Piano)
```bash
# Download from: https://magenta.tensorflow.org/datasets/maestro
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip
```

---

## Training Configuration Guide

### For Fast Experimentation (CPU/Small GPU)
```python
ModelConfig(
    d_model=128,          # Very small
    n_layers=2,           # Just 2 layers
    n_heads=2,            # 2 attention heads
    d_ff=512,             # Small feedforward
    max_seq_len=256       # Short sequences
)

TrainingConfig(
    batch_size=2,         # Tiny batches
    num_epochs=5,         # Quick test
    learning_rate=1e-3    # Higher LR for faster learning
)
```

**Training time:** ~10-30 minutes on CPU

### For Good Quality (GPU Recommended)
```python
ModelConfig(
    d_model=512,          # Standard size
    n_layers=6,           # 6 layers
    n_heads=8,            # 8 attention heads
    d_ff=2048,            # Standard feedforward
    max_seq_len=512       # Medium sequences
)

TrainingConfig(
    batch_size=16,        # Larger batches
    num_epochs=25,        # More epochs
    learning_rate=1e-4    # Standard LR
)
```

**Training time:** ~2-4 hours on GPU

### For Best Quality (Powerful GPU)
```python
ModelConfig(
    d_model=768,          # Large model
    n_layers=12,          # Deep network
    n_heads=12,           # Many heads
    d_ff=3072,            # Large feedforward
    max_seq_len=1024      # Long sequences
)

TrainingConfig(
    batch_size=32,        # Large batches
    num_epochs=50,        # Many epochs
    learning_rate=5e-5    # Lower LR
)
```

**Training time:** ~8-12 hours on powerful GPU

---

## Monitoring Training

### Watch Training Progress
```bash
# Training will show:
# Epoch 1/25 - Train loss: 4.5234, Val loss: 4.3421
# Epoch 2/25 - Train loss: 3.8765, Val loss: 3.7123
# ...
```

**Good signs:**
- Loss decreasing over time
- Validation loss following training loss
- No NaN or Inf values

**Bad signs:**
- Loss not decreasing after 5+ epochs
- Validation loss much higher than training loss (overfitting)
- NaN values (learning rate too high)

### Check Checkpoints
```bash
ls checkpoints/
# You should see:
# best_epoch_5_loss_3.2145.pt
# checkpoint_epoch_10.pt
# ...
```

---

## Testing Your Trained Model

### Quick Test
```python
# test_generation.py
import torch
from src.model import create_model
from src.tokenizer import MIDITokenizer
from src.config import ModelConfig, TokenizerConfig

# Load tokenizer
tokenizer = MIDITokenizer(TokenizerConfig())

# Create model
model_config = ModelConfig(d_model=512, n_layers=6, n_heads=8)
model = create_model(model_config, tokenizer.vocab_size)

# Load trained weights
checkpoint = torch.load('checkpoints/best_epoch_10_loss_2.5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Create emotion tensor (0=joy, 1=sadness, 2=anger, 3=calm, 4=surprise, 5=fear)
emotion = torch.tensor([0], device=device)  # Joy

# Generate tokens
start_token = torch.tensor([[tokenizer.bos_token_id]], device=device)
generated = model.generate(
    start_token,
    emotion=emotion,
    max_length=256,
    temperature=1.0,
    top_k=10
)

# Decode to MIDI
midi = tokenizer.decode(generated[0].cpu().tolist())
midi.write('test_output.mid')
print("Generated: test_output.mid")
```

---

## Troubleshooting

### Problem: "No MIDI files found"
**Solution:** Check your dataset path and folder structure

### Problem: "CUDA out of memory"
**Solution:** Reduce batch_size or model size:
```python
TrainingConfig(batch_size=2)  # Smaller batches
ModelConfig(d_model=256, n_layers=4)  # Smaller model
```

### Problem: "Loss is NaN"
**Solution:** Lower learning rate:
```python
TrainingConfig(learning_rate=1e-5)  # Lower LR
```

### Problem: "Training is too slow"
**Solution:** 
- Use GPU if available
- Reduce model size
- Reduce sequence length
- Use fewer training samples

### Problem: "Generated music still sounds random"
**Solution:**
- Train for more epochs (at least 10-20)
- Use more training data
- Check that loss is actually decreasing
- Try lower temperature during generation (0.7-0.9)

---

## Minimal Training Example (No Dataset Required)

If you just want to test the training pipeline:

```python
# minimal_train.py
import torch
from torch.utils.data import Dataset, DataLoader
from src.model import create_model
from src.config import ModelConfig
from src.tokenizer import MIDITokenizer, TokenizerConfig

# Create dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.tokenizer = MIDITokenizer(TokenizerConfig())
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random tokens
        tokens = torch.randint(0, 500, (256,))
        emotion = torch.randint(0, 6, (1,)).item()
        return {'tokens': tokens, 'emotion': emotion}

# Setup
tokenizer = MIDITokenizer(TokenizerConfig())
model = create_model(ModelConfig(d_model=128, n_layers=2, n_heads=2), tokenizer.vocab_size)
dataset = DummyDataset()
loader = DataLoader(dataset, batch_size=4)

# Simple training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for epoch in range(5):
    total_loss = 0
    for batch in loader:
        tokens = batch['tokens']
        emotions = batch['emotion']
        
        # Forward
        logits = model(tokens[:, :-1], emotion=emotions)
        
        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/5 - Loss: {total_loss/len(loader):.4f}")

print("Training complete!")
```

This will verify your training pipeline works.

---

## Recommended Training Path

### Day 1: Quick Test
1. Download EMOPIA dataset (or use 10-20 MIDI files)
2. Train small model for 5 epochs
3. Test generation
4. **Goal:** Verify training works

### Day 2-3: Proper Training
1. Train medium model for 25 epochs
2. Monitor loss curves
3. Test different emotions
4. **Goal:** Get coherent music

### Day 4+: Optimization
1. Train large model for 50+ epochs
2. Experiment with hyperparameters
3. Use RL fine-tuning (Phase 5)
4. **Goal:** High-quality music generation

---

## Expected Results

### After 5 Epochs
- Loss: ~3.5-4.0
- Music: Basic note sequences, some patterns
- Quality: 3/10

### After 25 Epochs
- Loss: ~2.0-2.5
- Music: Recognizable melodies, rhythm patterns
- Quality: 6/10

### After 50+ Epochs
- Loss: ~1.5-2.0
- Music: Coherent compositions, emotion-appropriate
- Quality: 8/10

---

## Next Steps After Training

1. **Test Generation**
   ```bash
   python generate_music.py --emotion joy --duration 2
   ```

2. **Use API**
   ```bash
   python api.py
   # Then use frontend at http://localhost:5001
   ```

3. **RL Fine-Tuning** (Phase 5)
   ```bash
   python src/training/example_rl_usage.py
   ```

4. **Collect Human Feedback**
   - Navigate to `/feedback` page
   - Rate generated samples
   - Improve model with feedback

---

## Summary

**To get real music instead of random notes:**

1. ✅ Get MIDI dataset (EMOPIA recommended)
2. ✅ Run `python train.py`
3. ✅ Wait for training (2-4 hours for good results)
4. ✅ Test with `python generate_music.py`
5. ✅ Use trained model in API/frontend

**Minimum requirements:**
- 10-20 MIDI files per emotion
- 10 epochs of training
- Loss below 3.0

**For best results:**
- 100+ MIDI files per emotion
- 25-50 epochs of training
- Loss below 2.0
- GPU for faster training

Good luck with training! 🎵
