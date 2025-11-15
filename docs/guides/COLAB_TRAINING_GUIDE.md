# 🚀 Colab T4 GPU Training - Best Quality Settings

## Quick Start

### 1. Enable T4 GPU
Runtime → Change runtime type → GPU → T4

### 2. Upload Project
```python
from google.colab import files
# Upload your project zip
uploaded = files.upload()
!unzip music-gen.zip
%cd music-gen
```

### 3. Install Dependencies
```python
!pip install torch pretty_midi mido flask numpy pandas tqdm pyyaml
```

### 4. Upload Checkpoint
```python
!mkdir -p checkpoints
# Upload best_epoch_24_loss_1.8154.pt to checkpoints/
```

### 5. Start Training
```python
!python train_colab_optimized.py
```

## Optimized Settings (Already Configured)

### Model: BEST QUALITY
- d_model: 512
- n_layers: 8 (more layers!)
- dropout: 0.15 (better generalization)
- max_seq_len: 1024 (longer music)

### Training: T4 OPTIMIZED
- batch_size: 12 (optimal for T4)
- learning_rate: 1e-4
- epochs: 150 (best quality)
- augmentation: Aggressive (±7 semitones, ±15% tempo)

### Generation: OPTIMAL
- temperature: 0.75
- top_k: 60
- top_p: 0.92
- max_tokens: 3072
- min_notes: 100

## Expected Results

| Epoch | Time | Loss | Quality | Notes |
|-------|------|------|---------|-------|
| 50 | 4-6h | 1.2-1.5 | Good | 100-300 |
| 100 | 10-12h | 0.9-1.1 | Very Good | 200-500 |
| 150 | 16-20h | 0.7-0.9 | Excellent | 300-700 |

## Download Checkpoints
```python
from google.colab import files
import glob
for f in glob.glob('checkpoints/best_*.pt'):
    files.download(f)
```

**Everything is optimized for best quality on T4 GPU!** 🎵
