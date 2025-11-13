# Training Guide - EMOPIA Music Generation

## Quick Start

```bash
python train.py
```

That's it! The script will handle everything automatically.

---

## What Happens During Training

### 1. **Data Loading** (1-2 minutes)
- Loads EMOPIA dataset (~1000 MIDI files)
- Balances emotion classes (oversample minority classes)
- Splits into train/val/test (70/15/15)
- Tokenizes MIDI files into sequences

### 2. **Model Initialization** (10 seconds)
- Creates Transformer model (~26M parameters)
- Initializes emotion embeddings (6 emotions)
- Sets up duration control
- Moves model to GPU/MPS/CPU

### 3. **Training Loop** (several hours)
- **Per Epoch**: ~5-10 minutes (depends on hardware)
- **Total Time**: ~4-8 hours for 50 epochs
- Saves checkpoints every 5 epochs
- Validates after each epoch
- Tracks best model automatically

### 4. **Checkpointing**
- Saves model every 5 epochs
- Keeps best 3 models based on validation loss
- Can resume training if interrupted

---

## Training Options

### Quick Training (Test Run)
Edit `train.py` line 73:
```python
num_epochs=5,  # Change from 50 to 5
```

**Time**: ~30 minutes  
**Quality**: Poor (for testing only)

### Standard Training (Recommended)
```python
num_epochs=50,
batch_size=8,
```

**Time**: ~4-8 hours  
**Quality**: Good music generation

### Long Training (Best Quality)
```python
num_epochs=100,
batch_size=16,  # If you have enough GPU memory
```

**Time**: ~12-24 hours  
**Quality**: Excellent music generation

---

## Hardware Requirements

### Minimum (CPU Only)
- **RAM**: 8GB
- **Storage**: 5GB
- **Time per epoch**: ~20 minutes
- **Total time (50 epochs)**: ~16 hours

### Recommended (GPU/MPS)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (or Apple Silicon M1/M2)
- **RAM**: 16GB
- **Storage**: 5GB
- **Time per epoch**: ~5 minutes
- **Total time (50 epochs)**: ~4 hours

### Optimal (High-end GPU)
- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM)
- **RAM**: 32GB
- **Storage**: 10GB
- **Time per epoch**: ~2 minutes
- **Total time (50 epochs)**: ~2 hours

---

## Monitoring Training

### During Training
You'll see:
```
Epoch 1/50
Training: 100%|████████| 217/217 [05:23<00:00, loss=4.23]
Validation: 100%|████████| 40/40 [00:45<00:00]
Train loss: 4.2345
Val loss: 4.1234
New best model! Val loss: 4.1234
✓ Checkpoint saved: checkpoints/best_epoch_1_loss_4.1234.pt
```

### What to Watch
- **Train loss**: Should decrease steadily
- **Val loss**: Should decrease (if it increases, model is overfitting)
- **Best model**: Saved when val loss improves

### Good Training Signs
✅ Loss decreases from ~6.0 to ~2.0  
✅ Val loss follows train loss  
✅ No NaN or Inf values  
✅ Checkpoints save successfully

### Bad Training Signs
❌ Loss stays at ~6.0 (not learning)  
❌ Val loss increases while train loss decreases (overfitting)  
❌ NaN or Inf values (gradient explosion)  
❌ Out of memory errors

---

## Troubleshooting

### Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in `train.py`:
```python
batch_size=4,  # Reduce from 8 to 4
```

Or reduce model size:
```python
d_model=256,  # Reduce from 512
n_layers=4,   # Reduce from 6
```

### Training Too Slow
**Problem**: Each epoch takes >30 minutes

**Solutions**:
1. Use GPU instead of CPU
2. Reduce dataset size (use fewer samples)
3. Reduce model size
4. Reduce sequence length:
```python
max_seq_len=256,  # Reduce from 512
```

### Loss Not Decreasing
**Problem**: Loss stays at ~6.0 after many epochs

**Solutions**:
1. Check learning rate (try 5e-5 or 2e-4)
2. Check data loading (ensure MIDI files are valid)
3. Increase model size
4. Train longer

### Model Overfitting
**Problem**: Train loss decreases but val loss increases

**Solutions**:
1. Increase dropout:
```python
dropout=0.2,  # Increase from 0.1
```

2. Add more data augmentation
3. Reduce model size
4. Stop training earlier (use best checkpoint)

---

## Resume Training

If training is interrupted:

```bash
python train.py
```

The trainer will automatically detect and load the latest checkpoint!

Or manually specify:
```python
trainer.train(resume_from='./checkpoints/checkpoint_epoch_10.pt')
```

---

## After Training

### 1. Find Your Best Model
```bash
ls -lh checkpoints/best_*.pt
```

Look for the file with lowest loss, e.g.:
```
best_epoch_45_loss_2.1234.pt  ← Use this one!
```

### 2. Generate Music
Update `generate_music.py` to load your trained model:

```python
# Add this before creating the model
checkpoint_path = "./checkpoints/best_epoch_45_loss_2.1234.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```

Then run:
```bash
python generate_music.py
```

### 3. Test Quality
Generate music for each emotion and listen:
- Does happy music sound upbeat?
- Does sad music sound melancholic?
- Is the music coherent (not random noise)?

---

## Training Tips

### 1. Start Small
- Train for 5 epochs first to test everything works
- Then train for 50 epochs

### 2. Monitor Regularly
- Check training every 30 minutes
- Look at loss curves
- Generate samples after 10-20 epochs to check quality

### 3. Save Everything
- Don't delete checkpoints until training is complete
- Keep logs for analysis
- Save generated samples for comparison

### 4. Experiment
- Try different learning rates (1e-5 to 1e-3)
- Try different model sizes
- Try different batch sizes
- Compare results

---

## Expected Results

### After 10 Epochs
- Loss: ~3.5
- Music: Somewhat coherent, recognizable notes
- Emotion: Weak emotion correlation

### After 30 Epochs
- Loss: ~2.5
- Music: Coherent melodies, good rhythm
- Emotion: Moderate emotion correlation

### After 50 Epochs
- Loss: ~2.0
- Music: High-quality melodies, good structure
- Emotion: Strong emotion correlation

### After 100 Epochs
- Loss: ~1.5
- Music: Excellent quality, complex structures
- Emotion: Very strong emotion correlation

---

## Training Checklist

Before starting:
- [ ] EMOPIA dataset downloaded and in `./EMOPIA_1.0/`
- [ ] At least 5GB free disk space
- [ ] Python dependencies installed
- [ ] GPU drivers installed (if using GPU)

During training:
- [ ] Monitor loss values
- [ ] Check checkpoints are saving
- [ ] Ensure no errors in logs
- [ ] System not overheating

After training:
- [ ] Best model identified
- [ ] Generated sample music
- [ ] Tested music quality
- [ ] Saved best checkpoint

---

## Quick Commands

```bash
# Start training
python train.py

# Monitor training (in another terminal)
tail -f logs/music_generation/log.txt

# Check checkpoints
ls -lh checkpoints/

# Generate music after training
python generate_music.py

# Test specific checkpoint
python -c "import torch; print(torch.load('checkpoints/best_epoch_45_loss_2.1234.pt')['epoch'])"
```

---

## Need Help?

Common issues:
1. **Dataset not found**: Ensure `./EMOPIA_1.0/` exists
2. **Out of memory**: Reduce batch size
3. **Slow training**: Use GPU or reduce model size
4. **Bad music**: Train longer or adjust hyperparameters

The training script is designed to be robust and handle most issues automatically!
