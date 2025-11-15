# Scripts Directory

This directory contains all utility scripts for training, testing, and working with the music generation model.

## Running Scripts

**Important:** All scripts should be run from the project root directory, not from within the scripts folder.

### Training Scripts

```bash
# From project root:
python scripts/training/train.py                    # Initial training
python scripts/training/train_continued.py          # Continue from checkpoint
python scripts/training/train_colab_optimized.py    # Optimized for Colab T4 GPU
python scripts/training/rl_finetune.py             # RL fine-tuning
python scripts/training/quick_start_training.py     # Quick start guide
```

### Testing Scripts

```bash
# From project root:
python scripts/testing/test_trained_model.py        # Test model generation
python scripts/testing/test_api.py                  # Test API endpoints
python scripts/testing/verify_api_setup.py          # Verify API configuration
```

### Utility Scripts

```bash
# From project root:
python scripts/utilities/generate_music.py          # Generate music samples
python scripts/utilities/create_demo_midi.py        # Create demo MIDI files
python scripts/utilities/show_model.py              # Display model architecture
python scripts/analyze_balance.py                   # Analyze dataset balance
python scripts/prepare_dataset.py                   # Prepare dataset for training
```

## Directory Structure

```
scripts/
├── training/          # Training-related scripts
├── testing/           # Testing and verification scripts
├── utilities/         # Helper utilities
├── analyze_balance.py # Dataset analysis
└── prepare_dataset.py # Dataset preparation
```

## Notes

- All paths in scripts are relative to the project root
- Make sure you're in the project root directory before running any script
- Training scripts will save checkpoints to `checkpoints/` directory
- Generated outputs go to `output/` directory
