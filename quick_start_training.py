#!/usr/bin/env python3
"""
Quick Start Training Script
Automatically downloads sample data and starts training
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile

def check_dataset():
    """Check if EMOPIA dataset exists"""
    emopia_path = Path("./EMOPIA_1.0")
    if emopia_path.exists():
        print("✓ EMOPIA dataset found!")
        return True
    
    print("✗ EMOPIA dataset not found")
    return False

def check_any_midi_files():
    """Check if there are any MIDI files in common locations"""
    search_paths = [
        "./data",
        "./midi_files",
        "./dataset",
        "./music",
        "."
    ]
    
    for path in search_paths:
        if Path(path).exists():
            midi_files = list(Path(path).rglob("*.mid")) + list(Path(path).rglob("*.midi"))
            if midi_files:
                print(f"✓ Found {len(midi_files)} MIDI files in {path}")
                return True
    
    return False

def download_sample_midis():
    """Download a few sample MIDI files for testing"""
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE MIDI FILES")
    print("="*60)
    
    # Create sample dataset directory
    sample_dir = Path("./sample_dataset")
    sample_dir.mkdir(exist_ok=True)
    
    # Create emotion folders
    emotions = ["joy", "sadness", "anger", "calm"]
    for emotion in emotions:
        (sample_dir / emotion).mkdir(exist_ok=True)
    
    print("\nNote: For real training, you need a proper dataset.")
    print("This will create a minimal dataset for testing only.")
    print("\nRecommended: Download EMOPIA dataset from:")
    print("https://zenodo.org/record/5090631")
    
    return str(sample_dir)

def create_minimal_training_script():
    """Create a minimal training script for quick testing"""
    script = """#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path

class MinimalMusicModel(nn.Module):
    def __init__(self, vocab_size=500, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512),
            num_layers=n_layers
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

class DummyDataset(Dataset):
    def __init__(self, size=100, seq_len=128):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randint(0, 500, (self.seq_len,))

print("Starting minimal training...")
model = MinimalMusicModel()
dataset = DummyDataset()
loader = DataLoader(dataset, batch_size=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch[:, :-1])
        loss = criterion(output.reshape(-1, 500), batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/5 - Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "minimal_model.pt")
print("\\nTraining complete! Model saved to minimal_model.pt")
"""
    
    with open("minimal_train.py", "w") as f:
        f.write(script)
    
    print("\n✓ Created minimal_train.py for testing")

def main():
    print("\n" + "="*60)
    print("MUSIC GENERATION - QUICK START TRAINING")
    print("="*60)
    
    print("\n[Step 1] Checking for datasets...")
    
    has_emopia = check_dataset()
    has_midi = check_any_midi_files()
    
    if has_emopia:
        print("\n✓ You have EMOPIA dataset!")
        print("\nReady to train. Run:")
        print("  python train.py")
        return
    
    if has_midi:
        print("\n✓ You have MIDI files!")
        print("\nYou can train with your existing MIDI files.")
        print("Organize them into emotion folders and run:")
        print("  python train.py")
        return
    
    print("\n✗ No dataset found")
    print("\n" + "="*60)
    print("OPTIONS TO GET STARTED")
    print("="*60)
    
    print("\n1. RECOMMENDED: Download EMOPIA Dataset")
    print("   - Best quality emotion-labeled music")
    print("   - ~1000 MIDI files")
    print("   - Download from: https://zenodo.org/record/5090631")
    print("   - Extract to ./EMOPIA_1.0/")
    
    print("\n2. Use Your Own MIDI Files")
    print("   - Organize into emotion folders:")
    print("     my_dataset/joy/*.mid")
    print("     my_dataset/sadness/*.mid")
    print("     etc.")
    
    print("\n3. Download Free MIDI Datasets")
    print("   - Lakh MIDI: http://colinraffel.com/projects/lmd/")
    print("   - MAESTRO: https://magenta.tensorflow.org/datasets/maestro")
    
    print("\n4. Test Training Pipeline (No Real Data)")
    print("   - Just verify training works")
    print("   - Won't produce good music")
    
    print("\n" + "="*60)
    choice = input("\nChoose option (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        print("\nPlease download EMOPIA dataset manually:")
        print("1. Visit: https://zenodo.org/record/5090631")
        print("2. Download EMOPIA_1.0.zip")
        print("3. Extract to current directory")
        print("4. Run: python train.py")
    
    elif choice == '2':
        print("\nOrganize your MIDI files like this:")
        print("  my_dataset/")
        print("    joy/song1.mid")
        print("    sadness/song2.mid")
        print("    ...")
        print("\nThen edit train.py to use your dataset path")
    
    elif choice == '3':
        print("\nFree MIDI datasets:")
        print("- Lakh MIDI (176k files): http://colinraffel.com/projects/lmd/")
        print("- MAESTRO (1200 files): https://magenta.tensorflow.org/datasets/maestro")
    
    elif choice == '4':
        print("\nCreating minimal training script...")
        create_minimal_training_script()
        print("\nRun: python minimal_train.py")
        print("\nNote: This won't produce good music, just tests the pipeline")
    
    elif choice.lower() == 'q':
        print("\nExiting...")
        return
    
    else:
        print("\nInvalid choice")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Get a dataset (EMOPIA recommended)")
    print("2. Run: python train.py")
    print("3. Wait for training to complete")
    print("4. Test: python generate_music.py")
    print("5. Use API: python api.py")
    print("\nSee TRAINING_GUIDE_COMPLETE.md for detailed instructions")

if __name__ == "__main__":
    main()
