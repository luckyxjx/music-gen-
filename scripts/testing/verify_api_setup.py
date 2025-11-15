#!/usr/bin/env python3
"""
Verify API is set up correctly to use trained model
"""

from pathlib import Path
import torch

print("="*60)
print("API SETUP VERIFICATION")
print("="*60)

# Check checkpoint exists
checkpoint_path = "checkpoints/best_epoch_24_loss_1.8154.pt"
if Path(checkpoint_path).exists():
    print(f"\n✓ Checkpoint found: {checkpoint_path}")
    
    # Load and check
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
else:
    print(f"\n✗ Checkpoint NOT found: {checkpoint_path}")
    print("  API will use untrained model!")

# Check soundfont
if Path("soundfont.sf2").exists():
    print(f"\n✓ SoundFont found: soundfont.sf2")
else:
    print(f"\n✗ SoundFont NOT found")

# Check API file
if Path("api.py").exists():
    with open("api.py", "r") as f:
        content = f.read()
        
    if "use_demo = data.get('use_demo', False)" in content:
        print(f"\n✓ API configured to use trained model (use_demo=False)")
    else:
        print(f"\n⚠️  API may be using demo mode")
    
    if "best_epoch_24_loss_1.8154.pt" in content:
        print(f"✓ API will load trained checkpoint")
    else:
        print(f"⚠️  API may not load checkpoint")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("\n1. Start API:")
print("   python api.py")
print("\n2. Check console output for:")
print("   ✓ Loading trained checkpoint: checkpoints/best_epoch_24_loss_1.8154.pt")
print("   ✓ Loaded checkpoint from epoch 24")
print("\n3. Test generation:")
print("   curl -X POST http://localhost:5001/api/generate \\")
print("     -H 'Content-Type: application/json' \\")
print("     -d '{\"text\": \"happy music\", \"use_demo\": false}'")
print("\n4. Start frontend:")
print("   cd client && npm run dev")
print("\n" + "="*60)
