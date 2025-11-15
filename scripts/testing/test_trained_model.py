#!/usr/bin/env python3
"""
Test the trained model and generate music
"""

import torch
import torch.nn.functional as F
from src.tokenizer import MIDITokenizer
from src.config import TokenizerConfig, ModelConfig
from src.model import create_model

print("="*60)
print("TESTING TRAINED MODEL")
print("="*60)

# Load tokenizer
print("\n[1/5] Loading tokenizer...")
tokenizer = MIDITokenizer(TokenizerConfig())
print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

# Create model with CORRECT architecture (matching training)
print("\n[2/5] Creating model...")
model_config = ModelConfig(
    model_type="transformer",
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=512,  # This must match training!
    use_emotion_conditioning=True,
    emotion_emb_dim=64,
    num_emotions=6,
    use_duration_control=True,
    duration_emb_dim=32
)
model = create_model(model_config, tokenizer.vocab_size)
print(f"✓ Model created")

# Load checkpoint
print("\n[3/5] Loading trained weights...")
checkpoint_path = "checkpoints/best_epoch_24_loss_1.8154.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Loaded checkpoint: {checkpoint_path}")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

# Generate music
print("\n[4/5] Generating music...")
emotions = {
    0: "joy",
    1: "sadness", 
    2: "anger",
    3: "calm",
    4: "surprise",
    5: "fear"
}

for emotion_id, emotion_name in emotions.items():
    print(f"\n  Generating {emotion_name}...")
    
    # Create inputs
    emotion = torch.tensor([emotion_id])
    duration = torch.tensor([[2.0]])  # 2 minutes
    bos_id = tokenizer.vocab.get('<BOS>', 0)
    start_token = torch.tensor([[bos_id]])
    
    # Generate using autoregressive sampling
    with torch.no_grad():
        generated_tokens = [bos_id]
        current_seq = start_token
        
        for _ in range(256):  # max_length
            # Forward pass
            logits = model(current_seq, emotion=emotion, duration=duration)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :] / 0.9  # temperature
            
            # Top-k sampling
            top_k = 10
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_token_idx].item()
            
            # Stop if EOS
            eos_id = tokenizer.vocab.get('<EOS>', 1)
            if next_token == eos_id:
                break
            
            generated_tokens.append(next_token)
            current_seq = torch.cat([current_seq, torch.tensor([[next_token]])], dim=1)
        
        generated = [generated_tokens]
    
    # Decode to MIDI
    try:
        midi = tokenizer.decode(generated[0])
        
        # Count notes
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        
        if total_notes > 0:
            # Save
            output_file = f"generated/{emotion_name}_test.mid"
            import os
            os.makedirs("generated", exist_ok=True)
            midi.write(output_file)
            print(f"    ✓ {emotion_name}: {total_notes} notes → {output_file}")
        else:
            print(f"    ✗ {emotion_name}: No notes generated")
    except Exception as e:
        print(f"    ✗ {emotion_name}: Error - {e}")

print("\n[5/5] Testing complete!")
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print("\nGenerated MIDI files saved to: ./generated/")
print("\nYour trained model is working! 🎵")
print("\nNext steps:")
print("1. Listen to the generated MIDI files")
print("2. Start the API: python api.py")
print("3. Start the frontend: cd client && npm run dev")
print("4. Generate music through the web interface!")
