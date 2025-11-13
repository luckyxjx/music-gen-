#!/usr/bin/env python3
"""
Generate music from a trained model (or use pretrained weights)
"""

import torch
import numpy as np
from pathlib import Path

from src.config import ModelConfig, TokenizerConfig, GenerationConfig
from src.model import create_model
from src.tokenizer import MIDITokenizer
from src.generation.text_parser import parse_text_input


class MusicGenerator:
    """Generate music from model"""
    
    def __init__(self, model, tokenizer, config: GenerationConfig, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
    
    def generate(
        self,
        emotion: int = 0,
        duration_minutes: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 10,
        max_tokens: int = 512
    ):
        """
        Generate music
        
        Args:
            emotion: Emotion index (0=joy, 1=sadness, 2=anger, 3=calm, 4=surprise, 5=fear)
            duration_minutes: Target duration in minutes
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
        """
        # Start with BOS token
        tokens = [self.tokenizer.vocab['<BOS>']]
        
        # Prepare emotion and duration
        emotion_tensor = torch.tensor([emotion]).to(self.device)
        duration_tensor = torch.tensor([[duration_minutes]]).to(self.device)
        
        print(f"\nGenerating music...")
        print(f"  Emotion: {emotion}")
        print(f"  Duration: {duration_minutes} minutes")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k: {top_k}")
        
        # Generate tokens
        for i in range(max_tokens):
            # Prepare input
            input_tokens = torch.tensor([tokens]).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(input_tokens, emotion=emotion_tensor, duration=duration_tensor)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for EOS
            if next_token == self.tokenizer.vocab['<EOS>']:
                break
            
            tokens.append(next_token)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1} tokens...")
        
        print(f"  Total tokens generated: {len(tokens)}")
        
        return tokens
    
    def save_midi(self, tokens, output_path: str):
        """Convert tokens to MIDI and save"""
        print(f"\nConverting to MIDI...")
        midi = self.tokenizer.decode(tokens)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))
        
        print(f"✓ Saved to: {output_path}")
        
        # Print stats
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        
        print(f"\nMIDI Stats:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total notes: {total_notes}")
        print(f"  Instruments: {len(midi.instruments)}")
        
        return midi


def generate_with_untrained_model():
    """Generate music with an untrained model (random initialization)"""
    print("\n" + "="*60)
    print("MUSIC GENERATION DEMO")
    print("(Using untrained model - random music)")
    print("="*60)
    
    # Create tokenizer
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    
    print(f"\n✓ Tokenizer created (vocab size: {tokenizer.vocab_size})")
    
    # Create model
    model_config = ModelConfig(
        model_type="transformer",
        d_model=256,  # Smaller for faster generation
        n_layers=4,
        n_heads=4,
        use_emotion_conditioning=True,
        use_duration_control=True
    )
    
    model = create_model(model_config, tokenizer.vocab_size)
    
    print(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Create generator
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    generator = MusicGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig(),
        device=device
    )
    
    # Generate for each emotion
    emotions = {
        0: "joy",
        1: "sadness",
        2: "anger",
        3: "calm",
        4: "surprise",
        5: "fear"
    }
    
    output_dir = Path("./generated_music")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n" + "="*60)
    print("GENERATING MUSIC FOR EACH EMOTION")
    print("="*60)
    
    for emotion_idx, emotion_name in emotions.items():
        print(f"\n{'='*60}")
        print(f"Emotion: {emotion_name.upper()}")
        print(f"{'='*60}")
        
        # Generate
        tokens = generator.generate(
            emotion=emotion_idx,
            duration_minutes=0.5,  # 30 seconds
            temperature=1.0,
            top_k=20,
            max_tokens=256
        )
        
        # Save
        output_path = output_dir / f"{emotion_name}_sample.mid"
        generator.save_midi(tokens, output_path)
    
    print(f"\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✓ Generated 6 MIDI files in: {output_dir}")
    print("\nNote: These are from an UNTRAINED model (random weights)")
    print("The music will be random/chaotic until the model is trained.")
    print("\nTo get good music:")
    print("1. Train the model: python train.py")
    print("2. Generate with trained weights")
    
    print("\n" + "="*60)
    print("HOW TO PLAY THE MIDI FILES")
    print("="*60)
    print("\nOption 1: Use a MIDI player")
    print("  - macOS: GarageBand, Logic Pro")
    print("  - Windows: Windows Media Player, VLC")
    print("  - Linux: TiMidity++, VLC")
    
    print("\nOption 2: Convert to audio")
    print("  pip install midi2audio")
    print("  python -c \"from midi2audio import FluidSynth; FluidSynth().midi_to_audio('generated_music/joy_sample.mid', 'joy.wav')\"")
    
    print("\nOption 3: Online MIDI player")
    print("  Upload to: https://onlinesequencer.net/import")
    
    return output_dir


def generate_with_emotion_transition():
    """Generate music with emotion transition"""
    print("\n" + "="*60)
    print("EMOTION TRANSITION GENERATION")
    print("="*60)
    
    # Create tokenizer and model
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    
    model_config = ModelConfig(
        model_type="transformer",
        d_model=256,
        n_layers=4,
        n_heads=4,
        use_emotion_conditioning=True,
        use_duration_control=True
    )
    
    model = create_model(model_config, tokenizer.vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    generator = MusicGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig(),
        device=device
    )
    
    print("\n✓ Model ready for emotion transitions")
    print(f"✓ Device: {device}")
    
    # Emotion transitions
    transitions = [
        ("joy", "calm", "Happy to Calm"),
        ("sadness", "joy", "Sad to Happy"),
        ("anger", "calm", "Angry to Calm")
    ]
    
    output_dir = Path("./generated_transitions")
    output_dir.mkdir(exist_ok=True)
    
    emotions_map = {"joy": 0, "sadness": 1, "anger": 2, "calm": 3, "surprise": 4, "fear": 5}
    
    for start_emotion, end_emotion, description in transitions:
        print(f"\n{'='*60}")
        print(f"Transition: {description}")
        print(f"{'='*60}")
        
        start_idx = emotions_map[start_emotion]
        end_idx = emotions_map[end_emotion]
        
        # Generate with interpolation (simulate by generating segments)
        all_tokens = [tokenizer.vocab['<BOS>']]
        
        # Generate 3 segments with interpolated emotions
        for i, alpha in enumerate([0.0, 0.5, 1.0]):
            print(f"\n  Segment {i+1}/3 (alpha={alpha:.1f})...")
            
            # Simple interpolation: alternate between emotions
            if alpha < 0.33:
                emotion = start_idx
            elif alpha < 0.67:
                emotion = (start_idx + end_idx) // 2 if start_idx != end_idx else start_idx
            else:
                emotion = end_idx
            
            tokens = generator.generate(
                emotion=emotion,
                duration_minutes=0.5,
                temperature=1.0,
                top_k=20,
                max_tokens=100
            )
            
            # Append tokens (skip BOS after first segment)
            if i > 0 and tokens[0] == tokenizer.vocab['<BOS>']:
                tokens = tokens[1:]
            all_tokens.extend(tokens)
        
        # Save
        output_path = output_dir / f"transition_{start_emotion}_to_{end_emotion}.mid"
        generator.save_midi(all_tokens, output_path)
    
    print(f"\n" + "="*60)
    print("EMOTION TRANSITION GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✓ Generated {len(transitions)} transition files in: {output_dir}")
    
    return output_dir


def generate_from_text():
    """Generate music from text input"""
    print("\n" + "="*60)
    print("TEXT-TO-MUSIC GENERATION")
    print("="*60)
    
    # Create tokenizer and model
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    
    model_config = ModelConfig(
        model_type="transformer",
        d_model=256,
        n_layers=4,
        n_heads=4,
        use_emotion_conditioning=True,
        use_duration_control=True
    )
    
    model = create_model(model_config, tokenizer.vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    generator = MusicGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig(),
        device=device
    )
    
    print("\n✓ Model ready")
    print(f"✓ Device: {device}")
    
    # Example text inputs
    examples = [
        "I'm happy, give me an upbeat 2-minute track",
        "I feel sad, make me a calm piano piece for 1 minute",
        "I'm angry, create an intense 30 second track",
        "Give me a peaceful and relaxing 3-minute song"
    ]
    
    print("\n" + "="*60)
    print("EXAMPLE TEXT INPUTS")
    print("="*60)
    
    output_dir = Path("./generated_from_text")
    output_dir.mkdir(exist_ok=True)
    
    for i, text in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: \"{text}\"")
        print(f"{'='*60}")
        
        # Parse text
        parsed = parse_text_input(text)
        print(f"\nParsed:")
        print(f"  Emotion: {parsed['emotion']}")
        print(f"  Duration: {parsed['duration_minutes']} minutes")
        
        # Generate
        tokens = generator.generate(
            emotion=parsed['emotion_index'],
            duration_minutes=parsed['duration_minutes'],
            temperature=1.0,
            top_k=20,
            max_tokens=256
        )
        
        # Save
        output_path = output_dir / f"example_{i}_{parsed['emotion']}.mid"
        generator.save_midi(tokens, output_path)
    
    print(f"\n" + "="*60)
    print("TEXT-TO-MUSIC GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✓ Generated {len(examples)} MIDI files in: {output_dir}")
    
    return output_dir


def main():
    """Main function"""
    print("\n" + "="*60)
    print("MUSIC GENERATION OPTIONS")
    print("="*60)
    print("\n1. Generate from text input")
    print("2. Generate for all emotions")
    print("3. Generate with emotion transitions ")
    print("4. Exit")
    
    choice = input("\nChoose option [1-4]: ").strip()
    
    try:
        if choice == "1":
            output_dir = generate_from_text()
        elif choice == "2":
            output_dir = generate_with_untrained_model()
        elif choice == "3":
            output_dir = generate_with_emotion_transition()
        else:
            print("\nExiting...")
            return
        
        print(f"\n✓ Success! Check the '{output_dir}' folder for MIDI files.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
