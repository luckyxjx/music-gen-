"""
Improved music generator with constraints and better sampling
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class ImprovedMusicGenerator:
    """Enhanced generator with constraints for better music quality"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # Token type tracking
        self.note_on_tokens = set()
        self.note_off_tokens = set()
        self.time_shift_tokens = set()
        self.velocity_tokens = set()
        
        for token_id, token_str in tokenizer.id_to_token.items():
            if token_str.startswith('NOTE_ON'):
                self.note_on_tokens.add(token_id)
            elif token_str.startswith('NOTE_OFF'):
                self.note_off_tokens.add(token_id)
            elif token_str.startswith('TIME_SHIFT'):
                self.time_shift_tokens.add(token_id)
            elif token_str.startswith('VELOCITY'):
                self.velocity_tokens.add(token_id)
    
    def generate_with_constraints(
        self,
        emotion: int = 0,
        duration_minutes: float = 2.0,
        temperature: float = 0.75,  # OPTIMAL: Lower for better structure
        top_k: int = 60,  # OPTIMAL: Higher for more variety
        top_p: float = 0.92,  # OPTIMAL: Nucleus sampling sweet spot
        max_tokens: int = 3072,  # OPTIMAL: Much longer for full songs
        min_notes: int = 100,  # OPTIMAL: Ensure substantial music
        max_consecutive_time_shifts: int = 3,  # OPTIMAL: Tighter constraint
        repetition_penalty: float = 1.3  # OPTIMAL: Stronger penalty
    ):
        """
        Generate music with constraints to ensure quality
        
        Args:
            emotion: Emotion index
            duration_minutes: Target duration
            temperature: Sampling temperature (lower = more conservative)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            max_tokens: Maximum tokens to generate
            min_notes: Minimum notes before allowing EOS
            max_consecutive_time_shifts: Max time shifts in a row
            repetition_penalty: Penalty for repeating tokens
        """
        # Initialize
        bos_id = self.tokenizer.vocab.get('<BOS>', 0)
        eos_id = self.tokenizer.vocab.get('<EOS>', 1)
        
        tokens = [bos_id]
        emotion_tensor = torch.tensor([emotion]).to(self.device)
        duration_tensor = torch.tensor([[duration_minutes]]).to(self.device)
        
        # Tracking
        note_count = 0
        consecutive_time_shifts = 0
        token_counts = {}  # For repetition penalty
        
        print(f"\nGenerating with improved constraints...")
        print(f"  Emotion: {emotion}")
        print(f"  Duration: {duration_minutes} min")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k: {top_k}, Top-p: {top_p}")
        print(f"  Min notes: {min_notes}")
        
        for i in range(max_tokens):
            # Prepare input
            input_tokens = torch.tensor([tokens]).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(input_tokens, emotion=emotion_tensor, duration=duration_tensor)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :].clone()
            
            # Apply repetition penalty
            for token_id, count in token_counts.items():
                if count > 0:
                    next_token_logits[token_id] /= (repetition_penalty ** count)
            
            # Apply constraints
            if consecutive_time_shifts >= max_consecutive_time_shifts:
                # Mask out time shift tokens
                for token_id in self.time_shift_tokens:
                    next_token_logits[token_id] = float('-inf')
            
            # Don't allow EOS until we have enough notes
            if note_count < min_notes:
                next_token_logits[eos_id] = float('-inf')
            
            # Temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for EOS
            if next_token == eos_id:
                print(f"  Reached EOS at {i+1} tokens with {note_count} notes")
                break
            
            # Update tracking
            if next_token in self.note_on_tokens:
                note_count += 1
                consecutive_time_shifts = 0
            elif next_token in self.time_shift_tokens:
                consecutive_time_shifts += 1
            else:
                consecutive_time_shifts = 0
            
            # Update token counts for repetition penalty
            token_counts[next_token] = token_counts.get(next_token, 0) + 1
            
            tokens.append(next_token)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1} tokens, {note_count} notes...")
        
        print(f"  Total: {len(tokens)} tokens, {note_count} notes")
        
        return tokens
    
    def save_midi(self, tokens, output_path: str):
        """Convert tokens to MIDI and save with stats"""
        print(f"\nConverting to MIDI...")
        
        # Analyze tokens
        token_types = {}
        for token_id in tokens:
            if token_id in self.tokenizer.id_to_token:
                token_str = self.tokenizer.id_to_token[token_id]
                token_type = token_str.split('_')[0]
                token_types[token_type] = token_types.get(token_type, 0) + 1
        
        print(f"Token breakdown:")
        for token_type, count in sorted(token_types.items()):
            print(f"  {token_type}: {count}")
        
        # Decode
        midi = self.tokenizer.decode(tokens)
        
        # Save
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))
        
        # Stats
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        
        print(f"\n✓ Saved to: {output_path}")
        print(f"MIDI Stats:")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"  Total notes: {total_notes}")
        print(f"  Instruments: {len(midi.instruments)}")
        print(f"  Notes per second: {total_notes/duration if duration > 0 else 0:.2f}")
        
        return midi
