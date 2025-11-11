"""
MIDI tokenizer with REMI-like representation
"""

import pretty_midi
import numpy as np
from typing import List, Dict, Optional
from src.config import TokenizerConfig


class MIDITokenizer:
    """REMI-like MIDI tokenizer"""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build token vocabulary"""
        vocab = {}
        idx = 0
        
        # Special tokens
        vocab['<PAD>'] = idx; idx += 1
        vocab['<BOS>'] = idx; idx += 1
        vocab['<EOS>'] = idx; idx += 1
        vocab['<UNK>'] = idx; idx += 1
        
        # Instrument tokens (if multi-instrument)
        if self.config.instruments:
            for inst in self.config.instruments:
                vocab[f'INST_{inst}'] = idx; idx += 1
        
        # Note tokens (0-127)
        for pitch in range(128):
            vocab[f'NOTE_ON_{pitch}'] = idx; idx += 1
            vocab[f'NOTE_OFF_{pitch}'] = idx; idx += 1
        
        # Time shift tokens
        for shift in range(self.config.max_shift_steps):
            vocab[f'TIME_SHIFT_{shift}'] = idx; idx += 1
        
        # Velocity tokens
        for i, vel in enumerate(self.config.velocity_buckets[:-1]):
            next_vel = self.config.velocity_buckets[i + 1]
            vocab[f'VELOCITY_{vel}_{next_vel}'] = idx; idx += 1
        
        # Bar marker
        vocab['BAR'] = idx; idx += 1
        
        return vocab
    
    def encode(self, midi: pretty_midi.PrettyMIDI) -> List[int]:
        """Encode MIDI to token sequence"""
        tokens = [self.vocab['<BOS>']]
        
        # Collect all events
        events = []
        for inst_idx, instrument in enumerate(midi.instruments):
            if instrument.is_drum:
                continue
            
            # Determine instrument type
            inst_name = self._get_instrument_name(inst_idx, len(midi.instruments))
            
            for note in instrument.notes:
                events.append({
                    'time': note.start,
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'instrument': inst_name
                })
                events.append({
                    'time': note.end,
                    'type': 'note_off',
                    'pitch': note.pitch,
                    'instrument': inst_name
                })
        
        # Sort by time
        events.sort(key=lambda x: x['time'])
        
        # Convert to tokens
        current_time = 0
        current_inst = None
        
        for event in events:
            # Time shift
            time_diff = event['time'] - current_time
            if time_diff > 0:
                shifts = int(time_diff / 0.0625)  # 16th note = 0.0625 seconds at 120 BPM
                shifts = min(shifts, self.config.max_shift_steps - 1)
                if shifts > 0:
                    tokens.append(self.vocab[f'TIME_SHIFT_{shifts}'])
                    current_time += shifts * 0.0625
            
            # Instrument change
            if self.config.instruments and event['instrument'] != current_inst:
                inst_token = f"INST_{event['instrument']}"
                if inst_token in self.vocab:
                    tokens.append(self.vocab[inst_token])
                current_inst = event['instrument']
            
            # Note event
            if event['type'] == 'note_on':
                # Velocity
                vel_bucket = self._get_velocity_bucket(event['velocity'])
                if vel_bucket in self.vocab:
                    tokens.append(self.vocab[vel_bucket])
                
                # Note on
                tokens.append(self.vocab[f"NOTE_ON_{event['pitch']}"])
            else:
                # Note off
                tokens.append(self.vocab[f"NOTE_OFF_{event['pitch']}"])
            
            # Limit sequence length
            if len(tokens) >= self.config.max_events:
                break
        
        tokens.append(self.vocab['<EOS>'])
        return tokens
    
    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """Decode token sequence to MIDI"""
        midi = pretty_midi.PrettyMIDI()
        
        # Create instruments
        instruments = {}
        for inst_name in self.config.instruments:
            program = self.config.instrument_programs.get(inst_name, 0)
            instruments[inst_name] = pretty_midi.Instrument(program=program, name=inst_name)
        
        # Default instrument if no multi-instrument
        if not instruments:
            instruments['default'] = pretty_midi.Instrument(program=0)
        
        current_time = 0.0
        current_velocity = 64
        current_inst = list(instruments.keys())[0]
        active_notes = {}  # (inst, pitch) -> start_time
        
        for token_id in tokens:
            if token_id not in self.id_to_token:
                continue
            
            token = self.id_to_token[token_id]
            
            if token.startswith('TIME_SHIFT_'):
                shift = int(token.split('_')[-1])
                current_time += shift * 0.0625
            
            elif token.startswith('INST_'):
                inst_name = token.replace('INST_', '')
                if inst_name in instruments:
                    current_inst = inst_name
            
            elif token.startswith('VELOCITY_'):
                parts = token.split('_')
                current_velocity = int(parts[1])
            
            elif token.startswith('NOTE_ON_'):
                pitch = int(token.split('_')[-1])
                active_notes[(current_inst, pitch)] = (current_time, current_velocity)
            
            elif token.startswith('NOTE_OFF_'):
                pitch = int(token.split('_')[-1])
                key = (current_inst, pitch)
                if key in active_notes:
                    start_time, velocity = active_notes[key]
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    instruments[current_inst].notes.append(note)
                    del active_notes[key]
        
        # Add instruments to MIDI
        for inst in instruments.values():
            if len(inst.notes) > 0:
                midi.instruments.append(inst)
        
        return midi
    
    def _get_instrument_name(self, inst_idx: int, total_insts: int) -> str:
        """Determine instrument name from index"""
        if not self.config.instruments:
            return 'default'
        
        if total_insts == 1:
            return 'melody'
        elif inst_idx == 0:
            return 'melody'
        elif inst_idx == 1:
            return 'harmony'
        else:
            return 'drums'
    
    def _get_velocity_bucket(self, velocity: int) -> str:
        """Get velocity bucket token"""
        for i, vel in enumerate(self.config.velocity_buckets[:-1]):
            next_vel = self.config.velocity_buckets[i + 1]
            if vel <= velocity < next_vel:
                return f'VELOCITY_{vel}_{next_vel}'
        return f'VELOCITY_{self.config.velocity_buckets[-2]}_{self.config.velocity_buckets[-1]}'
