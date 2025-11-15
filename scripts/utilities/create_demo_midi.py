#!/usr/bin/env python3
"""
Create demo MIDI files for testing the audio playback system
This generates actual musical content for demonstration purposes
"""

import pretty_midi
from pathlib import Path
import random

def create_demo_midi(emotion: str, duration: float, output_path: str):
    """
    Create a demo MIDI file with actual musical content
    
    Args:
        emotion: Emotion name (joy, sadness, calm, etc.)
        duration: Duration in seconds
        output_path: Path to save MIDI file
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an instrument (Piano)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    
    # Define emotion-based musical parameters
    emotion_params = {
        'joy': {'tempo': 120, 'scale': [60, 62, 64, 65, 67, 69, 71, 72], 'velocity': 80},
        'sadness': {'tempo': 60, 'scale': [60, 62, 63, 65, 67, 68, 70, 72], 'velocity': 60},
        'calm': {'tempo': 80, 'scale': [60, 62, 64, 67, 69, 72], 'velocity': 70},
        'anger': {'tempo': 140, 'scale': [60, 61, 63, 65, 66, 68, 70, 72], 'velocity': 100},
        'fear': {'tempo': 100, 'scale': [60, 61, 63, 64, 66, 68, 69, 71], 'velocity': 75},
        'surprise': {'tempo': 110, 'scale': [60, 62, 64, 66, 68, 70, 72], 'velocity': 85}
    }
    
    params = emotion_params.get(emotion, emotion_params['calm'])
    tempo = params['tempo']
    scale = params['scale']
    velocity = params['velocity']
    
    # Calculate note duration based on tempo
    beat_duration = 60.0 / tempo
    
    # Generate notes
    current_time = 0.0
    note_durations = [beat_duration * 0.5, beat_duration, beat_duration * 1.5]
    
    while current_time < duration:
        pitch = random.choice(scale)
        note_duration = random.choice(note_durations)
        
        if current_time + note_duration > duration:
            note_duration = duration - current_time
        
        note = pretty_midi.Note(
            velocity=velocity + random.randint(-10, 10),
            pitch=pitch,
            start=current_time,
            end=current_time + note_duration
        )
        piano.notes.append(note)
        
        current_time += note_duration * random.uniform(0.8, 1.2)

    
    # Add chords for richer sound
    if emotion in ['joy', 'calm']:
        chord_times = [i * beat_duration * 4 for i in range(int(duration / (beat_duration * 4)))]
        for chord_time in chord_times:
            if chord_time < duration:
                root = random.choice([60, 65, 67])
                for offset in [0, 4, 7]:  # Major triad
                    note = pretty_midi.Note(
                        velocity=velocity - 20,
                        pitch=root + offset,
                        start=chord_time,
                        end=min(chord_time + beat_duration * 2, duration)
                    )
                    piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"✓ Created demo MIDI: {output_path}")
    print(f"  Duration: {duration:.2f}s, Notes: {len(piano.notes)}, Emotion: {emotion}")


def create_all_demo_files():
    """Create demo MIDI files for all emotions"""
    output_dir = Path("demo_midi")
    output_dir.mkdir(exist_ok=True)
    
    emotions = ['joy', 'sadness', 'calm', 'anger', 'fear', 'surprise']
    durations = [10, 20, 30]  # seconds
    
    print("Creating demo MIDI files...")
    print("=" * 60)
    
    for emotion in emotions:
        for duration in durations:
            filename = f"{emotion}_{duration}s.mid"
            output_path = output_dir / filename
            create_demo_midi(emotion, duration, str(output_path))
    
    print("=" * 60)
    print(f"✓ Created {len(emotions) * len(durations)} demo MIDI files in {output_dir}/")
    print("\nYou can use these files to test audio conversion:")
    print(f"  python3 -c \"from src.generation.audio_converter import convert_midi_to_audio; convert_midi_to_audio('demo_midi/joy_10s.mid', 'mp3')\"")


if __name__ == "__main__":
    create_all_demo_files()
