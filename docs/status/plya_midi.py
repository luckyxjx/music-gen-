"""
play_midi.py

Utility to convert MIDI to WAV and play it.
"""

from midi2audio import FluidSynth
import pygame
from pathlib import Path
import os

SOUNDFRONT = Path(__file__).parent / "FluidR3_GM" / "FluidR3_GM.sf2"

def play_midi(midi_file, soundfont=SOUNDFRONT):
    """
    Convert MIDI -> WAV -> play using pygame.
    - midi_file: path to .mid file
    - soundfont: optional .sf2 soundfont path (default system soundfont)
    """
    if not os.path.exists(midi_file):
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")

    wav_file = midi_file.replace(".mid", ".wav")

    # Convert with FluidSynth
    fs = FluidSynth(sound_font=soundfont) if soundfont else FluidSynth()
    fs.midi_to_audio(midi_file, wav_file)

    # Play with pygame
    pygame.mixer.init()
    pygame.mixer.music.load(wav_file)
    pygame.mixer.music.play()
    print(f"▶️ Playing {midi_file}...")

    # Keep script alive while playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    print("Finished playback.")


if __name__ == "__main__":
    # Example usage
    test_file = "outputs/generated_uncond_3.mid"
    if os.path.exists(test_file):
        play_midi(test_file)
    else:
        print("No generated MIDI found. Run mvp_demo.py first.")
