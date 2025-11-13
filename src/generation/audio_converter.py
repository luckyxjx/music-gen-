"""
MIDI to audio conversion
"""

from pathlib import Path
from typing import Optional


class AudioConverter:
    """Convert MIDI to WAV/MP3"""
    
    def __init__(self, soundfont_path: Optional[str] = None):
        self.soundfont_path = soundfont_path
    
    def midi_to_wav(self, midi_path: str, output_path: str, sample_rate: int = 44100) -> bool:
        """
        Convert MIDI to WAV using FluidSynth
        
        Args:
            midi_path: Path to MIDI file
            output_path: Path to save WAV file
            sample_rate: Sample rate (default: 44100)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from midi2audio import FluidSynth
            
            # Use default soundfont if not specified
            if self.soundfont_path:
                fs = FluidSynth(sound_font=self.soundfont_path, sample_rate=sample_rate)
            else:
                fs = FluidSynth(sample_rate=sample_rate)
            
            fs.midi_to_audio(midi_path, output_path)
            return True
            
        except ImportError:
            print("Error: midi2audio not installed")
            print("Install with: pip install midi2audio")
            print("Also requires FluidSynth: brew install fluid-synth (macOS) or apt-get install fluidsynth (Linux)")
            return False
        except Exception as e:
            print(f"Error converting MIDI to WAV: {e}")
            return False
    
    def wav_to_mp3(self, wav_path: str, output_path: str, bitrate: str = "192k") -> bool:
        """
        Convert WAV to MP3
        
        Args:
            wav_path: Path to WAV file
            output_path: Path to save MP3 file
            bitrate: MP3 bitrate (default: 192k)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_wav(wav_path)
            audio.export(output_path, format="mp3", bitrate=bitrate)
            return True
            
        except ImportError:
            print("Error: pydub not installed")
            print("Install with: pip install pydub")
            print("Also requires ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            return False
        except Exception as e:
            print(f"Error converting WAV to MP3: {e}")
            return False
    
    def midi_to_mp3(self, midi_path: str, output_path: str, sample_rate: int = 44100, bitrate: str = "192k") -> bool:
        """
        Convert MIDI directly to MP3
        
        Args:
            midi_path: Path to MIDI file
            output_path: Path to save MP3 file
            sample_rate: Sample rate (default: 44100)
            bitrate: MP3 bitrate (default: 192k)
        
        Returns:
            True if successful, False otherwise
        """
        # Create temporary WAV file
        temp_wav = Path(output_path).with_suffix('.temp.wav')
        
        try:
            # Convert MIDI to WAV
            if not self.midi_to_wav(midi_path, str(temp_wav), sample_rate):
                return False
            
            # Convert WAV to MP3
            if not self.wav_to_mp3(str(temp_wav), output_path, bitrate):
                return False
            
            # Clean up temp file
            if temp_wav.exists():
                temp_wav.unlink()
            
            return True
            
        except Exception as e:
            print(f"Error converting MIDI to MP3: {e}")
            # Clean up temp file
            if temp_wav.exists():
                temp_wav.unlink()
            return False


def convert_midi_to_audio(midi_path: str, output_format: str = "wav", **kwargs) -> Optional[str]:
    """
    Convenience function to convert MIDI to audio
    
    Args:
        midi_path: Path to MIDI file
        output_format: Output format ('wav' or 'mp3')
        **kwargs: Additional arguments for conversion
    
    Returns:
        Path to output file if successful, None otherwise
    """
    midi_path = Path(midi_path)
    output_path = midi_path.with_suffix(f'.{output_format}')
    
    converter = AudioConverter()
    
    if output_format == "wav":
        success = converter.midi_to_wav(str(midi_path), str(output_path), **kwargs)
    elif output_format == "mp3":
        success = converter.midi_to_mp3(str(midi_path), str(output_path), **kwargs)
    else:
        print(f"Unsupported format: {output_format}")
        return None
    
    if success:
        return str(output_path)
    return None
