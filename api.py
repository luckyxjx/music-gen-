#!/usr/bin/env python3
"""
REST API for music generation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from pathlib import Path
import uuid
import os

from src.config import ModelConfig, TokenizerConfig, GenerationConfig
from src.model import create_model
from src.tokenizer import MIDITokenizer
from src.generation.text_parser import parse_text_input
from src.generation.audio_converter import AudioConverter
from generate_music import MusicGenerator
from src.training.human_feedback import HumanFeedbackCollector

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables
model = None
tokenizer = None
generator = None
device = None
audio_converter = None
feedback_collector = None

# Output directory
OUTPUT_DIR = Path("./generated_api")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize feedback collector
feedback_collector = HumanFeedbackCollector(feedback_dir="human_feedback")


def initialize_model():
    """Initialize model on startup"""
    global model, tokenizer, generator, device, audio_converter
    
    print("Initializing model...")
    
    # Create tokenizer
    tokenizer_config = TokenizerConfig()
    tokenizer = MIDITokenizer(tokenizer_config)
    
    # Create model
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
    
    # Create generator
    generator = MusicGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig(),
        device=device
    )
    
    # Create audio converter
    soundfont_path = "soundfont.sf2"
    audio_converter = AudioConverter(soundfont_path=soundfont_path)
    
    print(f"✓ Model initialized on {device}")
    print(f"✓ Audio converter initialized with SoundFont: {soundfont_path}")


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/api/generate', methods=['POST'])
def generate_music():
    """
    Generate music from text input
    
    Request body:
    {
        "text": "I'm happy, give me an upbeat 2-minute track",
        "temperature": 1.0,
        "top_k": 20,
        "use_demo": false
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 20)
        use_demo = data.get('use_demo', True)  # Default to demo mode
        
        if not text:
            return jsonify({'error': 'Text input required'}), 400
        
        # Parse text
        parsed = parse_text_input(text)
        
        # Generate unique ID
        generation_id = str(uuid.uuid4())
        
        midi_filename = f"{generation_id}.mid"
        midi_path = OUTPUT_DIR / midi_filename
        
        if use_demo:
            # Use demo generation with actual music
            from create_demo_midi import create_demo_midi
            duration_seconds = parsed['duration_minutes'] * 60
            create_demo_midi(parsed['emotion'], duration_seconds, str(midi_path))
            tokens_generated = 0  # Demo mode doesn't use tokens
        else:
            # Generate music using model
            tokens = generator.generate(
                emotion=parsed['emotion_index'],
                duration_minutes=parsed['duration_minutes'],
                temperature=temperature,
                top_k=top_k,
                max_tokens=512
            )
            
            # Save MIDI
            generator.save_midi(tokens, str(midi_path))
            tokens_generated = len(tokens)
        
        # Convert to MP3
        mp3_filename = f"{generation_id}.mp3"
        mp3_path = OUTPUT_DIR / mp3_filename
        audio_success = audio_converter.midi_to_mp3(str(midi_path), str(mp3_path))
        
        response_data = {
            'success': True,
            'generation_id': generation_id,
            'midi_file': f'/api/download/{generation_id}.mid',
            'emotion': parsed['emotion'],
            'duration': parsed['duration_minutes'],
            'tokens_generated': tokens_generated,
            'demo_mode': use_demo
        }
        
        # Add audio file if conversion succeeded
        if audio_success:
            response_data['audio_file'] = f'/api/download/{generation_id}.mp3'
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-emotion', methods=['POST'])
def generate_by_emotion():
    """
    Generate music by emotion and duration
    
    Request body:
    {
        "emotion": "joy",
        "duration": 2.0,
        "temperature": 1.0,
        "top_k": 20,
        "use_demo": false
    }
    """
    try:
        data = request.json
        emotion_name = data.get('emotion', 'calm')
        duration = data.get('duration', 2.0)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 20)
        use_demo = data.get('use_demo', True)  # Default to demo mode
        
        # Map emotion to index
        emotions_map = {
            'joy': 0, 'sadness': 1, 'anger': 2,
            'calm': 3, 'surprise': 4, 'fear': 5
        }
        emotion_idx = emotions_map.get(emotion_name.lower(), 3)
        
        # Generate unique ID
        generation_id = str(uuid.uuid4())
        
        midi_filename = f"{generation_id}.mid"
        midi_path = OUTPUT_DIR / midi_filename
        
        if use_demo:
            # Use demo generation with actual music
            from create_demo_midi import create_demo_midi
            duration_seconds = duration * 60
            create_demo_midi(emotion_name, duration_seconds, str(midi_path))
            tokens_generated = 0  # Demo mode doesn't use tokens
        else:
            # Generate music using model
            tokens = generator.generate(
                emotion=emotion_idx,
                duration_minutes=duration,
                temperature=temperature,
                top_k=top_k,
                max_tokens=512
            )
            
            # Save MIDI
            generator.save_midi(tokens, str(midi_path))
            tokens_generated = len(tokens)
        
        # Convert to MP3
        mp3_filename = f"{generation_id}.mp3"
        mp3_path = OUTPUT_DIR / mp3_filename
        audio_success = audio_converter.midi_to_mp3(str(midi_path), str(mp3_path))
        
        response_data = {
            'success': True,
            'generation_id': generation_id,
            'midi_file': f'/api/download/{generation_id}.mid',
            'emotion': emotion_name,
            'duration': duration,
            'tokens_generated': tokens_generated,
            'demo_mode': use_demo
        }
        
        # Add audio file if conversion succeeded
        if audio_success:
            response_data['audio_file'] = f'/api/download/{generation_id}.mp3'
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated MIDI or audio file"""
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Determine mimetype based on extension
        if filename.endswith('.mid'):
            mimetype = 'audio/midi'
        elif filename.endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.endswith('.wav'):
            mimetype = 'audio/wav'
        else:
            mimetype = 'application/octet-stream'
        
        return send_file(
            file_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get list of available emotions"""
    return jsonify({
        'emotions': [
            {'id': 0, 'name': 'joy', 'description': 'Happy, upbeat, energetic'},
            {'id': 1, 'name': 'sadness', 'description': 'Sad, melancholic, slow'},
            {'id': 2, 'name': 'anger', 'description': 'Intense, aggressive, fast'},
            {'id': 3, 'name': 'calm', 'description': 'Peaceful, relaxed, serene'},
            {'id': 4, 'name': 'surprise', 'description': 'Unexpected, varied'},
            {'id': 5, 'name': 'fear', 'description': 'Tense, anxious, uncertain'}
        ]
    })


@app.route('/api/feedback/samples', methods=['GET'])
def get_feedback_samples():
    """Get samples for human feedback"""
    try:
        # Get recent generations from localStorage (in practice, from database)
        # For now, return samples from generated_api directory
        samples = []
        
        if OUTPUT_DIR.exists():
            midi_files = list(OUTPUT_DIR.glob("*.mid"))
            
            for midi_file in midi_files[:20]:  # Limit to 20 samples
                generation_id = midi_file.stem
                
                # Check if audio file exists
                audio_file = OUTPUT_DIR / f"{generation_id}.mp3"
                
                # Try to extract metadata from filename or use defaults
                sample = {
                    'id': generation_id,
                    'emotion': 'unknown',  # Would be stored in metadata
                    'duration': 2.0,
                    'midi_file': f'/api/download/{midi_file.name}',
                    'audio_file': f'/api/download/{audio_file.name}' if audio_file.exists() else None,
                    'timestamp': midi_file.stat().st_mtime
                }
                
                # Check if feedback already exists
                existing_feedback = feedback_collector.get_feedback_for_sample(generation_id)
                if not existing_feedback:  # Only include samples without feedback
                    samples.append(sample)
        
        return jsonify({
            'success': True,
            'samples': samples[:10]  # Return max 10 samples
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/feedback/submit', methods=['POST'])
def submit_feedback():
    """Submit human feedback for a sample"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['generation_id', 'emotion', 'emotion_accuracy', 
                          'musical_quality', 'overall_rating']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Add feedback
        feedback_entry = feedback_collector.add_feedback(
            generation_id=data['generation_id'],
            emotion=data['emotion'],
            emotion_accuracy=data['emotion_accuracy'],
            musical_quality=data['musical_quality'],
            overall_rating=data['overall_rating'],
            comments=data.get('comments', '')
        )
        
        return jsonify({
            'success': True,
            'feedback': feedback_entry,
            'message': 'Feedback submitted successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    try:
        stats = feedback_collector.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/feedback/export', methods=['GET'])
def export_feedback():
    """Export feedback for RL training"""
    try:
        training_data = feedback_collector.export_for_training()
        
        return jsonify({
            'success': True,
            'data': training_data,
            'count': len(training_data['generation_ids'])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Start server
    print("\n" + "="*60)
    print("MUSIC GENERATION API SERVER")
    print("="*60)
    print("\nAPI Endpoints:")
    print("  POST /api/generate - Generate from text")
    print("  POST /api/generate-emotion - Generate by emotion")
    print("  GET  /api/download/<filename> - Download MIDI")
    print("  GET  /api/emotions - List emotions")
    print("  GET  /api/health - Health check")
    print("  GET  /api/feedback/samples - Get samples for feedback")
    print("  POST /api/feedback/submit - Submit human feedback")
    print("  GET  /api/feedback/stats - Get feedback statistics")
    print("  GET  /api/feedback/export - Export feedback for training")
    print("\nServer starting on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)