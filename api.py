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
from generate_music import MusicGenerator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables
model = None
tokenizer = None
generator = None
device = None

# Output directory
OUTPUT_DIR = Path("./generated_api")
OUTPUT_DIR.mkdir(exist_ok=True)


def initialize_model():
    """Initialize model on startup"""
    global model, tokenizer, generator, device
    
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
    
    print(f"✓ Model initialized on {device}")


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
        "top_k": 20
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 20)
        
        if not text:
            return jsonify({'error': 'Text input required'}), 400
        
        # Parse text
        parsed = parse_text_input(text)
        
        # Generate unique ID
        generation_id = str(uuid.uuid4())
        
        # Generate music
        tokens = generator.generate(
            emotion=parsed['emotion_index'],
            duration_minutes=parsed['duration_minutes'],
            temperature=temperature,
            top_k=top_k,
            max_tokens=512
        )
        
        # Save MIDI
        midi_filename = f"{generation_id}.mid"
        midi_path = OUTPUT_DIR / midi_filename
        generator.save_midi(tokens, str(midi_path))
        
        return jsonify({
            'success': True,
            'generation_id': generation_id,
            'midi_file': f'/api/download/{generation_id}.mid',
            'emotion': parsed['emotion'],
            'duration': parsed['duration_minutes'],
            'tokens_generated': len(tokens)
        })
    
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
        "top_k": 20
    }
    """
    try:
        data = request.json
        emotion_name = data.get('emotion', 'calm')
        duration = data.get('duration', 2.0)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 20)
        
        # Map emotion to index
        emotions_map = {
            'joy': 0, 'sadness': 1, 'anger': 2,
            'calm': 3, 'surprise': 4, 'fear': 5
        }
        emotion_idx = emotions_map.get(emotion_name.lower(), 3)
        
        # Generate unique ID
        generation_id = str(uuid.uuid4())
        
        # Generate music
        tokens = generator.generate(
            emotion=emotion_idx,
            duration_minutes=duration,
            temperature=temperature,
            top_k=top_k,
            max_tokens=512
        )
        
        # Save MIDI
        midi_filename = f"{generation_id}.mid"
        midi_path = OUTPUT_DIR / midi_filename
        generator.save_midi(tokens, str(midi_path))
        
        return jsonify({
            'success': True,
            'generation_id': generation_id,
            'midi_file': f'/api/download/{generation_id}.mid',
            'emotion': emotion_name,
            'duration': duration,
            'tokens_generated': len(tokens)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated MIDI file"""
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            mimetype='audio/midi',
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
    print("\nServer starting on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
