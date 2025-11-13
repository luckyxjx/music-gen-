# Full Stack Setup Guide - EMOPIA Music Generator

This guide will help you connect and run the complete EMOPIA music generation system with both backend API and frontend interface.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐
│   Frontend      │         │   Backend API    │
│   (React +      │ ◄─────► │   (Flask)        │
│    Vite)        │  HTTP   │                  │
│   Port: 5173    │         │   Port: 5000     │
└─────────────────┘         └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Music Generator │
                            │  (PyTorch Model) │
                            └──────────────────┘
```

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **PyTorch** (CPU or GPU version)
- **CUDA** (optional, for GPU acceleration)

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make the script executable (first time only)
chmod +x start_app.sh

# Start both backend and frontend
./start_app.sh
```

This will:
1. Start the backend API on `http://localhost:5000`
2. Wait for model initialization (15 seconds)
3. Start the frontend on `http://localhost:5173`

### Option 2: Manual Setup

#### Step 1: Start Backend API

```bash
# Install Python dependencies (if not already installed)
pip install -r requirements.txt

# Start the API server
python3 api.py
```

Wait for the message: `✓ Model initialized on [device]`

The backend will be available at `http://localhost:5000`

#### Step 2: Start Frontend

In a new terminal:

```bash
# Navigate to client folder
cd client

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

The backend provides the following REST API endpoints:

### 1. Health Check
```
GET /api/health
```
Returns server status and model information.

### 2. Generate Music from Text
```
POST /api/generate
Content-Type: application/json

{
  "text": "I'm happy, give me an upbeat 2-minute track",
  "temperature": 1.0,
  "top_k": 20
}
```

### 3. Generate Music by Emotion
```
POST /api/generate-emotion
Content-Type: application/json

{
  "emotion": "joy",
  "duration": 2.0,
  "temperature": 1.0,
  "top_k": 20
}
```

### 4. Download MIDI File
```
GET /api/download/<generation_id>.mid
```

### 5. List Available Emotions
```
GET /api/emotions
```

Returns: `joy`, `sadness`, `anger`, `calm`, `surprise`, `fear`

## Frontend Features

### Landing Page
- Animated 3D sphere visualization
- Navigation menu
- "Get Started" button to enter chat interface

### Chat Interface
- Text input for music generation requests
- Real-time loading states
- Result display with:
  - Detected emotion
  - Duration
  - Token count
  - Download button for MIDI file

### Example Prompts

Try these in the chat interface:

- "I'm happy, give me an upbeat 2-minute track"
- "Create a sad melody for 1 minute"
- "Generate an angry and intense 3-minute piece"
- "I need calm music for meditation, 5 minutes"
- "Surprise me with something unexpected for 2 minutes"
- "Create fearful and tense music for 1.5 minutes"

## Configuration

### Backend Configuration

Edit `api.py` to customize:

```python
# Model configuration
model_config = ModelConfig(
    model_type="transformer",
    d_model=256,        # Model dimension
    n_layers=4,         # Number of layers
    n_heads=4,          # Attention heads
    use_emotion_conditioning=True,
    use_duration_control=True
)

# Server configuration
app.run(host='0.0.0.0', port=5000, debug=False)
```

### Frontend Configuration

Edit `client/src/App.tsx` to change API URL:

```typescript
const API_BASE_URL = 'http://localhost:5000'
```

For production, update to your deployed backend URL.

## CORS Configuration

The backend has CORS enabled for all origins:

```python
from flask_cors import CORS
CORS(app)
```

For production, restrict to specific origins:

```python
CORS(app, origins=['https://yourdomain.com'])
```

## File Structure

```
.
├── api.py                      # Backend API server
├── client/                     # Frontend application
│   ├── src/
│   │   ├── App.tsx            # Main React component
│   │   ├── App.css            # Styles
│   │   └── main.tsx           # Entry point
│   ├── package.json           # Node dependencies
│   └── vite.config.ts         # Vite configuration
├── src/                       # Backend source code
│   ├── model.py               # PyTorch model
│   ├── tokenizer.py           # MIDI tokenizer
│   ├── generation/            # Generation utilities
│   └── ...
├── generated_api/             # Generated MIDI files
└── start_app.sh              # Startup script
```

## Troubleshooting

### Backend Issues

**Problem**: Model fails to load
```bash
# Check if PyTorch is installed
python3 -c "import torch; print(torch.__version__)"

# Install PyTorch if missing
pip install torch
```

**Problem**: Port 5000 already in use
```bash
# Find and kill the process
lsof -ti:5000 | xargs kill -9

# Or change the port in api.py
app.run(host='0.0.0.0', port=5001, debug=False)
```

### Frontend Issues

**Problem**: Cannot connect to backend
- Ensure backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Verify API_BASE_URL in `App.tsx`

**Problem**: Port 5173 already in use
```bash
# Kill the process
lsof -ti:5173 | xargs kill -9

# Or Vite will automatically use next available port
```

### Generation Issues

**Problem**: Music generation is slow
- This is normal on CPU (15-30 seconds)
- For faster generation, use GPU with CUDA
- Reduce `max_tokens` in generation config

**Problem**: Generated music sounds random
- Ensure model is properly trained
- Check if checkpoint file exists
- Adjust `temperature` (lower = more predictable)

## Development

### Backend Development

```bash
# Run with debug mode
python3 api.py
# Edit api.py and restart server
```

### Frontend Development

```bash
cd client
npm run dev
# Changes auto-reload with hot module replacement
```

### Building for Production

```bash
# Build frontend
cd client
npm run build

# Serve with a static server
npm install -g serve
serve -s dist -p 5173
```

## Testing the Integration

1. **Start both servers** using `./start_app.sh`

2. **Open browser** to `http://localhost:5173`

3. **Click "Get Started"** to enter chat interface

4. **Enter a prompt**: "I'm happy, give me an upbeat 2-minute track"

5. **Wait for generation** (15-30 seconds on CPU)

6. **Download MIDI** file and play in your DAW or MIDI player

## Performance Tips

- **Use GPU**: 10-20x faster than CPU
- **Reduce duration**: Shorter tracks generate faster
- **Lower temperature**: More predictable, slightly faster
- **Batch requests**: Queue multiple generations

## Next Steps

- Train the model on your dataset (see `TRAINING_GUIDE.md`)
- Customize emotions and their mappings
- Add audio playback in the browser
- Implement user authentication
- Deploy to production server

## Support

For issues or questions:
- Check existing documentation in `docs/`
- Review API integration guide: `API_INTEGRATION.md`
- Check training guide: `TRAINING_GUIDE.md`

---

**Happy Music Generation! 🎵**
