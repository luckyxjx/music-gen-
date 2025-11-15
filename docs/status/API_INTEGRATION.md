# API Integration Guide

## Quick Start

### 1. Install API Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server
```bash
python api.py
```

Server will start on `http://localhost:5000`

---

## API Endpoints

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "mps"
}
```

---

### Generate from Text
```http
POST /api/generate
Content-Type: application/json

{
  "text": "I'm happy, give me an upbeat 2-minute track",
  "temperature": 1.0,
  "top_k": 20
}
```

**Response:**
```json
{
  "success": true,
  "generation_id": "abc123-def456",
  "midi_file": "/api/download/abc123-def456.mid",
  "emotion": "joy",
  "duration": 2.0,
  "tokens_generated": 256
}
```

---

### Generate by Emotion
```http
POST /api/generate-emotion
Content-Type: application/json

{
  "emotion": "joy",
  "duration": 2.0,
  "temperature": 1.0,
  "top_k": 20
}
```

**Response:**
```json
{
  "success": true,
  "generation_id": "abc123-def456",
  "midi_file": "/api/download/abc123-def456.mid",
  "emotion": "joy",
  "duration": 2.0,
  "tokens_generated": 256
}
```

---

### Download MIDI File
```http
GET /api/download/{filename}
```

Returns the MIDI file for download.

---

### List Emotions
```http
GET /api/emotions
```

**Response:**
```json
{
  "emotions": [
    {"id": 0, "name": "joy", "description": "Happy, upbeat, energetic"},
    {"id": 1, "name": "sadness", "description": "Sad, melancholic, slow"},
    {"id": 2, "name": "anger", "description": "Intense, aggressive, fast"},
    {"id": 3, "name": "calm", "description": "Peaceful, relaxed, serene"},
    {"id": 4, "name": "surprise", "description": "Unexpected, varied"},
    {"id": 5, "name": "fear", "description": "Tense, anxious, uncertain"}
  ]
}
```

---

## Frontend Integration

### JavaScript/React Example

```javascript
// Generate music from text
async function generateMusic(text) {
  const response = await fetch('http://localhost:5000/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      temperature: 1.0,
      top_k: 20
    })
  });
  
  const data = await response.json();
  
  if (data.success) {
    // Download the MIDI file
    window.location.href = `http://localhost:5000${data.midi_file}`;
  }
}

// Generate by emotion
async function generateByEmotion(emotion, duration) {
  const response = await fetch('http://localhost:5000/api/generate-emotion', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      emotion: emotion,
      duration: duration,
      temperature: 1.0,
      top_k: 20
    })
  });
  
  const data = await response.json();
  return data;
}

// Get available emotions
async function getEmotions() {
  const response = await fetch('http://localhost:5000/api/emotions');
  const data = await response.json();
  return data.emotions;
}
```

---

## CORS Configuration

The API has CORS enabled by default, allowing requests from any origin. For production, update `api.py`:

```python
CORS(app, origins=['http://your-frontend-domain.com'])
```

---

## Environment Variables

Create a `.env` file for configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=False

# Model Configuration
MODEL_CHECKPOINT=./checkpoints/best_model.pt
DEVICE=auto  # auto, cuda, mps, cpu

# Generation Configuration
DEFAULT_TEMPERATURE=1.0
DEFAULT_TOP_K=20
MAX_DURATION=5.0
```

---

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

### Using Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements_api.txt

EXPOSE 5000

CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t music-gen-api .
docker run -p 5000:5000 music-gen-api
```

---

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:5000/api/health

# Generate from text
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "I am happy, make me upbeat music"}'

# Generate by emotion
curl -X POST http://localhost:5000/api/generate-emotion \
  -H "Content-Type: application/json" \
  -d '{"emotion": "joy", "duration": 2.0}'

# List emotions
curl http://localhost:5000/api/emotions
```

### Using Python
```python
import requests

# Generate music
response = requests.post('http://localhost:5000/api/generate', json={
    'text': 'I am happy, make me upbeat music',
    'temperature': 1.0,
    'top_k': 20
})

data = response.json()
print(f"Generated: {data['midi_file']}")

# Download MIDI
midi_url = f"http://localhost:5000{data['midi_file']}"
midi_response = requests.get(midi_url)

with open('generated.mid', 'wb') as f:
    f.write(midi_response.content)
```

---

## Frontend Integration Checklist

- [ ] Install API dependencies
- [ ] Start API server
- [ ] Test health endpoint
- [ ] Test generation endpoints
- [ ] Update frontend API URL
- [ ] Handle MIDI file downloads
- [ ] Add error handling
- [ ] Test CORS configuration
- [ ] Add loading states
- [ ] Display generation results

---

## Troubleshooting

### CORS Errors
- Ensure `flask-cors` is installed
- Check browser console for specific CORS errors
- Verify API URL in frontend matches server

### Model Loading Issues
- Ensure model files are in correct location
- Check available memory (model needs ~500MB)
- Verify PyTorch installation

### Generation Timeout
- Increase request timeout in frontend
- Reduce `max_tokens` parameter
- Use smaller model for faster generation

---

## Next Steps

1. Add your frontend folder to the repo
2. Update frontend API endpoints to point to `http://localhost:5000`
3. Test the integration
4. Deploy both frontend and backend

The API is ready to connect with your frontend!
