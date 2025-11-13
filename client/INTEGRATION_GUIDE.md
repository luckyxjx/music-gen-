# Frontend Integration Guide

This document explains how the React frontend connects to the Flask backend API.

## Architecture

```
┌─────────────────────────────────────────┐
│         React Frontend (Vite)           │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │         App Component              │ │
│  │                                    │ │
│  │  • Landing Page (3D Sphere)       │ │
│  │  • Chat Interface                 │ │
│  │  • API Integration                │ │
│  └───────────────────────────────────┘ │
│                 │                       │
│                 │ HTTP Requests         │
│                 ▼                       │
└─────────────────────────────────────────┘
                  │
                  │ fetch()
                  │
┌─────────────────▼─────────────────────┐
│         Flask Backend API              │
│                                        │
│  • POST /api/generate                 │
│  • POST /api/generate-emotion         │
│  • GET  /api/download/<file>          │
│  • GET  /api/emotions                 │
│  • GET  /api/health                   │
└────────────────────────────────────────┘
```

## Key Components

### 1. API Configuration

```typescript
const API_BASE_URL = 'http://localhost:5000'
```

Change this for production deployment.

### 2. State Management

```typescript
const [inputText, setInputText] = useState('')
const [isGenerating, setIsGenerating] = useState(false)
const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null)
const [error, setError] = useState<string | null>(null)
```

### 3. Generation Handler

```typescript
const handleGenerate = async () => {
  setIsGenerating(true)
  setError(null)
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: inputText,
        temperature: 1.0,
        top_k: 20
      })
    })
    
    const data = await response.json()
    setGenerationResult(data)
  } catch (err) {
    setError(err.message)
  } finally {
    setIsGenerating(false)
  }
}
```

## API Request/Response Examples

### Generate Music from Text

**Request:**
```typescript
POST http://localhost:5000/api/generate
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
  "generation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "midi_file": "/api/download/a1b2c3d4-e5f6-7890-abcd-ef1234567890.mid",
  "emotion": "joy",
  "duration": 2.0,
  "tokens_generated": 512
}
```

### Download MIDI File

```typescript
const handleDownload = () => {
  window.open(`${API_BASE_URL}${generationResult.midi_file}`, '_blank')
}
```

## UI States

### 1. Idle State
- Input field enabled
- Submit button shows arrow icon
- No result or error displayed

### 2. Loading State
- Input field disabled
- Submit button shows spinner
- User cannot submit new requests

### 3. Success State
- Result container appears with animation
- Shows emotion, duration, and token count
- Download button available

### 4. Error State
- Error message displayed in red box
- Input field re-enabled
- User can retry

## Styling

### Color Scheme
- Background: `#242121`
- Input background: `#2a2525`
- Text: `#FFFFFF`
- Placeholder: `#888888`
- Error: `#ff3b30`
- Success gradient: `#667eea` → `#764ba2`

### Animations
- Fade in: Result container
- Spin: Loading spinner
- Hover effects: Buttons

## Error Handling

```typescript
try {
  const response = await fetch(...)
  if (!response.ok) {
    throw new Error('Failed to generate music')
  }
  // Handle success
} catch (err) {
  setError(err instanceof Error ? err.message : 'An error occurred')
}
```

## CORS Configuration

The backend must have CORS enabled:

```python
from flask_cors import CORS
CORS(app)
```

## Development Workflow

1. **Start Backend:**
   ```bash
   python3 api.py
   ```

2. **Start Frontend:**
   ```bash
   cd client
   npm run dev
   ```

3. **Test in Browser:**
   - Open `http://localhost:5173`
   - Click "Get Started"
   - Enter a prompt
   - Wait for generation
   - Download MIDI

## Production Deployment

### Frontend Build

```bash
cd client
npm run build
```

Output: `client/dist/`

### Environment Variables

Create `.env` file:

```
VITE_API_BASE_URL=https://api.yourdomain.com
```

Update `App.tsx`:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'
```

### Backend CORS

Update for production:

```python
CORS(app, origins=['https://yourdomain.com'])
```

## Troubleshooting

### CORS Errors

**Problem:** Browser console shows CORS error

**Solution:**
1. Ensure backend has `flask-cors` installed
2. Check CORS configuration in `api.py`
3. Verify API_BASE_URL is correct

### Network Errors

**Problem:** `Failed to fetch`

**Solution:**
1. Check backend is running on port 5000
2. Verify no firewall blocking
3. Check browser network tab for details

### Timeout Errors

**Problem:** Request times out

**Solution:**
1. Generation takes 15-30 seconds on CPU
2. Increase timeout if needed
3. Consider using GPU for faster generation

## Future Enhancements

- [ ] Add audio playback in browser
- [ ] Implement progress bar for generation
- [ ] Add history of generated tracks
- [ ] Support multiple emotion selection
- [ ] Add waveform visualization
- [ ] Implement user authentication
- [ ] Add sharing functionality

## Resources

- React Documentation: https://react.dev
- Vite Documentation: https://vitejs.dev
- Fetch API: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API
- TypeScript: https://www.typescriptlang.org

---

**Need Help?** Check the main `FULLSTACK_SETUP.md` guide.
