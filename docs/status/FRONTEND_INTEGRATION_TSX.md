# Frontend Integration - TypeScript/React

## TypeScript Types

```typescript
// types/api.ts

export interface GenerateRequest {
  text: string;
  temperature?: number;
  top_k?: number;
}

export interface GenerateByEmotionRequest {
  emotion: 'joy' | 'sadness' | 'anger' | 'calm' | 'surprise' | 'fear';
  duration: number;
  temperature?: number;
  top_k?: number;
}

export interface GenerateResponse {
  success: boolean;
  generation_id: string;
  midi_file: string;
  emotion: string;
  duration: number;
  tokens_generated: number;
}

export interface Emotion {
  id: number;
  name: string;
  description: string;
}

export interface EmotionsResponse {
  emotions: Emotion[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string | null;
}
```

## API Service

```typescript
// services/musicApi.ts

const API_BASE_URL = 'http://localhost:5000/api';

export class MusicAPI {
  
  static async generateFromText(
    text: string,
    temperature: number = 1.0,
    topK: number = 20
  ): Promise<GenerateResponse> {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        temperature,
        top_k: topK
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  static async generateByEmotion(
    emotion: string,
    duration: number,
    temperature: number = 1.0,
    topK: number = 20
  ): Promise<GenerateResponse> {
    const response = await fetch(`${API_BASE_URL}/generate-emotion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        emotion,
        duration,
        temperature,
        top_k: topK
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  static async getEmotions(): Promise<EmotionsResponse> {
    const response = await fetch(`${API_BASE_URL}/emotions`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  static async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  static getDownloadUrl(filename: string): string {
    return `${API_BASE_URL}/download/${filename}`;
  }
  
  static async downloadMidi(midiFile: string): Promise<Blob> {
    const response = await fetch(`http://localhost:5000${midiFile}`);
    
    if (!response.ok) {
      throw new Error(`Download error: ${response.statusText}`);
    }
    
    return response.blob();
  }
}
```

## React Hook

```typescript
// hooks/useMusicGeneration.ts

import { useState } from 'react';
import { MusicAPI } from '../services/musicApi';
import type { GenerateResponse } from '../types/api';

export const useMusicGeneration = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  
  const generateFromText = async (text: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await MusicAPI.generateFromText(text);
      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };
  
  const generateByEmotion = async (emotion: string, duration: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await MusicAPI.generateByEmotion(emotion, duration);
      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };
  
  const downloadMidi = async (midiFile: string, filename: string = 'music.mid') => {
    try {
      const blob = await MusicAPI.downloadMidi(midiFile);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Download failed';
      setError(errorMessage);
      throw err;
    }
  };
  
  return {
    loading,
    error,
    result,
    generateFromText,
    generateByEmotion,
    downloadMidi
  };
};
```

## Example Component

```tsx
// components/MusicGenerator.tsx

import React, { useState } from 'react';
import { useMusicGeneration } from '../hooks/useMusicGeneration';

export const MusicGenerator: React.FC = () => {
  const [text, setText] = useState('');
  const { loading, error, result, generateFromText, downloadMidi } = useMusicGeneration();
  
  const handleGenerate = async () => {
    if (!text.trim()) return;
    
    try {
      await generateFromText(text);
    } catch (err) {
      console.error('Generation failed:', err);
    }
  };
  
  const handleDownload = async () => {
    if (!result) return;
    
    try {
      await downloadMidi(result.midi_file, `${result.emotion}_${result.generation_id}.mid`);
    } catch (err) {
      console.error('Download failed:', err);
    }
  };
  
  return (
    <div className="music-generator">
      <h2>Generate Music from Text</h2>
      
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="I'm happy, give me an upbeat 2-minute track"
        rows={4}
        disabled={loading}
      />
      
      <button onClick={handleGenerate} disabled={loading || !text.trim()}>
        {loading ? 'Generating...' : 'Generate Music'}
      </button>
      
      {error && (
        <div className="error">
          Error: {error}
        </div>
      )}
      
      {result && (
        <div className="result">
          <h3>Generated Successfully!</h3>
          <p>Emotion: {result.emotion}</p>
          <p>Duration: {result.duration} minutes</p>
          <p>Tokens: {result.tokens_generated}</p>
          
          <button onClick={handleDownload}>
            Download MIDI
          </button>
        </div>
      )}
    </div>
  );
};
```

## Emotion Selector Component

```tsx
// components/EmotionSelector.tsx

import React, { useState, useEffect } from 'react';
import { MusicAPI } from '../services/musicApi';
import { useMusicGeneration } from '../hooks/useMusicGeneration';
import type { Emotion } from '../types/api';

export const EmotionSelector: React.FC = () => {
  const [emotions, setEmotions] = useState<Emotion[]>([]);
  const [selectedEmotion, setSelectedEmotion] = useState<string>('joy');
  const [duration, setDuration] = useState<number>(2.0);
  const { loading, error, result, generateByEmotion, downloadMidi } = useMusicGeneration();
  
  useEffect(() => {
    loadEmotions();
  }, []);
  
  const loadEmotions = async () => {
    try {
      const response = await MusicAPI.getEmotions();
      setEmotions(response.emotions);
    } catch (err) {
      console.error('Failed to load emotions:', err);
    }
  };
  
  const handleGenerate = async () => {
    try {
      await generateByEmotion(selectedEmotion, duration);
    } catch (err) {
      console.error('Generation failed:', err);
    }
  };
  
  const handleDownload = async () => {
    if (!result) return;
    
    try {
      await downloadMidi(result.midi_file, `${result.emotion}_${duration}min.mid`);
    } catch (err) {
      console.error('Download failed:', err);
    }
  };
  
  return (
    <div className="emotion-selector">
      <h2>Generate by Emotion</h2>
      
      <div className="emotion-grid">
        {emotions.map((emotion) => (
          <button
            key={emotion.id}
            className={`emotion-btn ${selectedEmotion === emotion.name ? 'selected' : ''}`}
            onClick={() => setSelectedEmotion(emotion.name)}
            disabled={loading}
          >
            <div className="emotion-name">{emotion.name}</div>
            <div className="emotion-desc">{emotion.description}</div>
          </button>
        ))}
      </div>
      
      <div className="duration-control">
        <label>
          Duration: {duration} minutes
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.5"
            value={duration}
            onChange={(e) => setDuration(parseFloat(e.target.value))}
            disabled={loading}
          />
        </label>
      </div>
      
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Music'}
      </button>
      
      {error && <div className="error">Error: {error}</div>}
      
      {result && (
        <div className="result">
          <h3>✓ Generated!</h3>
          <button onClick={handleDownload}>Download MIDI</button>
        </div>
      )}
    </div>
  );
};
```

## Environment Configuration

```typescript
// config/env.ts

export const config = {
  apiBaseUrl: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  isDevelopment: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
};
```

## .env File

```bash
# .env.development
REACT_APP_API_URL=http://localhost:5000/api

# .env.production
REACT_APP_API_URL=https://your-api-domain.com/api
```

---

## Integration Steps

1. **Add types** to your project
2. **Create API service** with TypeScript types
3. **Create custom hooks** for state management
4. **Update components** to use the API
5. **Configure environment** variables
6. **Test integration** with backend

---

## Benefits of TSX

✅ **Type Safety** - Catch errors at compile time  
✅ **IntelliSense** - Better autocomplete in IDE  
✅ **Refactoring** - Easier to maintain  
✅ **Documentation** - Types serve as docs  
✅ **Error Prevention** - Fewer runtime errors

---

## Next Steps

1. Add your TSX frontend to the repo
2. Copy the types and API service
3. Update API_BASE_URL to match your setup
4. Test the integration
5. Build and deploy!

The API works perfectly with TypeScript/React! 🚀
