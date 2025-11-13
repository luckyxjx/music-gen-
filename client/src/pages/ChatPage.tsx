import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './ChatPage.css'

// API configuration
const API_BASE_URL = 'http://localhost:5001'

interface GenerationResult {
  success: boolean
  generation_id: string
  midi_file: string
  emotion: string
  duration: number
  tokens_generated: number
}

interface MusicCategory {
  id: string
  name: string
  emotion: string
  description: string
  icon: string
}

const musicCategories: MusicCategory[] = [
  { id: 'happy', name: 'Happy & Upbeat', emotion: 'joy', description: 'Energetic and joyful vibes', icon: '😊' },
  { id: 'sad', name: 'Sad & Melancholic', emotion: 'sadness', description: 'Emotional and thoughtful', icon: '😢' },
  { id: 'calm', name: 'Calm & Peaceful', emotion: 'calm', description: 'Relaxing and serene', icon: '😌' },
  { id: 'energetic', name: 'Energetic & Intense', emotion: 'anger', description: 'Powerful and aggressive', icon: '⚡' },
  { id: 'mysterious', name: 'Mysterious & Dark', emotion: 'fear', description: 'Tense and atmospheric', icon: '🌙' },
  { id: 'surprise', name: 'Surprising & Varied', emotion: 'surprise', description: 'Unexpected and dynamic', icon: '✨' },
]

function ChatPage() {
  const navigate = useNavigate()
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [duration, setDuration] = useState(2)
  const [inputText, setInputText] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [chatHistory, setChatHistory] = useState<Array<{type: 'user' | 'bot', message: string}>>([
    { type: 'bot', message: 'Hello! I can help you create music. Describe what you want to hear, or choose a category from the left.' }
  ])

  const handleCategoryGenerate = async () => {
    if (!selectedCategory) {
      setError('Please select a category')
      return
    }

    const category = musicCategories.find(c => c.id === selectedCategory)
    if (!category) return

    setIsGenerating(true)
    setError(null)
    setGenerationResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-emotion`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          emotion: category.emotion,
          duration: duration,
          temperature: 1.0,
          top_k: 20
        })
      })

      if (!response.ok) {
        throw new Error('Failed to generate music')
      }

      const data: GenerationResult = await response.json()
      setGenerationResult(data)
      setChatHistory(prev => [...prev, 
        { type: 'bot', message: `✓ Generated ${category.name} music (${duration} minutes)` }
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsGenerating(false)
    }
  }

  const handleChatGenerate = async () => {
    if (!inputText.trim()) {
      setError('Please enter a description')
      return
    }

    const userInput = inputText
    
    // Add user message
    setChatHistory(prev => [...prev, { type: 'user', message: userInput }])
    setInputText('')
    
    // Add "Generating music..." message immediately
    setChatHistory(prev => [...prev, { type: 'bot', message: '🎵 Generating music...' }])
    
    setIsGenerating(true)
    setError(null)
    setGenerationResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: userInput,
          temperature: 1.0,
          top_k: 20
        })
      })

      if (!response.ok) {
        throw new Error('Failed to generate music')
      }

      const data: GenerationResult = await response.json()
      setGenerationResult(data)
      
      // Replace "Generating..." with success message
      setChatHistory(prev => {
        const newHistory = [...prev]
        newHistory[newHistory.length - 1] = { 
          type: 'bot', 
          message: `✓ Music generated successfully! Emotion: ${data.emotion}, Duration: ${data.duration} minutes` 
        }
        return newHistory
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      
      // Replace "Generating..." with error message
      setChatHistory(prev => {
        const newHistory = [...prev]
        newHistory[newHistory.length - 1] = { 
          type: 'bot', 
          message: '✗ Failed to generate music. Please try again.' 
        }
        return newHistory
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = () => {
    if (generationResult) {
      window.open(`${API_BASE_URL}${generationResult.midi_file}`, '_blank')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isGenerating) {
      handleChatGenerate()
    }
  }

  return (
    <div className="chat-page">
      <button className="back-btn" onClick={() => navigate('/')}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Back
      </button>

      <div className="chat-container">
        {/* Left Panel - Categories */}
        <div className="left-panel">
          <h2 className="panel-title">QUICK GENERATE</h2>
          <p className="panel-subtitle">Choose a mood and duration</p>

          <div className="categories-list">
            {musicCategories.map((category) => (
              <button
                key={category.id}
                className={`category-card ${selectedCategory === category.id ? 'selected' : ''}`}
                onClick={() => setSelectedCategory(category.id)}
              >
                <span className="category-icon">{category.icon}</span>
                <div className="category-info">
                  <h3 className="category-name">{category.name}</h3>
                  <p className="category-description">{category.description}</p>
                </div>
              </button>
            ))}
          </div>

          <div className="duration-control">
            <label className="duration-label">Duration: {duration} minutes</label>
            <input
              type="range"
              min="1"
              max="5"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="duration-slider"
            />
          </div>

          <button
            className="generate-btn"
            onClick={handleCategoryGenerate}
            disabled={!selectedCategory || isGenerating}
          >
            {isGenerating ? 'GENERATING...' : 'GENERATE MUSIC'}
          </button>
        </div>

        {/* Right Panel - Chat */}
        <div className="right-panel">
          <h2 className="panel-title">AI CHAT</h2>
          <p className="panel-subtitle">Describe your perfect soundtrack</p>

          <div className="chat-messages">
            {chatHistory.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">{msg.message}</div>
              </div>
            ))}
            {isGenerating && (
              <div className="message bot">
                <div className="message-content typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="E.g., Play me some funky music, I'm feeling good..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isGenerating}
            />
            <button
              className="send-btn"
              onClick={handleChatGenerate}
              disabled={isGenerating || !inputText.trim()}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>

          {error && (
            <div className="error-message">{error}</div>
          )}

          {generationResult && (
            <div className="result-panel">
              <h3 className="result-title">✓ Music Generated!</h3>
              <div className="result-info">
                <span>Emotion: {generationResult.emotion}</span>
                <span>Duration: {generationResult.duration}min</span>
                <span>Tokens: {generationResult.tokens_generated}</span>
              </div>
              <button className="download-btn" onClick={handleDownload}>
                DOWNLOAD MIDI
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChatPage
