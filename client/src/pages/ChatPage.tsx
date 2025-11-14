import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './ChatPage.css'

const API_BASE_URL = 'http://localhost:5001'

interface GenerationResult {
  success: boolean
  generation_id: string
  midi_file: string
  emotion: string
  duration: number
  tokens_generated: number
}

interface ChatSession {
  id: string
  title: string
  messages: Array<{type: 'user' | 'bot', message: string, result?: GenerationResult}>
  timestamp: string
}

function ChatPage() {
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [inputText, setInputText] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  useEffect(() => {
    // Load sessions from localStorage
    const savedSessions = localStorage.getItem('chatSessions')
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions)
      setSessions(parsed)
      if (parsed.length > 0) {
        setCurrentSessionId(parsed[0].id)
      }
    } else {
      // Create first session
      createNewSession()
    }
  }, [])

  const createNewSession = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      timestamp: new Date().toISOString()
    }
    const updatedSessions = [newSession, ...sessions]
    setSessions(updatedSessions)
    setCurrentSessionId(newSession.id)
    localStorage.setItem('chatSessions', JSON.stringify(updatedSessions))
  }

  const deleteSession = (sessionId: string) => {
    const updatedSessions = sessions.filter(s => s.id !== sessionId)
    setSessions(updatedSessions)
    localStorage.setItem('chatSessions', JSON.stringify(updatedSessions))
    
    if (currentSessionId === sessionId) {
      if (updatedSessions.length > 0) {
        setCurrentSessionId(updatedSessions[0].id)
      } else {
        createNewSession()
      }
    }
  }

  const updateSessionTitle = (sessionId: string, firstMessage: string) => {
    const title = firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : '')
    const updatedSessions = sessions.map(s => 
      s.id === sessionId ? { ...s, title } : s
    )
    setSessions(updatedSessions)
    localStorage.setItem('chatSessions', JSON.stringify(updatedSessions))
  }

  const handleGenerate = async () => {
    if (!inputText.trim() || !currentSessionId) return

    const userInput = inputText
    const currentSession = sessions.find(s => s.id === currentSessionId)
    if (!currentSession) return

    // Update title if this is the first user message
    const userMessages = currentSession.messages.filter(m => m.type === 'user')
    if (userMessages.length === 0) {
      updateSessionTitle(currentSessionId, userInput)
    }

    // Add user message
    const updatedMessages = [
      ...currentSession.messages,
      { type: 'user' as const, message: userInput },
      { type: 'bot' as const, message: '♪ Generating music...' }
    ]
    
    const updatedSessions = sessions.map(s =>
      s.id === currentSessionId ? { ...s, messages: updatedMessages } : s
    )
    setSessions(updatedSessions)
    setInputText('')
    setIsGenerating(true)
    setError(null)

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
      
      // Save to generation history
      const historyItem = {
        id: data.generation_id,
        emotion: data.emotion,
        duration: data.duration,
        timestamp: new Date().toISOString(),
        midiFile: `${API_BASE_URL}${data.midi_file}`
      }
      
      const existingHistory = localStorage.getItem('generationHistory')
      const history = existingHistory ? JSON.parse(existingHistory) : []
      history.unshift(historyItem)
      localStorage.setItem('generationHistory', JSON.stringify(history))
      
      // Update session with result
      const finalMessages = [
        ...currentSession.messages,
        { type: 'user' as const, message: userInput },
        { 
          type: 'bot' as const, 
          message: `✓ Music generated! Emotion: ${data.emotion}, Duration: ${data.duration} minutes`,
          result: data
        }
      ]
      
      const finalSessions = sessions.map(s =>
        s.id === currentSessionId ? { ...s, messages: finalMessages } : s
      )
      setSessions(finalSessions)
      localStorage.setItem('chatSessions', JSON.stringify(finalSessions))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      
      // Update with error message
      const errorMessages = [
        ...currentSession.messages.slice(0, -1),
        { type: 'bot' as const, message: '✗ Failed to generate music. Please try again.' }
      ]
      
      const errorSessions = sessions.map(s =>
        s.id === currentSessionId ? { ...s, messages: errorMessages } : s
      )
      setSessions(errorSessions)
      localStorage.setItem('chatSessions', JSON.stringify(errorSessions))
    } finally {
      setIsGenerating(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isGenerating) {
      handleGenerate()
    }
  }

  const handleDownload = (midiFile: string) => {
    window.open(`${API_BASE_URL}${midiFile}`, '_blank')
  }

  const handleLogout = () => {
    localStorage.removeItem('user')
    navigate('/')
  }

  const currentSession = sessions.find(s => s.id === currentSessionId)

  return (
    <div className="chat-page">
      {/* Sidebar */}
      <div className={`chat-sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2 className="sidebar-title">ONGOING BEATS</h2>
          <button className="new-chat-btn" onClick={createNewSession}>
            + New Chat
          </button>
        </div>

        <div className="sessions-list">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${currentSessionId === session.id ? 'active' : ''}`}
              onClick={() => setCurrentSessionId(session.id)}
            >
              <div className="session-content">
                <div className="session-title">{session.title}</div>
                <div className="session-time">
                  {new Date(session.timestamp).toLocaleDateString()}
                </div>
              </div>
              <button
                className="delete-session-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  deleteSession(session.id)
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <button className="sidebar-action-btn" onClick={() => navigate('/dashboard')}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" fill="white"/>
            </svg>
            Dashboard
          </button>
          <button className="sidebar-action-btn logout" onClick={handleLogout}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Logout
          </button>
        </div>
      </div>

      {/* Sidebar Toggle Button */}
      <button 
        className="sidebar-toggle-btn" 
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          {isSidebarOpen ? (
            <path d="M15 18l-6-6 6-6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          ) : (
            <path d="M9 18l6-6-6-6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          )}
        </svg>
      </button>

      {/* Main Chat Area */}
      <div className="chat-main">
        <div className="chat-messages">
          {currentSession?.messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-content">
                <h1 className="welcome-title">ONGOING BEATS</h1>
                <p className="welcome-subtitle">AI-Powered Music Generation</p>
                
                <div className="welcome-examples">
                  <h2 className="examples-title">Try asking for:</h2>
                  <div className="example-cards">
                    <div className="example-card" onClick={() => setInputText('Create happy upbeat music for 2 minutes')}>
                      <div className="example-icon">♪</div>
                      <p className="example-text">"Create happy upbeat music for 2 minutes"</p>
                    </div>
                    <div className="example-card" onClick={() => setInputText('Play me some calm peaceful music for 3 minutes')}>
                      <div className="example-icon">~</div>
                      <p className="example-text">"Play me some calm peaceful music for 3 minutes"</p>
                    </div>
                    <div className="example-card" onClick={() => setInputText('Generate intense energetic music for 1 minute')}>
                      <div className="example-icon">⚡</div>
                      <p className="example-text">"Generate intense energetic music for 1 minute"</p>
                    </div>
                    <div className="example-card" onClick={() => setInputText('Make mysterious dark music for 4 minutes')}>
                      <div className="example-icon">◐</div>
                      <p className="example-text">"Make mysterious dark music for 4 minutes"</p>
                    </div>
                  </div>
                </div>

                <div className="welcome-features">
                  <div className="feature-badge">
                    <span className="badge-icon">∞</span>
                    <span className="badge-text">Unlimited Generations</span>
                  </div>
                  <div className="feature-badge">
                    <span className="badge-icon">◆</span>
                    <span className="badge-text">6 Emotions</span>
                  </div>
                  <div className="feature-badge">
                    <span className="badge-icon">↓</span>
                    <span className="badge-text">MIDI Export</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            currentSession?.messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">
                  {msg.message}
                  {msg.result && (
                    <div className="message-result">
                      <div className="result-info">
                        <span>ID: {msg.result.generation_id.slice(0, 8)}</span>
                        <span>Tokens: {msg.result.tokens_generated}</span>
                      </div>
                      <button 
                        className="download-btn-inline" 
                        onClick={() => handleDownload(msg.result!.midi_file)}
                      >
                        Download MIDI
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="chat-input-container">
          {error && <div className="error-message">{error}</div>}
          <div className="chat-input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="Describe your music... (e.g., 'Play me happy upbeat music for 2 minutes')"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isGenerating}
            />
            <button
              className="send-btn"
              onClick={handleGenerate}
              disabled={isGenerating || !inputText.trim()}
            >
              {isGenerating ? '...' : '→'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
