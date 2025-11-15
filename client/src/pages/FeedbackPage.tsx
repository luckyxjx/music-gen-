import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './FeedbackPage.css'

const API_BASE_URL = 'http://localhost:5001'

interface GenerationSample {
  id: string
  emotion: string
  duration: number
  midi_file: string
  audio_file?: string
  timestamp: string
  feedback?: FeedbackData
}

interface FeedbackData {
  emotion_accuracy: number  // 1-5 scale
  musical_quality: number   // 1-5 scale
  overall_rating: number    // 1-5 scale
  comments: string
  timestamp: string
}

function FeedbackPage() {
  const navigate = useNavigate()
  const [samples, setSamples] = useState<GenerationSample[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null)
  const [feedback, setFeedback] = useState<Partial<FeedbackData>>({
    emotion_accuracy: 3,
    musical_quality: 3,
    overall_rating: 3,
    comments: ''
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [message, setMessage] = useState<{type: 'success' | 'error', text: string} | null>(null)

  useEffect(() => {
    loadSamples()
  }, [])

  useEffect(() => {
    // Stop audio when changing samples
    if (audioElement) {
      audioElement.pause()
      setIsPlaying(false)
    }
  }, [currentIndex])

  const loadSamples = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/feedback/samples`)
      if (response.ok) {
        const data = await response.json()
        setSamples(data.samples || [])
      }
    } catch (error) {
      console.error('Failed to load samples:', error)
    }
  }

  const handlePlayPause = () => {
    const currentSample = samples[currentIndex]
    if (!currentSample?.audio_file) return

    if (audioElement) {
      if (isPlaying) {
        audioElement.pause()
        setIsPlaying(false)
      } else {
        audioElement.play()
        setIsPlaying(true)
      }
    } else {
      const audio = new Audio(`${API_BASE_URL}${currentSample.audio_file}`)
      audio.addEventListener('ended', () => setIsPlaying(false))
      audio.addEventListener('error', () => {
        setMessage({ type: 'error', text: 'Failed to load audio' })
        setIsPlaying(false)
      })
      audio.play()
      setAudioElement(audio)
      setIsPlaying(true)
    }
  }

  const handleSubmitFeedback = async () => {
    const currentSample = samples[currentIndex]
    if (!currentSample) return

    setIsSubmitting(true)
    setMessage(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/feedback/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          generation_id: currentSample.id,
          emotion: currentSample.emotion,
          ...feedback,
          timestamp: new Date().toISOString()
        })
      })

      if (response.ok) {
        setMessage({ type: 'success', text: 'Feedback submitted successfully!' })
        
        // Move to next sample
        setTimeout(() => {
          if (currentIndex < samples.length - 1) {
            setCurrentIndex(currentIndex + 1)
            setFeedback({
              emotion_accuracy: 3,
              musical_quality: 3,
              overall_rating: 3,
              comments: ''
            })
            setMessage(null)
          } else {
            setMessage({ type: 'success', text: 'All samples rated! Thank you!' })
          }
        }, 1500)
      } else {
        setMessage({ type: 'error', text: 'Failed to submit feedback' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Network error' })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSkip = () => {
    if (currentIndex < samples.length - 1) {
      setCurrentIndex(currentIndex + 1)
      setFeedback({
        emotion_accuracy: 3,
        musical_quality: 3,
        overall_rating: 3,
        comments: ''
      })
      setMessage(null)
    }
  }

  const currentSample = samples[currentIndex]

  if (samples.length === 0) {
    return (
      <div className="feedback-page">
        <div className="feedback-empty">
          <h2>No samples available for feedback</h2>
          <p>Generate some music first!</p>
          <button onClick={() => navigate('/chat')}>Go to Chat</button>
        </div>
      </div>
    )
  }

  return (
    <div className="feedback-page">
      <div className="feedback-header">
        <button className="back-btn" onClick={() => navigate('/chat')}>
          ← Back to Chat
        </button>
        <h1>Human Feedback</h1>
        <div className="progress-indicator">
          Sample {currentIndex + 1} of {samples.length}
        </div>
      </div>

      <div className="feedback-container">
        <div className="sample-info">
          <h2>Sample Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">ID:</span>
              <span className="info-value">{currentSample.id.slice(0, 8)}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Target Emotion:</span>
              <span className="info-value emotion-badge">{currentSample.emotion}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Duration:</span>
              <span className="info-value">{currentSample.duration} min</span>
            </div>
          </div>

          <div className="audio-player">
            <button className="play-button" onClick={handlePlayPause}>
              {isPlaying ? (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                  <rect x="6" y="4" width="4" height="16" fill="white" rx="1"/>
                  <rect x="14" y="4" width="4" height="16" fill="white" rx="1"/>
                </svg>
              ) : (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                  <path d="M8 5v14l11-7z" fill="white"/>
                </svg>
              )}
            </button>
            <span className="play-label">{isPlaying ? 'Playing...' : 'Click to Play'}</span>
          </div>
        </div>

        <div className="feedback-form">
          <h2>Rate This Sample</h2>

          <div className="rating-section">
            <label className="rating-label">
              Emotion Accuracy
              <span className="rating-description">How well does the music match the target emotion?</span>
            </label>
            <div className="rating-stars">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  className={`star ${(feedback.emotion_accuracy || 0) >= star ? 'active' : ''}`}
                  onClick={() => setFeedback({ ...feedback, emotion_accuracy: star })}
                >
                  ★
                </button>
              ))}
            </div>
          </div>

          <div className="rating-section">
            <label className="rating-label">
              Musical Quality
              <span className="rating-description">How coherent and pleasant is the music?</span>
            </label>
            <div className="rating-stars">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  className={`star ${(feedback.musical_quality || 0) >= star ? 'active' : ''}`}
                  onClick={() => setFeedback({ ...feedback, musical_quality: star })}
                >
                  ★
                </button>
              ))}
            </div>
          </div>

          <div className="rating-section">
            <label className="rating-label">
              Overall Rating
              <span className="rating-description">Your overall impression</span>
            </label>
            <div className="rating-stars">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  className={`star ${(feedback.overall_rating || 0) >= star ? 'active' : ''}`}
                  onClick={() => setFeedback({ ...feedback, overall_rating: star })}
                >
                  ★
                </button>
              ))}
            </div>
          </div>

          <div className="comments-section">
            <label className="rating-label">
              Comments (Optional)
              <span className="rating-description">Any additional thoughts?</span>
            </label>
            <textarea
              className="comments-input"
              placeholder="Share your thoughts about this music..."
              value={feedback.comments}
              onChange={(e) => setFeedback({ ...feedback, comments: e.target.value })}
              rows={4}
            />
          </div>

          {message && (
            <div className={`feedback-message ${message.type}`}>
              {message.text}
            </div>
          )}

          <div className="action-buttons">
            <button
              className="skip-btn"
              onClick={handleSkip}
              disabled={isSubmitting}
            >
              Skip
            </button>
            <button
              className="submit-btn"
              onClick={handleSubmitFeedback}
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FeedbackPage
