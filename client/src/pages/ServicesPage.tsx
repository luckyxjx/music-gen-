import { useNavigate } from 'react-router-dom'
import './ServicesPage.css'

function ServicesPage() {
  const navigate = useNavigate()

  const services = [
    {
      id: 1,
      title: 'AI-Powered Music Generation',
      description: 'Our advanced Transformer neural network model generates completely original music compositions. Trained on diverse musical patterns, it creates unique melodies, harmonies, and rhythms that never repeat.',
      features: [
        'Transformer architecture with 256-dimensional embeddings',
        '4-layer deep neural network with multi-head attention',
        'Real-time generation with GPU/MPS acceleration',
        'Generates up to 512 musical tokens per composition',
        'Completely original - no two tracks are identical'
      ]
    },
    {
      id: 2,
      title: 'Natural Language to Music',
      description: 'Simply type what you want to hear in plain English. Our intelligent text parser understands your mood, desired duration, and musical preferences, then generates music that matches your description perfectly.',
      features: [
        'Conversational input: "Play me happy music for 3 minutes"',
        'Automatic emotion detection from your words',
        'Duration parsing (1-5 minutes)',
        'Context-aware interpretation',
        'No musical knowledge required'
      ]
    },
    {
      id: 3,
      title: '6 Emotion Categories',
      description: 'Each emotion is deeply embedded into the generation process, creating authentic musical expressions. Our model conditions every note on your chosen emotion for consistent, mood-appropriate compositions.',
      features: [
        'Joy: Happy, upbeat, energetic melodies',
        'Sadness: Melancholic, slow, emotional pieces',
        'Anger: Intense, aggressive, fast-paced rhythms',
        'Calm: Peaceful, relaxed, serene atmospheres',
        'Surprise: Unexpected, varied, dynamic patterns',
        'Fear: Tense, anxious, uncertain progressions'
      ]
    },
    {
      id: 4,
      title: 'Precise Duration Control',
      description: 'Control exactly how long your music plays. Our duration conditioning system ensures your track is the perfect length for your needs, from quick 1-minute intros to full 5-minute compositions.',
      features: [
        'Range: 1 to 5 minutes',
        'Embedded duration control in the model',
        'Consistent pacing throughout',
        'Perfect for specific use cases',
        'No awkward cuts or loops'
      ]
    },
    {
      id: 5,
      title: 'MIDI File Export',
      description: 'Download your generated music as standard MIDI files. Compatible with all major DAWs (Ableton, FL Studio, Logic Pro, etc.), you can edit, remix, and use them in any project.',
      features: [
        'Standard MIDI format (.mid)',
        'Compatible with all DAWs',
        'Fully editable tracks',
        'Royalty-free for personal use',
        'Instant download after generation'
      ]
    },
    {
      id: 6,
      title: 'Instant Playback & Streaming',
      description: 'Listen to your generated music immediately in your browser. Our built-in MIDI player lets you preview tracks instantly without downloading, so you can iterate quickly until you find the perfect sound.',
      features: [
        'Browser-based MIDI playback',
        'No downloads needed to preview',
        'Instant streaming after generation',
        'Quick iteration workflow',
        'Play/pause controls'
      ]
    },
    {
      id: 7,
      title: 'Advanced Generation Controls',
      description: 'Fine-tune your music generation with professional parameters. Adjust temperature for creativity vs. consistency, and control top-k sampling for more focused or diverse outputs.',
      features: [
        'Temperature control (0.1-2.0)',
        'Top-k sampling for quality control',
        'Adjustable creativity levels',
        'Professional-grade parameters',
        'Real-time parameter updates'
      ]
    },
    {
      id: 8,
      title: 'REST API Access',
      description: 'Integrate our music generation into your own applications. Our RESTful API provides programmatic access to all features with simple JSON requests and responses.',
      features: [
        'POST /api/generate - Text-to-music endpoint',
        'POST /api/generate-emotion - Direct emotion control',
        'GET /api/download - MIDI file retrieval',
        'GET /api/emotions - List all emotions',
        'Full CORS support for web apps'
      ]
    },
    {
      id: 9,
      title: 'Multi-Device Support',
      description: 'Our system automatically detects and uses the best available hardware on your device. Whether you have a GPU, Apple Silicon, or CPU, you get optimized performance.',
      features: [
        'CUDA GPU acceleration (NVIDIA)',
        'Apple MPS support (M1/M2/M3)',
        'CPU fallback for compatibility',
        'Automatic device detection',
        'Optimized for all platforms'
      ]
    },
    {
      id: 10,
      title: 'Unique Token-Based System',
      description: 'Our custom MIDI tokenizer converts musical elements into a language the AI understands. This allows for precise control over notes, timing, velocity, and musical structure.',
      features: [
        'Custom MIDI tokenization',
        'Note, timing, and velocity encoding',
        'Structured musical grammar',
        'Efficient token vocabulary',
        'Lossless MIDI reconstruction'
      ]
    }
  ]

  return (
    <div className="services-page">
      <button className="back-btn" onClick={() => navigate('/')}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Back
      </button>

      <div className="services-container">
        <div className="services-hero">
          <h1 className="services-title">COMPLETE MUSIC GENERATION PLATFORM</h1>
          <p className="services-subtitle">
            Everything you need to create, customize, and export AI-generated music
          </p>
          <div className="services-stats">
            <div className="stat">
              <div className="stat-number">10+</div>
              <div className="stat-label">Core Features</div>
            </div>
            <div className="stat">
              <div className="stat-number">6</div>
              <div className="stat-label">Emotion Categories</div>
            </div>
            <div className="stat">
              <div className="stat-number">1-5</div>
              <div className="stat-label">Minutes Duration</div>
            </div>
            <div className="stat">
              <div className="stat-number">∞</div>
              <div className="stat-label">Unique Tracks</div>
            </div>
          </div>
        </div>

        <div className="services-grid">
          {services.map((service, index) => (
            <div 
              key={service.id} 
              className="service-card"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="service-number">{String(service.id).padStart(2, '0')}</div>
              <h3 className="service-title">{service.title}</h3>
              <p className="service-description">{service.description}</p>
              <ul className="service-features">
                {service.features.map((feature, idx) => (
                  <li key={idx} className="service-feature">
                    <span className="feature-bullet">•</span>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="services-cta">
          <h2 className="cta-heading">READY TO START CREATING?</h2>
          <p className="cta-text">
            Access our complete music generation platform with all features included.
            <br />
            No credit card required. No limits. Just pure creativity.
          </p>
          <div className="cta-features">
            <span className="cta-feature">✓ Unlimited Generations</span>
            <span className="cta-feature">✓ All Emotions Available</span>
            <span className="cta-feature">✓ MIDI Downloads</span>
            <span className="cta-feature">✓ API Access</span>
          </div>
          <button className="cta-button" onClick={() => navigate('/chat')}>
            START GENERATING MUSIC NOW
          </button>
        </div>
      </div>
    </div>
  )
}

export default ServicesPage
