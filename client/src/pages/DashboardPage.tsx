import { useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import './DashboardPage.css'

interface Generation {
  id: string
  emotion: string
  duration: number
  timestamp: string
  midiFile: string
}

function DashboardPage() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'overview' | 'history' | 'profile'>('overview')
  const [generations, setGenerations] = useState<Generation[]>([])
  const [stats, setStats] = useState({
    totalGenerations: 0,
    totalMinutes: 0,
    favoriteEmotion: 'Joy',
    thisWeek: 0
  })

  useEffect(() => {
    // Load generation history from localStorage
    const history = localStorage.getItem('generationHistory')
    if (history) {
      const parsed = JSON.parse(history)
      setGenerations(parsed)
      
      // Calculate stats
      const total = parsed.length
      const totalMins = parsed.reduce((sum: number, gen: Generation) => sum + gen.duration, 0)
      const emotions = parsed.map((gen: Generation) => gen.emotion)
      const emotionCounts = emotions.reduce((acc: any, emotion: string) => {
        acc[emotion] = (acc[emotion] || 0) + 1
        return acc
      }, {})
      const favorite = Object.keys(emotionCounts).reduce((a, b) => 
        emotionCounts[a] > emotionCounts[b] ? a : b, 'Joy'
      )
      
      // Count this week's generations
      const oneWeekAgo = new Date()
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7)
      const thisWeekCount = parsed.filter((gen: Generation) => 
        new Date(gen.timestamp) > oneWeekAgo
      ).length

      setStats({
        totalGenerations: total,
        totalMinutes: Math.round(totalMins),
        favoriteEmotion: favorite,
        thisWeek: thisWeekCount
      })
    }
  }, [])

  const handleDownload = (midiFile: string, id: string) => {
    const link = document.createElement('a')
    link.href = midiFile
    link.download = `music-${id}.mid`
    link.click()
  }

  const handleDelete = (id: string) => {
    const updated = generations.filter(gen => gen.id !== id)
    setGenerations(updated)
    localStorage.setItem('generationHistory', JSON.stringify(updated))
  }

  const clearHistory = () => {
    if (window.confirm('Are you sure you want to clear all generation history?')) {
      setGenerations([])
      localStorage.removeItem('generationHistory')
      setStats({
        totalGenerations: 0,
        totalMinutes: 0,
        favoriteEmotion: 'Joy',
        thisWeek: 0
      })
    }
  }

  return (
    <div className="dashboard-page">
      <nav className="dashboard-nav">
        <div className="nav-brand" onClick={() => navigate('/')}>
          <h2>ONGOING BEATS</h2>
        </div>
        <div className="nav-actions">
          <button className="nav-btn" onClick={() => navigate('/chat')}>
            + New Generation
          </button>
          <button className="nav-btn secondary" onClick={() => navigate('/')}>
            Home
          </button>
        </div>
      </nav>

      <div className="dashboard-container">
        <aside className="dashboard-sidebar">
          <div className="user-profile">
            <div className="user-avatar">
              <span className="avatar-icon">◉</span>
            </div>
            <h3 className="user-name">Music Creator</h3>
            <p className="user-email">creator@ongoingbeats.com</p>
          </div>

          <div className="sidebar-menu">
            <button 
              className={`menu-item ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              <span className="menu-icon">▦</span>
              Overview
            </button>
            <button 
              className={`menu-item ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <span className="menu-icon">♪</span>
              Generation History
            </button>
            <button 
              className={`menu-item ${activeTab === 'profile' ? 'active' : ''}`}
              onClick={() => setActiveTab('profile')}
            >
              <span className="menu-icon">⚙</span>
              Profile Settings
            </button>
          </div>
        </aside>

        <main className="dashboard-main">
          {activeTab === 'overview' && (
            <div className="overview-section">
              <h1 className="section-title">Dashboard Overview</h1>
              
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon">🎼</div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.totalGenerations}</div>
                    <div className="stat-label">Total Generations</div>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">⏱️</div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.totalMinutes}</div>
                    <div className="stat-label">Minutes Created</div>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">❤️</div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.favoriteEmotion}</div>
                    <div className="stat-label">Favorite Emotion</div>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">📈</div>
                  <div className="stat-content">
                    <div className="stat-value">{stats.thisWeek}</div>
                    <div className="stat-label">This Week</div>
                  </div>
                </div>
              </div>

              <div className="quick-actions">
                <h2 className="subsection-title">Quick Actions</h2>
                <div className="action-cards">
                  <div className="action-card" onClick={() => navigate('/chat')}>
                    <div className="action-icon">🎹</div>
                    <h3 className="action-title">Generate Music</h3>
                    <p className="action-description">Create new AI-generated music</p>
                  </div>
                  <div className="action-card" onClick={() => setActiveTab('history')}>
                    <div className="action-icon">📚</div>
                    <h3 className="action-title">View History</h3>
                    <p className="action-description">Browse your past generations</p>
                  </div>
                  <div className="action-card" onClick={() => navigate('/services')}>
                    <div className="action-icon">🔧</div>
                    <h3 className="action-title">Explore Features</h3>
                    <p className="action-description">Learn about our services</p>
                  </div>
                </div>
              </div>

              <div className="recent-activity">
                <h2 className="subsection-title">Recent Activity</h2>
                {generations.length > 0 ? (
                  <div className="activity-list">
                    {generations.slice(0, 5).map((gen) => (
                      <div key={gen.id} className="activity-item">
                        <div className="activity-icon">♪</div>
                        <div className="activity-details">
                          <div className="activity-title">
                            Generated {gen.emotion} music
                          </div>
                          <div className="activity-time">
                            {new Date(gen.timestamp).toLocaleString()}
                          </div>
                        </div>
                        <div className="activity-meta">
                          {gen.duration} min
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    <p>No generations yet. Start creating music!</p>
                    <button className="cta-btn" onClick={() => navigate('/chat')}>
                      Create Your First Track
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'history' && (
            <div className="history-section">
              <div className="section-header">
                <h1 className="section-title">Generation History</h1>
                {generations.length > 0 && (
                  <button className="clear-btn" onClick={clearHistory}>
                    Clear All
                  </button>
                )}
              </div>

              {generations.length > 0 ? (
                <div className="history-grid">
                  {generations.map((gen) => (
                    <div key={gen.id} className="history-card">
                      <div className="history-header">
                        <div className="history-emotion">{gen.emotion}</div>
                        <div className="history-duration">{gen.duration} min</div>
                      </div>
                      <div className="history-body">
                        <div className="history-id">ID: {gen.id.slice(0, 8)}...</div>
                        <div className="history-time">
                          {new Date(gen.timestamp).toLocaleDateString()} at{' '}
                          {new Date(gen.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                      <div className="history-actions">
                        <button 
                          className="action-btn download"
                          onClick={() => handleDownload(gen.midiFile, gen.id)}
                        >
                          <span>⬇️</span> Download
                        </button>
                        <button 
                          className="action-btn delete"
                          onClick={() => handleDelete(gen.id)}
                        >
                          <span>🗑️</span> Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">♪</div>
                  <h3>No Generation History</h3>
                  <p>Your generated music will appear here</p>
                  <button className="cta-btn" onClick={() => navigate('/chat')}>
                    Generate Your First Track
                  </button>
                </div>
              )}
            </div>
          )}

          {activeTab === 'profile' && (
            <div className="profile-section">
              <h1 className="section-title">Profile Settings</h1>
              
              <div className="profile-card">
                <h2 className="card-title">Personal Information</h2>
                <div className="form-group">
                  <label className="form-label">Display Name</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    defaultValue="Music Creator"
                    placeholder="Your name"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input 
                    type="email" 
                    className="form-input" 
                    defaultValue="creator@ongoingbeats.com"
                    placeholder="your@email.com"
                  />
                </div>
                <button className="save-btn">Save Changes</button>
              </div>

              <div className="profile-card">
                <h2 className="card-title">Generation Preferences</h2>
                <div className="form-group">
                  <label className="form-label">Default Emotion</label>
                  <select className="form-select">
                    <option>Joy</option>
                    <option>Sadness</option>
                    <option>Anger</option>
                    <option>Calm</option>
                    <option>Surprise</option>
                    <option>Fear</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Default Duration (minutes)</label>
                  <input 
                    type="number" 
                    className="form-input" 
                    defaultValue="2"
                    min="1"
                    max="5"
                  />
                </div>
                <button className="save-btn">Save Preferences</button>
              </div>

              <div className="profile-card danger">
                <h2 className="card-title">Danger Zone</h2>
                <p className="card-description">
                  Clear all your generation history and reset your account data.
                </p>
                <button className="danger-btn" onClick={clearHistory}>
                  Clear All Data
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default DashboardPage
