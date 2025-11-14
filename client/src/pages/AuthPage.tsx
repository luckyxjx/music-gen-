import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './AuthPage.css'

function AuthPage() {
  const navigate = useNavigate()
  const [isLogin, setIsLogin] = useState(true)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: ''
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // For now, just navigate to chat page
    // In production, you would authenticate with backend
    localStorage.setItem('user', JSON.stringify({ email: formData.email, name: formData.name || 'User' }))
    navigate('/chat')
  }

  return (
    <div className="auth-page">
      <button className="back-btn" onClick={() => navigate('/')}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Back
      </button>

      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-header">
            <h1 className="auth-title">ONGOING BEATS</h1>
            <p className="auth-subtitle">AI-Powered Music Generation</p>
          </div>

          <div className="auth-tabs">
            <button 
              className={`auth-tab ${isLogin ? 'active' : ''}`}
              onClick={() => setIsLogin(true)}
            >
              Login
            </button>
            <button 
              className={`auth-tab ${!isLogin ? 'active' : ''}`}
              onClick={() => setIsLogin(false)}
            >
              Sign Up
            </button>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {!isLogin && (
              <div className="form-group">
                <label className="form-label">Name</label>
                <input
                  type="text"
                  name="name"
                  className="form-input"
                  value={formData.name}
                  onChange={handleChange}
                  required={!isLogin}
                  placeholder="Enter your name"
                />
              </div>
            )}

            <div className="form-group">
              <label className="form-label">Email</label>
              <input
                type="email"
                name="email"
                className="form-input"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="Enter your email"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <input
                type="password"
                name="password"
                className="form-input"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Enter your password"
              />
            </div>

            {isLogin && (
              <div className="form-options">
                <label className="checkbox-label">
                  <input type="checkbox" className="checkbox" />
                  <span>Remember me</span>
                </label>
                <a href="#" className="forgot-link">Forgot password?</a>
              </div>
            )}

            <button type="submit" className="submit-btn">
              {isLogin ? 'Login' : 'Sign Up'}
            </button>
          </form>

          <div className="auth-footer">
            <p className="footer-text">
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button 
                className="toggle-btn" 
                onClick={() => setIsLogin(!isLogin)}
              >
                {isLogin ? 'Sign Up' : 'Login'}
              </button>
            </p>
          </div>
        </div>

        <div className="auth-features">
          <h2 className="features-title">Why Join Us?</h2>
          <div className="feature-list">
            <div className="feature-item">
              <div className="feature-icon">♪</div>
              <div className="feature-content">
                <h3 className="feature-title">Unlimited Music Generation</h3>
                <p className="feature-text">Create as many tracks as you want</p>
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-icon">◆</div>
              <div className="feature-content">
                <h3 className="feature-title">6 Emotion Categories</h3>
                <p className="feature-text">Express every mood through music</p>
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-icon">↓</div>
              <div className="feature-content">
                <h3 className="feature-title">MIDI Export</h3>
                <p className="feature-text">Download and use in any DAW</p>
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-icon">∞</div>
              <div className="feature-content">
                <h3 className="feature-title">Free Forever</h3>
                <p className="feature-text">No credit card required</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AuthPage
