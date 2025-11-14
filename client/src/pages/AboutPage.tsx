import { useNavigate } from 'react-router-dom'
import './AboutPage.css'

function AboutPage() {
  const navigate = useNavigate()

  return (
    <div className="about-page">
      <button className="back-btn" onClick={() => navigate('/')}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Back
      </button>

      <div className="about-container">
        <div className="about-hero">
          <h1 className="about-title">ABOUT US</h1>
          <p className="about-subtitle">
            Two passionate students building the future of AI music generation
          </p>
        </div>

        <div className="team-section">
          <h2 className="section-heading">OUR TEAM</h2>
          
          <div className="team-grid">
            <div className="team-card">
              <div className="team-icon">♪</div>
              <h3 className="team-name">Lucky</h3>
              <p className="team-role">AI Model Architect</p>
              <div className="team-description">
                <p>
                  The mastermind behind our music generation model. Lucky designed and trained 
                  the Transformer neural network that powers Ongoing Beats, implementing advanced 
                  techniques in emotion conditioning and duration control.
                </p>
                <div className="team-skills">
                  <span className="skill-tag">Deep Learning</span>
                  <span className="skill-tag">PyTorch</span>
                  <span className="skill-tag">Transformers</span>
                  <span className="skill-tag">MIDI Processing</span>
                </div>
              </div>
            </div>

            <div className="team-card">
              <div className="team-icon">{ }</div>
              <h3 className="team-name">Ayush Bhattarai</h3>
              <p className="team-role">Full Stack Developer</p>
              <div className="team-description">
                <p>
                  Built the entire web platform and backend infrastructure. Ayush created the 
                  seamless user experience, REST API, and integrated the AI model into a 
                  production-ready application.
                </p>
                <div className="team-skills">
                  <span className="skill-tag">React</span>
                  <span className="skill-tag">Flask</span>
                  <span className="skill-tag">TypeScript</span>
                  <span className="skill-tag">API Design</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="institution-section">
          <div className="institution-card">
            <h2 className="institution-name">Jaypee Institute of Information Technology</h2>
            <p className="institution-location">Sector 62, Noida</p>
            <p className="institution-program">B.Tech Computer Science & Engineering</p>
            <div className="institution-description">
              <p>
                We are undergraduate students pursuing our passion for artificial intelligence 
                and software development. This project represents our commitment to pushing 
                the boundaries of what's possible with AI in creative domains.
              </p>
            </div>
          </div>
        </div>

        <div className="mission-section">
          <h2 className="section-heading">OUR MISSION</h2>
          <div className="mission-content">
            <div className="mission-card">
              <div className="mission-icon">🎯</div>
              <h3 className="mission-title">Democratize Music Creation</h3>
              <p className="mission-text">
                Make music generation accessible to everyone, regardless of musical training 
                or technical expertise.
              </p>
            </div>
            <div className="mission-card">
              <div className="mission-icon">🚀</div>
              <h3 className="mission-title">Push AI Boundaries</h3>
              <p className="mission-text">
                Explore the intersection of artificial intelligence and creativity, advancing 
                the state of the art in generative models.
              </p>
            </div>
            <div className="mission-card">
              <div className="mission-icon">💡</div>
              <h3 className="mission-title">Inspire Innovation</h3>
              <p className="mission-text">
                Show what students can achieve with dedication, proving that groundbreaking 
                projects can come from anywhere.
              </p>
            </div>
          </div>
        </div>

        <div className="project-stats">
          <div className="stat-item">
            <div className="stat-value">2</div>
            <div className="stat-label">Team Members</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">1</div>
            <div className="stat-label">Transformer Model</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">6</div>
            <div className="stat-label">Emotion Categories</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">∞</div>
            <div className="stat-label">Possibilities</div>
          </div>
        </div>

        <div className="cta-section">
          <h2 className="cta-heading">TRY OUR CREATION</h2>
          <p className="cta-text">
            Experience what we've built. Generate your own AI music in seconds.
          </p>
          <button className="cta-button" onClick={() => navigate('/chat')}>
            START CREATING NOW
          </button>
        </div>
      </div>
    </div>
  )
}

export default AboutPage
