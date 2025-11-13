import { useNavigate } from 'react-router-dom'
import { useEffect, useState, useRef } from 'react'
import './LandingPage.css'

function LandingPage() {
  const navigate = useNavigate()
  const [scrollY, setScrollY] = useState(0)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleScroll = () => {
      const scrollTop = container.scrollTop
      setScrollY(scrollTop)
      
      // Add scroll-based classes for animations
      const sections = container.querySelectorAll('section')
      sections.forEach((section) => {
        const rect = section.getBoundingClientRect()
        const isVisible = rect.top < window.innerHeight * 0.75 && rect.bottom > 0
        if (isVisible) {
          section.classList.add('in-view')
        }
      })
    }

    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }

    container.addEventListener('scroll', handleScroll)
    window.addEventListener('mousemove', handleMouseMove)
    
    // Initial check
    handleScroll()

    return () => {
      container.removeEventListener('scroll', handleScroll)
      window.removeEventListener('mousemove', handleMouseMove)
    }
  }, [])

  const parallaxOffset = {
    x: (mousePosition.x - window.innerWidth / 2) * 0.02,
    y: (mousePosition.y - window.innerHeight / 2) * 0.02
  }

  return (
    <div className="landing-container" ref={containerRef}>
      {/* Animated cursor follower */}
      <div 
        className="cursor-glow" 
        style={{
          left: `${mousePosition.x}px`,
          top: `${mousePosition.y}px`
        }}
      />
      {/* Section 1: Hero */}
      <section className="section-1">
        <div className="parallax-bg" style={{ transform: `translateY(${scrollY * 0.3}px)` }}></div>
        <nav className="nav">
          <div className="nav-left">
            <a href="#" className="nav-link" onClick={(e) => { e.preventDefault(); window.scrollTo(0, 0); }}>Home</a>
            <a href="#" className="nav-link" onClick={(e) => { e.preventDefault(); navigate('/services'); }}>Services</a>
            <a href="#" className="nav-link">Pricing</a>
            <a href="#" className="nav-link" onClick={(e) => { e.preventDefault(); navigate('/about'); }}>About us</a>
            <a href="#" className="nav-link" onClick={(e) => { e.preventDefault(); navigate('/contact'); }}>Contact</a>
          </div>
          <button className="join-btn" onClick={() => navigate('/chat')}>join now</button>
        </nav>
        
        <div 
          className="content" 
          style={{ 
            transform: `translate(calc(-50% + ${parallaxOffset.x}px), calc(-50% + ${scrollY * 0.15}px + ${parallaxOffset.y}px))` 
          }}
        >
          <p className="subtitle">Generate Your Sound Now!</p>
          <h1 className="main-title">
            <span className="title-bold">ONGOING</span>
            <br/>
            <span className="title-regular">BEATS</span>
          </h1>
          <button className="get-started-btn" onClick={() => navigate('/chat')}>Get Started</button>
        </div>
      </section>

      {/* Section 2: Emotion Stripes */}
      <section className="section-2">
        <div 
          className="stripe-container"
          style={{
            opacity: 1,
            transform: 'translateY(0px)'
          }}
        >
          <div className="stripe stripe-white">
            <div className="stripe-text">
              HAPPY • JOYFUL • EXCITED • ENERGETIC • UPBEAT • GROOVY • FUNKY • RHYTHMIC • HAPPY • JOYFUL • EXCITED • ENERGETIC • UPBEAT • GROOVY • FUNKY • RHYTHMIC • 
            </div>
            <div className="stripe-text">
              HAPPY • JOYFUL • EXCITED • ENERGETIC • UPBEAT • GROOVY • FUNKY • RHYTHMIC • HAPPY • JOYFUL • EXCITED • ENERGETIC • UPBEAT • GROOVY • FUNKY • RHYTHMIC • 
            </div>
          </div>
          
          <div className="stripe stripe-black">
            <div className="stripe-text">
              SAD • MELANCHOLIC • EMOTIONAL • THOUGHTFUL • MOODY • ATMOSPHERIC • MYSTERIOUS • DARK • SAD • MELANCHOLIC • EMOTIONAL • THOUGHTFUL • MOODY • ATMOSPHERIC • MYSTERIOUS • DARK • 
            </div>
            <div className="stripe-text">
              SAD • MELANCHOLIC • EMOTIONAL • THOUGHTFUL • MOODY • ATMOSPHERIC • MYSTERIOUS • DARK • SAD • MELANCHOLIC • EMOTIONAL • THOUGHTFUL • MOODY • ATMOSPHERIC • MYSTERIOUS • DARK • 
            </div>
          </div>
          
          <div className="stripe stripe-white">
            <div className="stripe-text">
              ANGRY • INTENSE • POWERFUL • AGGRESSIVE • CALM • PEACEFUL • SERENE • RELAXING • ANGRY • INTENSE • POWERFUL • AGGRESSIVE • CALM • PEACEFUL • SERENE • RELAXING • 
            </div>
            <div className="stripe-text">
              ANGRY • INTENSE • POWERFUL • AGGRESSIVE • CALM • PEACEFUL • SERENE • RELAXING • ANGRY • INTENSE • POWERFUL • AGGRESSIVE • CALM • PEACEFUL • SERENE • RELAXING • 
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: How It Works */}
      <section className="section-3">
        <div className="how-it-works-content">
          <h2 className="how-it-works-heading">HOW IT WORKS</h2>
          <p className="section-subtitle">Transform your words into music in three simple steps</p>
          
          <div className="steps-container">
            <div className="step step-1">
              <div className="step-number">01</div>
              <h3 className="step-title">DESCRIBE YOUR MOOD</h3>
              <p className="step-description">
                Tell us how you're feeling or what vibe you want. 
                <br/><br/>
                <span className="example-text">"Play me some funky music, I'm feeling good"</span>
                <br/><br/>
                Our AI understands emotions, genres, and musical styles.
              </p>
            </div>

            <div className="step-arrow">→</div>

            <div className="step step-2">
              <div className="step-number">02</div>
              <h3 className="step-title">AI GENERATES</h3>
              <p className="step-description">
                Our advanced AI analyzes your input and creates unique music tailored to your mood.
                <br/><br/>
                Every track is original and personalized just for you.
              </p>
            </div>

            <div className="step-arrow">→</div>

            <div className="step step-3">
              <div className="step-number">03</div>
              <h3 className="step-title">LISTEN & DOWNLOAD</h3>
              <p className="step-description">
                Stream your generated music instantly on our platform.
                <br/><br/>
                Download MIDI files to use in your own projects or share with friends.
              </p>
            </div>
          </div>

          <div className="cta-section">
            <p className="cta-text">Ready to create your perfect soundtrack?</p>
            <button className="cta-button" onClick={() => navigate('/chat')}>
              START CREATING NOW
            </button>
          </div>
        </div>
      </section>

      {/* Section 4: Features & Footer */}
      <section className="section-4">
        <div className="features-content">
          <h2 className="features-heading">WHY CHOOSE US</h2>
          
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3 className="feature-title">INSTANT GENERATION</h3>
              <p className="feature-text">
                Get your music in seconds. No waiting, no delays. Just instant creativity.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🎵</div>
              <h3 className="feature-title">UNIQUE EVERY TIME</h3>
              <p className="feature-text">
                Every track is 100% original. No two generations are ever the same.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🎨</div>
              <h3 className="feature-title">EMOTION-DRIVEN</h3>
              <p className="feature-text">
                Our AI understands feelings and creates music that matches your mood perfectly.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">💾</div>
              <h3 className="feature-title">DOWNLOAD & OWN</h3>
              <p className="feature-text">
                Download MIDI files and use them in your projects. Full creative freedom.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🚀</div>
              <h3 className="feature-title">NO LIMITS</h3>
              <p className="feature-text">
                Generate as much as you want. Experiment freely without restrictions.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🎧</div>
              <h3 className="feature-title">STREAM INSTANTLY</h3>
              <p className="feature-text">
                Listen to your creations right away. No downloads required to preview.
              </p>
            </div>
          </div>

          <div className="final-cta">
            <h3 className="final-cta-heading">READY TO CREATE YOUR SOUNDTRACK?</h3>
            <p className="final-cta-text">
              Join thousands of creators making music with AI. Start your journey today.
            </p>
            <button className="final-cta-button" onClick={() => navigate('/chat')}>
              GET STARTED FOR FREE
            </button>
          </div>

          <footer className="footer">
            <div className="footer-content">
              <div className="footer-brand">
                <h4 className="footer-logo">ONGOING BEATS</h4>
                <p className="footer-tagline">AI-Powered Music Generation</p>
              </div>
              
              <div className="footer-links">
                <div className="footer-column">
                  <h5 className="footer-column-title">Product</h5>
                  <a href="#" className="footer-link">Features</a>
                  <a href="#" className="footer-link">How it Works</a>
                  <a href="#" className="footer-link">Pricing</a>
                </div>
                
                <div className="footer-column">
                  <h5 className="footer-column-title">Company</h5>
                  <a href="#" className="footer-link">About Us</a>
                  <a href="#" className="footer-link">Contact</a>
                  <a href="#" className="footer-link">Blog</a>
                </div>
                
                <div className="footer-column">
                  <h5 className="footer-column-title">Legal</h5>
                  <a href="#" className="footer-link">Privacy Policy</a>
                  <a href="#" className="footer-link">Terms of Service</a>
                  <a href="#" className="footer-link">License</a>
                </div>
              </div>
            </div>
            
            <div className="footer-bottom">
              <p className="footer-copyright">© 2024 Ongoing Beats. All rights reserved.</p>
            </div>
          </footer>
        </div>
      </section>
    </div>
  )
}

export default LandingPage
