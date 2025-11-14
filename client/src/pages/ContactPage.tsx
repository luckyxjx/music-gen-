import { useNavigate } from 'react-router-dom'
import { useState } from 'react'
import './ContactPage.css'

function ContactPage() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  })
  const [submitted, setSubmitted] = useState(false)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Here you would typically send the form data to a backend
    console.log('Form submitted:', formData)
    setSubmitted(true)
    setTimeout(() => {
      setSubmitted(false)
      setFormData({ name: '', email: '', subject: '', message: '' })
    }, 3000)
  }

  return (
    <div className="contact-page">
      <button className="back-btn" onClick={() => navigate('/')}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Back
      </button>

      <div className="contact-container">
        <div className="contact-hero">
          <h1 className="contact-title">GET IN TOUCH</h1>
          <p className="contact-subtitle">
            Have questions? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
          </p>
        </div>

        <div className="contact-content">
          <div className="contact-info-section">
            <h2 className="info-heading">CONTACT INFORMATION</h2>
            
            <div className="info-cards">
              <div className="info-card">
                <div className="info-icon">@</div>
                <h3 className="info-title">Email</h3>
                <p className="info-text">ayush.bhattarai@example.com</p>
                <p className="info-text">lucky@example.com</p>
              </div>

              <div className="info-card">
                <div className="info-icon">■</div>
                <h3 className="info-title">Institution</h3>
                <p className="info-text">Jaypee Institute of Information Technology</p>
                <p className="info-text">Sector 62, Noida</p>
              </div>

              <div className="info-card">
                <div className="info-icon">◆</div>
                <h3 className="info-title">Project</h3>
                <p className="info-text">Ongoing Beats</p>
                <p className="info-text">AI Music Generation Platform</p>
              </div>
            </div>

            <div className="social-section">
              <h3 className="social-heading">CONNECT WITH US</h3>
              <div className="social-links">
                <a href="#" className="social-link">
                  <span className="social-icon">💻</span>
                  GitHub
                </a>
                <a href="#" className="social-link">
                  <span className="social-icon">💼</span>
                  LinkedIn
                </a>
                <a href="#" className="social-link">
                  <span className="social-icon">🐦</span>
                  Twitter
                </a>
              </div>
            </div>
          </div>

          <div className="contact-form-section">
            <h2 className="form-heading">SEND US A MESSAGE</h2>
            
            {submitted ? (
              <div className="success-message">
                <div className="success-icon">✓</div>
                <h3>Message Sent Successfully!</h3>
                <p>We'll get back to you as soon as possible.</p>
              </div>
            ) : (
              <form className="contact-form" onSubmit={handleSubmit}>
                <div className="form-group">
                  <label htmlFor="name" className="form-label">Your Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    className="form-input"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    placeholder="John Doe"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="email" className="form-label">Your Email</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    className="form-input"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    placeholder="john@example.com"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="subject" className="form-label">Subject</label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    className="form-input"
                    value={formData.subject}
                    onChange={handleChange}
                    required
                    placeholder="What's this about?"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="message" className="form-label">Message</label>
                  <textarea
                    id="message"
                    name="message"
                    className="form-textarea"
                    value={formData.message}
                    onChange={handleChange}
                    required
                    placeholder="Tell us what's on your mind..."
                    rows={6}
                  />
                </div>

                <button type="submit" className="submit-btn">
                  SEND MESSAGE
                </button>
              </form>
            )}
          </div>
        </div>

        <div className="faq-section">
          <h2 className="faq-heading">FREQUENTLY ASKED QUESTIONS</h2>
          <div className="faq-grid">
            <div className="faq-card">
              <h3 className="faq-question">How does the AI generate music?</h3>
              <p className="faq-answer">
                Our Transformer neural network is trained on musical patterns and uses emotion 
                conditioning to create unique compositions based on your input.
              </p>
            </div>
            <div className="faq-card">
              <h3 className="faq-question">Can I use the generated music commercially?</h3>
              <p className="faq-answer">
                Generated music is royalty-free for personal use. For commercial use, please 
                contact us to discuss licensing options.
              </p>
            </div>
            <div className="faq-card">
              <h3 className="faq-question">What formats are supported?</h3>
              <p className="faq-answer">
                We currently support MIDI file export, which is compatible with all major DAWs 
                and music production software.
              </p>
            </div>
            <div className="faq-card">
              <h3 className="faq-question">Is there an API available?</h3>
              <p className="faq-answer">
                Yes! We provide a REST API for developers who want to integrate our music 
                generation into their own applications.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ContactPage
