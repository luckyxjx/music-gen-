import { useEffect, useRef, useState } from 'react'
import './App.css'

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

function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'chat'>('landing')
  const [inputText, setInputText] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chatSphereRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const setCanvasSize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    setCanvasSize()
    window.addEventListener('resize', setCanvasSize)

    // Sphere parameters - bigger sphere
    const radius = 220
    const numPoints = 400
    const points: Array<{ x: number; y: number; z: number; originalX: number; originalY: number; originalZ: number }> = []
    
    // Mouse tracking for cursor interaction
    let mouseX = canvas.width / 2
    let mouseY = canvas.height / 2
    let targetMouseX = mouseX
    let targetMouseY = mouseY
    
    const handleMouseMove = (e: MouseEvent) => {
      targetMouseX = e.clientX
      targetMouseY = e.clientY
    }
    
    canvas.addEventListener('mousemove', handleMouseMove)

    // Generate points on sphere surface using Fibonacci sphere algorithm for uniform distribution
    const goldenAngle = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < numPoints; i++) {
      const y = 1 - (i / (numPoints - 1)) * 2
      const radiusAtY = Math.sqrt(1 - y * y)
      const theta = goldenAngle * i
      const x = Math.cos(theta) * radiusAtY
      const z = Math.sin(theta) * radiusAtY
      
      points.push({
        x: x * radius,
        y: y * radius,
        z: z * radius,
        originalX: x,
        originalY: y,
        originalZ: z
      })
    }

    let time = 0
    const animate = () => {
      // Draw circular gradient background for the entire site
      const siteGradient = ctx.createRadialGradient(
        canvas.width / 2,
        canvas.height / 2,
        0,
        canvas.width / 2,
        canvas.height / 2,
        Math.max(canvas.width, canvas.height) * 0.8
      )
      siteGradient.addColorStop(0, '#2a2525')
      siteGradient.addColorStop(0.5, '#242121')
      siteGradient.addColorStop(1, '#1f1c1c')
      
      ctx.fillStyle = siteGradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Smooth mouse position interpolation - slower
      mouseX += (targetMouseX - mouseX) * 0.05
      mouseY += (targetMouseY - mouseY) * 0.05

      // Calculate center position (recalculate on each frame for responsiveness)
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2 - 100
      
      // Calculate direction from sphere center to cursor
      const dx = mouseX - centerX
      const dy = mouseY - centerY
      const cursorDistance = Math.sqrt(dx * dx + dy * dy)
      const cursorDirX = cursorDistance > 0 ? dx / cursorDistance : 0
      const cursorDirY = cursorDistance > 0 ? dy / cursorDistance : 0

      // Smooth and slow rotations
      const rotationX = Math.sin(time * 0.0008) * 0.5 + Math.cos(time * 0.0006) * 0.2
      const rotationY = time * 0.0004
      const rotationZ = Math.sin(time * 0.0006) * 0.3

      const projectedPoints = points.map(point => {
        // Apply rotation
        let x = point.originalX
        let y = point.originalY
        let z = point.originalZ

        // Rotate around Y axis
        const cosY = Math.cos(rotationY)
        const sinY = Math.sin(rotationY)
        const x1 = x * cosY - z * sinY
        const z1 = x * sinY + z * cosY

        // Rotate around X axis
        const cosX = Math.cos(rotationX)
        const sinX = Math.sin(rotationX)
        const y1 = y * cosX - z1 * sinX
        const z2 = y * sinX + z1 * cosX

        // Rotate around Z axis for more dynamic movement
        const cosZ = Math.cos(rotationZ)
        const sinZ = Math.sin(rotationZ)
        const x2 = x1 * cosZ - y1 * sinZ
        const y2 = x1 * sinZ + y1 * cosZ

        // Smooth wave distortion with gentle pulsing effect
        const pulse = Math.sin(time * 0.0012) * 0.1 + 1
        const wave1 = Math.sin(time * 0.0015 + point.originalX * 4 + point.originalY * 4) * 0.08
        const wave2 = Math.cos(time * 0.0012 + point.originalZ * 3) * 0.05
        
        // Cursor interaction - protrude toward cursor using 3D direction
        // Use the screen-projected direction (x2, y2) which represents the point's direction on screen
        const screenDirLength = Math.sqrt(x2 * x2 + y2 * y2)
        const normalizedScreenX = screenDirLength > 0 ? x2 / screenDirLength : 0
        const normalizedScreenY = screenDirLength > 0 ? y2 / screenDirLength : 0
        
        // Dot product to determine how aligned the point is with cursor direction
        const alignment = normalizedScreenX * cursorDirX + normalizedScreenY * cursorDirY
        // Protrusion strength decreases with distance from cursor - slightly increased
        // Increased distance range (900) and strength (0.04) for slightly more visible effect
        const distanceFactor = 1 - Math.min(cursorDistance / 900, 1)
        // Use smoother easing function for more gradual effect
        const smoothAlignment = Math.max(0, alignment)
        const easedAlignment = smoothAlignment * smoothAlignment // Quadratic easing
        const cursorProtrusion = easedAlignment * 0.04 * distanceFactor
        
        const scale = (1 + wave1 + wave2 + cursorProtrusion) * pulse

        return {
          x: x2 * radius * scale,
          y: y2 * radius * scale,
          z: z2 * radius * scale,
          screenX: centerX + x2 * radius * scale,
          screenY: centerY + y2 * radius * scale,
          depth: z2,
          originalX: point.originalX,
          originalY: point.originalY,
          originalZ: point.originalZ
        }
      })

      // Sort by depth for proper rendering
      projectedPoints.sort((a, b) => b.depth - a.depth)

      // Color cycling: blue → greenish → white → blue
      // Cycle duration: 8 seconds
      const colorCycle = (time * 0.0002) % 1
      let r, g, b
      
      if (colorCycle < 0.33) {
        // Blue to Greenish (0 to 0.33)
        const t = colorCycle / 0.33
        r = 70 + (100 - 70) * t
        g = 130 + (200 - 130) * t
        b = 255 - (255 - 150) * t
      } else if (colorCycle < 0.66) {
        // Greenish to White (0.33 to 0.66)
        const t = (colorCycle - 0.33) / 0.33
        r = 100 + (255 - 100) * t
        g = 200 + (255 - 200) * t
        b = 150 + (255 - 150) * t
      } else {
        // White to Blue (0.66 to 1.0)
        const t = (colorCycle - 0.66) / 0.34
        r = 255 - (255 - 70) * t
        g = 255 - (255 - 130) * t
        b = 255 - (255 - 255) * t
      }

      // Draw enhanced glow effect behind the sphere with color cycling
      const glowGradient1 = ctx.createRadialGradient(
        centerX,
        centerY,
        0,
        centerX,
        centerY,
        radius * 2.5
      )
      glowGradient1.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.15)`)
      glowGradient1.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, 0.08)`)
      glowGradient1.addColorStop(0.7, `rgba(${r}, ${g}, ${b}, 0.03)`)
      glowGradient1.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)
      
      ctx.fillStyle = glowGradient1
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius * 2.5, 0, Math.PI * 2)
      ctx.fill()
      
      // Draw subtle background color for the sphere (inner glow) with color cycling
      const bgGradient = ctx.createRadialGradient(
        centerX,
        centerY,
        0,
        centerX,
        centerY,
        radius * 1.8
      )
      bgGradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.12)`)
      bgGradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, 0.06)`)
      bgGradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)
      
      ctx.fillStyle = bgGradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius * 1.8, 0, Math.PI * 2)
      ctx.fill()

      // Draw connections between nearby points with smooth animated opacity and color cycling
      const connectionOpacity = 0.15 + Math.sin(time * 0.002) * 0.1
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${connectionOpacity})`
      ctx.lineWidth = 0.5
      for (let i = 0; i < projectedPoints.length; i++) {
        for (let j = i + 1; j < projectedPoints.length; j++) {
          const dx = projectedPoints[i].screenX - projectedPoints[j].screenX
          const dy = projectedPoints[i].screenY - projectedPoints[j].screenY
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance < 100) {
            ctx.beginPath()
            ctx.moveTo(projectedPoints[i].screenX, projectedPoints[i].screenY)
            ctx.lineTo(projectedPoints[j].screenX, projectedPoints[j].screenY)
            ctx.stroke()
          }
        }
      }

      // Draw glowing dots with minimal bloom
      const pulseIntensity = Math.sin(time * 0.0015) * 0.2 + 1
      projectedPoints.forEach(point => {
        const baseGlowSize = 1.5 + (1 + point.depth / radius) * 1
        const glowSize = baseGlowSize * pulseIntensity
        const baseAlpha = 0.35 + (1 + point.depth / radius) * 0.1
        const alpha = baseAlpha * (0.8 + Math.sin(time * 0.0024 + point.originalX * 2) * 0.2)

        // Outer glow - minimal bloom effect with color cycling
        const gradient = ctx.createRadialGradient(
          point.screenX,
          point.screenY,
          0,
          point.screenX,
          point.screenY,
          glowSize * 1.5
        )
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${alpha * 0.6})`)
        gradient.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, ${alpha * 0.2})`)
        gradient.addColorStop(0.7, `rgba(${r}, ${g}, ${b}, ${alpha * 0.05})`)
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(point.screenX, point.screenY, glowSize * 1.5, 0, Math.PI * 2)
        ctx.fill()

        // Core dot - color cycling with pulsing
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`
        ctx.beginPath()
        ctx.arc(point.screenX, point.screenY, glowSize, 0, Math.PI * 2)
        ctx.fill()
      })

      time += 10
      requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', setCanvasSize)
      canvas.removeEventListener('mousemove', handleMouseMove)
    }
  }, [])

  // Chat page sphere - smaller version in top-left
  useEffect(() => {
    if (currentPage !== 'chat') return
    
    const canvas = chatSphereRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size for small sphere
    const setCanvasSize = () => {
      canvas.width = 120
      canvas.height = 120
    }
    setCanvasSize()

    // Sphere parameters - smaller for chat page
    const radius = 40
    const numPoints = 200
    const points: Array<{ x: number; y: number; z: number; originalX: number; originalY: number; originalZ: number }> = []

    // Generate points on sphere surface
    const goldenAngle = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < numPoints; i++) {
      const y = 1 - (i / (numPoints - 1)) * 2
      const radiusAtY = Math.sqrt(1 - y * y)
      const theta = goldenAngle * i
      const x = Math.cos(theta) * radiusAtY
      const z = Math.sin(theta) * radiusAtY
      
      points.push({
        x: x * radius,
        y: y * radius,
        z: z * radius,
        originalX: x,
        originalY: y,
        originalZ: z
      })
    }

    let time = 0
    const animate = () => {
      if (currentPage !== 'chat') return
      
      ctx.fillStyle = '#242121'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      const centerX = canvas.width / 2
      const centerY = canvas.height / 2

      // Smooth and slow rotations
      const rotationX = Math.sin(time * 0.0008) * 0.5 + Math.cos(time * 0.0006) * 0.2
      const rotationY = time * 0.0004
      const rotationZ = Math.sin(time * 0.0006) * 0.3

      const projectedPoints = points.map(point => {
        let x = point.originalX
        let y = point.originalY
        let z = point.originalZ

        const cosY = Math.cos(rotationY)
        const sinY = Math.sin(rotationY)
        const x1 = x * cosY - z * sinY
        const z1 = x * sinY + z * cosY

        const cosX = Math.cos(rotationX)
        const sinX = Math.sin(rotationX)
        const y1 = y * cosX - z1 * sinX
        const z2 = y * sinX + z1 * cosX

        const cosZ = Math.cos(rotationZ)
        const sinZ = Math.sin(rotationZ)
        const x2 = x1 * cosZ - y1 * sinZ
        const y2 = x1 * sinZ + y1 * cosZ

        const pulse = Math.sin(time * 0.0012) * 0.1 + 1
        const wave1 = Math.sin(time * 0.0015 + point.originalX * 4 + point.originalY * 4) * 0.08
        const wave2 = Math.cos(time * 0.0012 + point.originalZ * 3) * 0.05
        const scale = (1 + wave1 + wave2) * pulse

        return {
          x: x2 * radius * scale,
          y: y2 * radius * scale,
          z: z2 * radius * scale,
          screenX: centerX + x2 * radius * scale,
          screenY: centerY + y2 * radius * scale,
          depth: z2,
          originalX: point.originalX,
          originalY: point.originalY,
          originalZ: point.originalZ
        }
      })

      projectedPoints.sort((a, b) => b.depth - a.depth)

      // Color cycling: blue → greenish → white → blue
      const colorCycle = (time * 0.0002) % 1
      let r, g, b
      
      if (colorCycle < 0.33) {
        const t = colorCycle / 0.33
        r = 70 + (100 - 70) * t
        g = 130 + (200 - 130) * t
        b = 255 - (255 - 150) * t
      } else if (colorCycle < 0.66) {
        const t = (colorCycle - 0.33) / 0.33
        r = 100 + (255 - 100) * t
        g = 200 + (255 - 200) * t
        b = 150 + (255 - 150) * t
      } else {
        const t = (colorCycle - 0.66) / 0.34
        r = 255 - (255 - 70) * t
        g = 255 - (255 - 130) * t
        b = 255 - (255 - 255) * t
      }

      // Draw connections
      const connectionOpacity = 0.15 + Math.sin(time * 0.002) * 0.1
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${connectionOpacity})`
      ctx.lineWidth = 0.5
      for (let i = 0; i < projectedPoints.length; i++) {
        for (let j = i + 1; j < projectedPoints.length; j++) {
          const dx = projectedPoints[i].screenX - projectedPoints[j].screenX
          const dy = projectedPoints[i].screenY - projectedPoints[j].screenY
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance < 20) {
            ctx.beginPath()
            ctx.moveTo(projectedPoints[i].screenX, projectedPoints[i].screenY)
            ctx.lineTo(projectedPoints[j].screenX, projectedPoints[j].screenY)
            ctx.stroke()
          }
        }
      }

      // Draw glowing dots
      const pulseIntensity = Math.sin(time * 0.0015) * 0.2 + 1
      projectedPoints.forEach(point => {
        const baseGlowSize = 0.8 + (1 + point.depth / radius) * 0.5
        const glowSize = baseGlowSize * pulseIntensity
        const baseAlpha = 0.35 + (1 + point.depth / radius) * 0.1
        const alpha = baseAlpha * (0.8 + Math.sin(time * 0.0024 + point.originalX * 2) * 0.2)

        const gradient = ctx.createRadialGradient(
          point.screenX,
          point.screenY,
          0,
          point.screenX,
          point.screenY,
          glowSize * 1.5
        )
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${alpha * 0.6})`)
        gradient.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, ${alpha * 0.2})`)
        gradient.addColorStop(0.7, `rgba(${r}, ${g}, ${b}, ${alpha * 0.05})`)
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(point.screenX, point.screenY, glowSize * 1.5, 0, Math.PI * 2)
        ctx.fill()

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`
        ctx.beginPath()
        ctx.arc(point.screenX, point.screenY, glowSize, 0, Math.PI * 2)
        ctx.fill()
      })

      time += 10
      requestAnimationFrame(animate)
    }

    animate()

    return () => {
      // Cleanup if needed
    }
  }, [currentPage])

  const handleGenerate = async () => {
    if (!inputText.trim()) {
      setError('Please enter a description')
      return
    }

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
          text: inputText,
          temperature: 1.0,
          top_k: 20
        })
      })

      if (!response.ok) {
        throw new Error('Failed to generate music')
      }

      const data: GenerationResult = await response.json()
      setGenerationResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
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
      handleGenerate()
    }
  }

  if (currentPage === 'chat') {
    return (
      <div className="chat-page">
        <div className="chat-logo">
          <canvas ref={chatSphereRef} className="chat-sphere-canvas"></canvas>
        </div>
        
        <div className="chat-content">
          <h1 className="chat-title">HOW MAY I HELP YOU?</h1>
          
          <div className="chat-input-container">
            <input 
              type="text" 
              className="chat-input" 
              placeholder="WHAT DO YOU WANT TO LISTEN ?"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isGenerating}
            />
            <button 
              className="chat-submit-btn"
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              {isGenerating ? (
                <div className="spinner"></div>
              ) : (
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </button>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {generationResult && (
            <div className="result-container">
              <h2 className="result-title">Music Generated!</h2>
              <div className="result-details">
                <p><strong>Emotion:</strong> {generationResult.emotion}</p>
                <p><strong>Duration:</strong> {generationResult.duration} minutes</p>
                <p><strong>Tokens:</strong> {generationResult.tokens_generated}</p>
              </div>
              <button className="download-btn" onClick={handleDownload}>
                Download MIDI
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="landing-page">
      <nav className="nav">
        <a href="#" className="nav-link">HOME</a>
        <a href="#" className="nav-link">ABOUT US</a>
        <a href="#" className="nav-link">CONTACT</a>
      </nav>
      
      <canvas ref={canvasRef} className="sphere-canvas"></canvas>
      
      <div className="content">
        <h1 className="tagline">LET AI CREATE WHAT YOU NEED</h1>
        <a href="#" className="cta-link" onClick={(e) => { e.preventDefault(); setCurrentPage('chat'); }}>GET STARTED</a>
      </div>
    </div>
  )
}

export default App
