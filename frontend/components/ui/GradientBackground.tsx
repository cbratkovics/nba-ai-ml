'use client'

import { useEffect, useRef } from 'react'

export default function GradientBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    let animationId: number
    let time = 0

    const gradient1 = { x: 0, y: 0 }
    const gradient2 = { x: canvas.width, y: canvas.height }

    const animate = () => {
      time += 0.001

      // Clear canvas
      ctx.fillStyle = '#0a0a0f'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Animate gradient positions
      gradient1.x = Math.sin(time) * canvas.width * 0.5 + canvas.width * 0.5
      gradient1.y = Math.cos(time * 0.8) * canvas.height * 0.5 + canvas.height * 0.5
      gradient2.x = Math.cos(time * 1.2) * canvas.width * 0.5 + canvas.width * 0.5
      gradient2.y = Math.sin(time * 0.9) * canvas.height * 0.5 + canvas.height * 0.5

      // Create radial gradients
      const grad1 = ctx.createRadialGradient(
        gradient1.x, gradient1.y, 0,
        gradient1.x, gradient1.y, canvas.width * 0.4
      )
      grad1.addColorStop(0, 'rgba(139, 92, 246, 0.3)')
      grad1.addColorStop(1, 'rgba(139, 92, 246, 0)')

      const grad2 = ctx.createRadialGradient(
        gradient2.x, gradient2.y, 0,
        gradient2.x, gradient2.y, canvas.width * 0.4
      )
      grad2.addColorStop(0, 'rgba(0, 212, 255, 0.3)')
      grad2.addColorStop(1, 'rgba(0, 212, 255, 0)')

      // Draw gradients
      ctx.fillStyle = grad1
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = grad2
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      animationId = requestAnimationFrame(animate)
    }

    animate()

    const handleResize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10 opacity-50"
      style={{ filter: 'blur(100px)' }}
    />
  )
}