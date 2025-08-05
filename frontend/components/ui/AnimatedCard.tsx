'use client'

import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface AnimatedCardProps {
  children: React.ReactNode
  className?: string
  delay?: number
  hover?: boolean
}

export default function AnimatedCard({ 
  children, 
  className, 
  delay = 0,
  hover = true 
}: AnimatedCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={hover ? { scale: 1.02 } : undefined}
      className={cn(
        'glass-card p-6',
        hover && 'hover:shadow-primary/20 hover:shadow-xl',
        'transition-all duration-300',
        className
      )}
    >
      {children}
    </motion.div>
  )
}