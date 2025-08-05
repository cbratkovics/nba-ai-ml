'use client'

import { ArrowLeft } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'

interface BackButtonProps {
  label?: string
  href?: string
}

export default function BackButton({ label = 'Back', href }: BackButtonProps) {
  const router = useRouter()
  
  const handleClick = () => {
    if (href) {
      router.push(href)
    } else {
      router.back()
    }
  }
  
  return (
    <motion.button
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      whileHover={{ x: -5 }}
      onClick={handleClick}
      className="flex items-center gap-2 text-text-secondary hover:text-text-primary transition-colors mb-6"
    >
      <ArrowLeft className="w-5 h-5" />
      <span className="text-sm font-medium">{label}</span>
    </motion.button>
  )
}