'use client'

import { useState } from 'react'
import { Copy, Check } from 'lucide-react'
import { motion } from 'framer-motion'

interface CodeBlockProps {
  code: string
  language?: string
  showLineNumbers?: boolean
}

export default function CodeBlock({ code, language = 'javascript', showLineNumbers = true }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const lines = code.split('\n')

  const getLanguageColor = () => {
    switch (language) {
      case 'python': return 'text-blue-400'
      case 'javascript': return 'text-yellow-400'
      case 'typescript': return 'text-blue-500'
      case 'bash': return 'text-green-400'
      case 'json': return 'text-orange-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="relative group">
      <div className="absolute top-2 right-2 flex items-center gap-2">
        <span className={`text-xs font-medium ${getLanguageColor()}`}>
          {language}
        </span>
        <button
          onClick={copyToClipboard}
          className="p-2 bg-card-hover rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-card"
        >
          {copied ? (
            <Check className="w-4 h-4 text-success" />
          ) : (
            <Copy className="w-4 h-4 text-text-secondary" />
          )}
        </button>
      </div>
      
      <div className="bg-card-hover rounded-lg p-4 overflow-x-auto">
        <pre className="text-sm">
          <code className="text-text-primary">
            {showLineNumbers ? (
              <table className="w-full">
                <tbody>
                  {lines.map((line, index) => (
                    <tr key={index}>
                      <td className="text-text-secondary text-xs pr-4 select-none">{index + 1}</td>
                      <td className="w-full">
                        <span dangerouslySetInnerHTML={{ __html: highlightSyntax(line, language) }} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div dangerouslySetInnerHTML={{ __html: highlightSyntax(code, language) }} />
            )}
          </code>
        </pre>
      </div>
    </div>
  )
}

function highlightSyntax(code: string, language: string): string {
  // Basic syntax highlighting (can be enhanced with a proper library)
  let highlighted = code
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  if (language === 'python') {
    highlighted = highlighted
      .replace(/\b(def|class|import|from|return|if|else|elif|for|while|try|except|with|as|True|False|None)\b/g, '<span class="text-purple-400">$1</span>')
      .replace(/(['"])([^'"]*)\1/g, '<span class="text-green-400">$1$2$1</span>')
      .replace(/#.*/g, '<span class="text-gray-500">$&</span>')
  } else if (language === 'javascript' || language === 'typescript') {
    highlighted = highlighted
      .replace(/\b(const|let|var|function|return|if|else|for|while|try|catch|class|extends|import|export|from|async|await|true|false|null|undefined)\b/g, '<span class="text-purple-400">$1</span>')
      .replace(/(['"])([^'"]*)\1/g, '<span class="text-green-400">$1$2$1</span>')
      .replace(/\/\/.*/g, '<span class="text-gray-500">$&</span>')
  } else if (language === 'bash') {
    highlighted = highlighted
      .replace(/\b(echo|cd|ls|mkdir|rm|cp|mv|cat|grep|sed|awk|curl|wget|git|npm|pip)\b/g, '<span class="text-purple-400">$1</span>')
      .replace(/(['"])([^'"]*)\1/g, '<span class="text-green-400">$1$2$1</span>')
      .replace(/#.*/g, '<span class="text-gray-500">$&</span>')
  } else if (language === 'json') {
    highlighted = highlighted
      .replace(/("[^"]*"):/g, '<span class="text-blue-400">$1</span>:')
      .replace(/:\s*("[^"]*")/g, ': <span class="text-green-400">$1</span>')
      .replace(/:\s*(\d+)/g, ': <span class="text-orange-400">$1</span>')
      .replace(/:\s*(true|false|null)/g, ': <span class="text-purple-400">$1</span>')
  }

  return highlighted
}