'use client'

import { useState, useMemo } from 'react'
import { ChevronUp, ChevronDown, Search } from 'lucide-react'
import { motion } from 'framer-motion'

interface Column {
  key: string
  label: string
  sortable?: boolean
  render?: (value: any, row: any) => React.ReactNode
}

interface DataTableProps {
  columns: Column[]
  data: any[]
  searchable?: boolean
  searchPlaceholder?: string
  itemsPerPage?: number
}

export default function DataTable({ 
  columns, 
  data, 
  searchable = true,
  searchPlaceholder = 'Search...',
  itemsPerPage = 10 
}: DataTableProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null)
  const [currentPage, setCurrentPage] = useState(1)

  const handleSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc'
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc'
    }
    setSortConfig({ key, direction })
  }

  const filteredAndSortedData = useMemo(() => {
    let processedData = [...data]

    // Filter
    if (searchTerm) {
      processedData = processedData.filter(row =>
        Object.values(row).some(value =>
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
    }

    // Sort
    if (sortConfig) {
      processedData.sort((a, b) => {
        const aValue = a[sortConfig.key]
        const bValue = b[sortConfig.key]
        
        if (aValue === null || aValue === undefined) return 1
        if (bValue === null || bValue === undefined) return -1
        
        if (aValue < bValue) {
          return sortConfig.direction === 'asc' ? -1 : 1
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'asc' ? 1 : -1
        }
        return 0
      })
    }

    return processedData
  }, [data, searchTerm, sortConfig])

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedData.length / itemsPerPage)
  const paginatedData = filteredAndSortedData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  )

  return (
    <div className="space-y-4">
      {searchable && (
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
          <input
            type="text"
            placeholder={searchPlaceholder}
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value)
              setCurrentPage(1)
            }}
            className="w-full pl-10 pr-4 py-2 bg-card-hover text-text-primary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10">
              {columns.map((column) => (
                <th
                  key={column.key}
                  className={`text-left py-3 px-4 text-sm font-medium text-text-secondary ${
                    column.sortable ? 'cursor-pointer hover:text-text-primary' : ''
                  }`}
                  onClick={() => column.sortable && handleSort(column.key)}
                >
                  <div className="flex items-center gap-1">
                    {column.label}
                    {column.sortable && (
                      <div className="flex flex-col">
                        <ChevronUp className={`w-3 h-3 -mb-1 ${
                          sortConfig?.key === column.key && sortConfig.direction === 'asc'
                            ? 'text-primary'
                            : 'text-text-secondary/30'
                        }`} />
                        <ChevronDown className={`w-3 h-3 -mt-1 ${
                          sortConfig?.key === column.key && sortConfig.direction === 'desc'
                            ? 'text-primary'
                            : 'text-text-secondary/30'
                        }`} />
                      </div>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, rowIndex) => (
              <motion.tr
                key={rowIndex}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: rowIndex * 0.05 }}
                className="border-b border-white/5 hover:bg-card-hover transition-colors"
              >
                {columns.map((column) => (
                  <td key={column.key} className="py-3 px-4 text-sm text-text-primary">
                    {column.render ? column.render(row[column.key], row) : row[column.key]}
                  </td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-text-secondary">
            Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredAndSortedData.length)} of {filteredAndSortedData.length} results
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-card-hover text-text-primary rounded-lg text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-card transition-colors"
            >
              Previous
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum
              if (totalPages <= 5) {
                pageNum = i + 1
              } else if (currentPage <= 3) {
                pageNum = i + 1
              } else if (currentPage >= totalPages - 2) {
                pageNum = totalPages - 4 + i
              } else {
                pageNum = currentPage - 2 + i
              }
              
              return (
                <button
                  key={i}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                    currentPage === pageNum
                      ? 'bg-primary text-white'
                      : 'bg-card-hover text-text-primary hover:bg-card'
                  }`}
                >
                  {pageNum}
                </button>
              )
            })}
            <button
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-card-hover text-text-primary rounded-lg text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-card transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}