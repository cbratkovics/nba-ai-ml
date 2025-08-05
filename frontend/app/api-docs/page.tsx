'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import AnimatedCard from '@/components/ui/AnimatedCard'
import { 
  Code, Copy, Check, ChevronRight, ChevronDown, 
  Play, Book, Key, Clock, Shield, Download,
  Globe, Terminal, FileCode, Package, Zap,
  AlertCircle, CheckCircle, XCircle, Info,
  ExternalLink, Github, Coffee, Database
} from 'lucide-react'

export default function ApiDocsPage() {
  const [selectedEndpoint, setSelectedEndpoint] = useState('predictions-get')
  const [activeTab, setActiveTab] = useState('overview')
  const [expandedCategory, setExpandedCategory] = useState('predictions')
  const [selectedLanguage, setSelectedLanguage] = useState('python')
  const [tryItOutValues, setTryItOutValues] = useState<any>({})
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [apiResponse, setApiResponse] = useState<any>(null)
  const [isExecuting, setIsExecuting] = useState(false)

  // API Status
  const apiStatus = {
    status: 'operational',
    latency: 87,
    uptime: 99.98,
    version: '2.0.3'
  }

  // Endpoint categories
  const endpointCategories = {
    predictions: {
      name: 'Predictions',
      description: 'ML model predictions for NBA statistics',
      endpoints: [
        {
          id: 'predictions-get',
          method: 'GET',
          path: '/api/predictions/{player_id}',
          description: 'Get predictions for a specific player',
          rateLimit: '1000/hour'
        },
        {
          id: 'predictions-post',
          method: 'POST',
          path: '/api/predictions/batch',
          description: 'Batch predictions for multiple players',
          rateLimit: '100/hour'
        }
      ]
    },
    players: {
      name: 'Players',
      description: 'Player data and statistics management',
      endpoints: [
        {
          id: 'players-list',
          method: 'GET',
          path: '/api/players',
          description: 'List all players with filtering',
          rateLimit: '5000/hour'
        },
        {
          id: 'players-get',
          method: 'GET',
          path: '/api/players/{id}',
          description: 'Get detailed player information',
          rateLimit: '5000/hour'
        },
        {
          id: 'players-update',
          method: 'PUT',
          path: '/api/players/{id}',
          description: 'Update player information',
          rateLimit: '500/hour'
        },
        {
          id: 'players-delete',
          method: 'DELETE',
          path: '/api/players/{id}',
          description: 'Delete a player record',
          rateLimit: '100/hour'
        }
      ]
    },
    experiments: {
      name: 'Experiments',
      description: 'A/B testing and experimentation endpoints',
      endpoints: [
        {
          id: 'experiments-list',
          method: 'GET',
          path: '/api/experiments',
          description: 'List all active experiments',
          rateLimit: '1000/hour'
        },
        {
          id: 'experiments-create',
          method: 'POST',
          path: '/api/experiments',
          description: 'Create a new experiment',
          rateLimit: '50/hour'
        },
        {
          id: 'experiments-results',
          method: 'GET',
          path: '/api/experiments/{id}/results',
          description: 'Get experiment results and metrics',
          rateLimit: '1000/hour'
        }
      ]
    },
    analytics: {
      name: 'Analytics',
      description: 'Usage analytics and metrics',
      endpoints: [
        {
          id: 'analytics-usage',
          method: 'GET',
          path: '/api/analytics/usage',
          description: 'Get API usage statistics',
          rateLimit: '100/hour'
        },
        {
          id: 'analytics-performance',
          method: 'GET',
          path: '/api/analytics/performance',
          description: 'Get performance metrics',
          rateLimit: '100/hour'
        }
      ]
    }
  }

  // Endpoint details
  const endpointDetails: { [key: string]: any } = {
    'predictions-get': {
      parameters: [
        { name: 'player_id', type: 'string', required: true, description: 'Unique identifier of the player', example: 'lebron-james-2544' },
        { name: 'date', type: 'string', required: false, description: 'Game date in YYYY-MM-DD format', example: '2025-01-20' },
        { name: 'opponent', type: 'string', required: false, description: 'Opponent team code', example: 'LAL' },
        { name: 'metrics', type: 'array', required: false, description: 'Specific metrics to predict', example: '["points", "assists", "rebounds"]' }
      ],
      responses: {
        200: {
          description: 'Successful prediction',
          schema: {
            player_id: 'string',
            predictions: {
              points: 'number',
              assists: 'number',
              rebounds: 'number',
              confidence: 'number'
            },
            model_version: 'string',
            timestamp: 'string'
          },
          example: {
            player_id: 'lebron-james-2544',
            predictions: {
              points: 27.3,
              assists: 8.2,
              rebounds: 7.8,
              confidence: 0.94
            },
            model_version: 'v2.3.1',
            timestamp: '2025-01-20T10:30:00Z'
          }
        },
        400: {
          description: 'Bad request - Invalid parameters',
          schema: {
            error: 'string',
            message: 'string',
            details: 'object'
          }
        },
        404: {
          description: 'Player not found',
          schema: {
            error: 'string',
            message: 'string'
          }
        },
        429: {
          description: 'Rate limit exceeded',
          schema: {
            error: 'string',
            message: 'string',
            retry_after: 'number'
          }
        }
      }
    },
    'predictions-post': {
      parameters: [
        { 
          name: 'players', 
          type: 'array', 
          required: true, 
          description: 'Array of player objects for batch prediction',
          example: '[{"player_id": "player1", "date": "2025-01-20"}]'
        },
        { name: 'parallel', type: 'boolean', required: false, description: 'Process predictions in parallel', example: 'true' },
        { name: 'include_confidence', type: 'boolean', required: false, description: 'Include confidence scores', example: 'true' }
      ],
      responses: {
        200: {
          description: 'Successful batch prediction',
          schema: {
            results: 'array',
            processed: 'number',
            failed: 'number',
            execution_time: 'number'
          },
          example: {
            results: [
              {
                player_id: 'player1',
                predictions: { points: 25.4, assists: 6.7, rebounds: 5.2 },
                confidence: 0.92
              },
              {
                player_id: 'player2',
                predictions: { points: 18.9, assists: 4.3, rebounds: 8.1 },
                confidence: 0.89
              }
            ],
            processed: 2,
            failed: 0,
            execution_time: 234
          }
        }
      }
    }
  }

  // Code examples
  const codeExamples: { [key: string]: { [key: string]: string } } = {
    'predictions-get': {
      python: `import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.nba-ml.com/v2"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(
    f"{BASE_URL}/predictions/lebron-james-2544",
    headers=headers,
    params={
        "date": "2025-01-20",
        "opponent": "LAL",
        "metrics": ["points", "assists", "rebounds"]
    }
)

if response.status_code == 200:
    prediction = response.json()
    print(f"Predicted points: {prediction['predictions']['points']}")
else:
    print(f"Error: {response.status_code}")`,
      javascript: `const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://api.nba-ml.com/v2';

const headers = {
  'Authorization': \`Bearer \${API_KEY}\`,
  'Content-Type': 'application/json'
};

const params = new URLSearchParams({
  date: '2025-01-20',
  opponent: 'LAL',
  metrics: JSON.stringify(['points', 'assists', 'rebounds'])
});

fetch(\`\${BASE_URL}/predictions/lebron-james-2544?\${params}\`, {
  method: 'GET',
  headers: headers
})
  .then(response => response.json())
  .then(data => {
    console.log(\`Predicted points: \${data.predictions.points}\`);
  })
  .catch(error => console.error('Error:', error));`,
      curl: `curl -X GET "https://api.nba-ml.com/v2/predictions/lebron-james-2544?date=2025-01-20&opponent=LAL&metrics=points,assists,rebounds" \\
  -H "Authorization: Bearer your_api_key_here" \\
  -H "Content-Type: application/json"`
    },
    'predictions-post': {
      python: `import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.nba-ml.com/v2"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "players": [
        {"player_id": "lebron-james-2544", "date": "2025-01-20"},
        {"player_id": "stephen-curry-201939", "date": "2025-01-20"}
    ],
    "parallel": True,
    "include_confidence": True
}

response = requests.post(
    f"{BASE_URL}/predictions/batch",
    headers=headers,
    json=data
)

if response.status_code == 200:
    results = response.json()
    for result in results['results']:
        print(f"{result['player_id']}: {result['predictions']['points']} pts")
else:
    print(f"Error: {response.status_code}")`,
      javascript: `const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://api.nba-ml.com/v2';

const headers = {
  'Authorization': \`Bearer \${API_KEY}\`,
  'Content-Type': 'application/json'
};

const data = {
  players: [
    { player_id: 'lebron-james-2544', date: '2025-01-20' },
    { player_id: 'stephen-curry-201939', date: '2025-01-20' }
  ],
  parallel: true,
  include_confidence: true
};

fetch(\`\${BASE_URL}/predictions/batch\`, {
  method: 'POST',
  headers: headers,
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(results => {
    results.results.forEach(result => {
      console.log(\`\${result.player_id}: \${result.predictions.points} pts\`);
    });
  })
  .catch(error => console.error('Error:', error));`,
      curl: `curl -X POST "https://api.nba-ml.com/v2/predictions/batch" \\
  -H "Authorization: Bearer your_api_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "players": [
      {"player_id": "lebron-james-2544", "date": "2025-01-20"},
      {"player_id": "stephen-curry-201939", "date": "2025-01-20"}
    ],
    "parallel": true,
    "include_confidence": true
  }'`
    }
  }

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const executeRequest = async () => {
    setIsExecuting(true)
    // Simulate API call
    setTimeout(() => {
      setApiResponse({
        status: 200,
        data: endpointDetails[selectedEndpoint]?.responses[200]?.example || {},
        headers: {
          'Content-Type': 'application/json',
          'X-RateLimit-Remaining': '999',
          'X-RateLimit-Reset': '1705750800'
        },
        time: 87
      })
      setIsExecuting(false)
    }, 1500)
  }

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'POST': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'PUT': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'DELETE': return 'bg-red-500/20 text-red-400 border-red-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-text-primary mb-2">API Documentation v2.0</h1>
              <p className="text-text-secondary">Complete reference for NBA ML Platform API</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-success/20 text-success rounded-lg text-sm">
                <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                Operational
              </div>
              <span className="text-sm text-text-secondary">v{apiStatus.version}</span>
            </div>
          </div>

          {/* Quick Links */}
          <div className="flex gap-4 mt-4">
            <button className="text-sm text-primary hover:text-primary-hover flex items-center gap-1">
              <Book className="w-4 h-4" />
              Getting Started
            </button>
            <button className="text-sm text-primary hover:text-primary-hover flex items-center gap-1">
              <Key className="w-4 h-4" />
              Authentication
            </button>
            <button className="text-sm text-primary hover:text-primary-hover flex items-center gap-1">
              <Clock className="w-4 h-4" />
              Rate Limits
            </button>
            <button className="text-sm text-primary hover:text-primary-hover flex items-center gap-1">
              <Package className="w-4 h-4" />
              SDKs
            </button>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar - Endpoint Categories */}
          <div className="lg:col-span-1">
            <AnimatedCard delay={0.1} className="p-4 sticky top-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">API Endpoints</h3>
              <div className="space-y-2">
                {Object.entries(endpointCategories).map(([key, category]) => (
                  <div key={key}>
                    <button
                      onClick={() => setExpandedCategory(expandedCategory === key ? '' : key)}
                      className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-card-hover transition-colors"
                    >
                      <span className="text-sm text-text-primary font-medium">{category.name}</span>
                      {expandedCategory === key ? (
                        <ChevronDown className="w-4 h-4 text-text-secondary" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-text-secondary" />
                      )}
                    </button>
                    {expandedCategory === key && (
                      <div className="ml-2 mt-1 space-y-1">
                        {category.endpoints.map((endpoint) => (
                          <button
                            key={endpoint.id}
                            onClick={() => setSelectedEndpoint(endpoint.id)}
                            className={`w-full text-left p-2 rounded-lg text-xs transition-colors ${
                              selectedEndpoint === endpoint.id
                                ? 'bg-primary/20 text-primary'
                                : 'text-text-secondary hover:text-text-primary hover:bg-card-hover'
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${
                                getMethodColor(endpoint.method)
                              }`}>
                                {endpoint.method}
                              </span>
                              <span className="truncate">{endpoint.path.split('/').pop()}</span>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </AnimatedCard>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Endpoint Details */}
            {(() => {
              const currentEndpoint = Object.values(endpointCategories)
                .flatMap(cat => cat.endpoints)
                .find(ep => ep.id === selectedEndpoint)
              
              if (!currentEndpoint) return null

              return (
                <AnimatedCard delay={0.2} className="p-6">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`px-3 py-1.5 rounded-lg text-sm font-medium border ${
                          getMethodColor(currentEndpoint.method)
                        }`}>
                          {currentEndpoint.method}
                        </span>
                        <code className="text-lg font-mono text-text-primary">{currentEndpoint.path}</code>
                      </div>
                      <p className="text-text-secondary">{currentEndpoint.description}</p>
                      <div className="flex items-center gap-4 mt-3 text-sm">
                        <span className="flex items-center gap-1 text-text-secondary">
                          <Clock className="w-3 h-3" />
                          Rate Limit: {currentEndpoint.rateLimit}
                        </span>
                        <span className="flex items-center gap-1 text-text-secondary">
                          <Shield className="w-3 h-3" />
                          Requires Authentication
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => copyToClipboard(currentEndpoint.path, 'endpoint')}
                      className="p-2 bg-card-hover rounded-lg hover:bg-card transition-colors"
                    >
                      {copiedCode === 'endpoint' ? (
                        <Check className="w-4 h-4 text-success" />
                      ) : (
                        <Copy className="w-4 h-4 text-text-secondary" />
                      )}
                    </button>
                  </div>

                  {/* Request Parameters */}
                  {endpointDetails[selectedEndpoint]?.parameters && (
                    <div className="mb-6">
                      <h3 className="text-sm font-semibold text-text-primary mb-3">Request Parameters</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-white/10">
                              <th className="text-left py-2 px-3 text-xs font-medium text-text-secondary">Parameter</th>
                              <th className="text-left py-2 px-3 text-xs font-medium text-text-secondary">Type</th>
                              <th className="text-left py-2 px-3 text-xs font-medium text-text-secondary">Required</th>
                              <th className="text-left py-2 px-3 text-xs font-medium text-text-secondary">Description</th>
                              <th className="text-left py-2 px-3 text-xs font-medium text-text-secondary">Example</th>
                            </tr>
                          </thead>
                          <tbody>
                            {endpointDetails[selectedEndpoint].parameters.map((param: any) => (
                              <tr key={param.name} className="border-b border-white/5">
                                <td className="py-2 px-3">
                                  <code className="text-xs text-primary">{param.name}</code>
                                </td>
                                <td className="py-2 px-3">
                                  <span className="text-xs text-text-secondary">{param.type}</span>
                                </td>
                                <td className="py-2 px-3">
                                  {param.required ? (
                                    <span className="text-xs text-danger">Required</span>
                                  ) : (
                                    <span className="text-xs text-text-secondary">Optional</span>
                                  )}
                                </td>
                                <td className="py-2 px-3">
                                  <span className="text-xs text-text-secondary">{param.description}</span>
                                </td>
                                <td className="py-2 px-3">
                                  <code className="text-xs text-text-secondary bg-card-hover px-2 py-1 rounded">
                                    {param.example}
                                  </code>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Try it out */}
                  <div className="mb-6">
                    <h3 className="text-sm font-semibold text-text-primary mb-3">Try it out</h3>
                    <div className="bg-card-hover rounded-lg p-4 space-y-3">
                      <div>
                        <label className="block text-xs text-text-secondary mb-1">API Key</label>
                        <input
                          type="password"
                          placeholder="Enter your API key"
                          className="w-full px-3 py-2 bg-background text-text-primary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        />
                      </div>
                      
                      {endpointDetails[selectedEndpoint]?.parameters?.map((param: any) => (
                        <div key={param.name}>
                          <label className="block text-xs text-text-secondary mb-1">
                            {param.name} {param.required && <span className="text-danger">*</span>}
                          </label>
                          <input
                            type="text"
                            placeholder={param.example}
                            value={tryItOutValues[param.name] || ''}
                            onChange={(e) => setTryItOutValues({ ...tryItOutValues, [param.name]: e.target.value })}
                            className="w-full px-3 py-2 bg-background text-text-primary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                          />
                        </div>
                      ))}

                      <button
                        onClick={executeRequest}
                        disabled={isExecuting}
                        className="w-full px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        {isExecuting ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Executing...
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            Execute Request
                          </>
                        )}
                      </button>
                    </div>

                    {/* Response */}
                    {apiResponse && (
                      <div className="mt-4 bg-card-hover rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              apiResponse.status === 200 ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
                            }`}>
                              {apiResponse.status} {apiResponse.status === 200 ? 'OK' : 'Error'}
                            </span>
                            <span className="text-xs text-text-secondary">Response time: {apiResponse.time}ms</span>
                          </div>
                          <button
                            onClick={() => copyToClipboard(JSON.stringify(apiResponse.data, null, 2), 'response')}
                            className="p-1.5 bg-background rounded hover:bg-card transition-colors"
                          >
                            {copiedCode === 'response' ? (
                              <Check className="w-3 h-3 text-success" />
                            ) : (
                              <Copy className="w-3 h-3 text-text-secondary" />
                            )}
                          </button>
                        </div>
                        <pre className="bg-background rounded-lg p-3 overflow-x-auto">
                          <code className="text-xs text-text-primary">
                            {JSON.stringify(apiResponse.data, null, 2)}
                          </code>
                        </pre>
                      </div>
                    )}
                  </div>
                </AnimatedCard>
              )
            })()}

            {/* Code Examples */}
            <AnimatedCard delay={0.3} className="p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Code Examples</h3>
              <div className="flex gap-2 mb-4">
                {['python', 'javascript', 'curl'].map((lang) => (
                  <button
                    key={lang}
                    onClick={() => setSelectedLanguage(lang)}
                    className={`px-4 py-2 rounded-lg text-sm capitalize transition-colors ${
                      selectedLanguage === lang
                        ? 'bg-primary text-white'
                        : 'bg-card-hover text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {lang === 'javascript' ? 'JavaScript' : lang === 'curl' ? 'cURL' : 'Python'}
                  </button>
                ))}
              </div>

              <div className="relative">
                <button
                  onClick={() => copyToClipboard(
                    codeExamples[selectedEndpoint]?.[selectedLanguage] || '',
                    `code-${selectedLanguage}`
                  )}
                  className="absolute top-3 right-3 p-2 bg-card-hover rounded-lg hover:bg-card transition-colors z-10"
                >
                  {copiedCode === `code-${selectedLanguage}` ? (
                    <Check className="w-4 h-4 text-success" />
                  ) : (
                    <Copy className="w-4 h-4 text-text-secondary" />
                  )}
                </button>
                <pre className="bg-card-hover rounded-lg p-4 overflow-x-auto">
                  <code className="text-sm text-text-primary">
                    {codeExamples[selectedEndpoint]?.[selectedLanguage] || '// Select an endpoint to view code examples'}
                  </code>
                </pre>
              </div>
            </AnimatedCard>

            {/* Response Schemas */}
            <AnimatedCard delay={0.4} className="p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Response Schemas</h3>
              <div className="space-y-4">
                {endpointDetails[selectedEndpoint]?.responses && 
                  Object.entries(endpointDetails[selectedEndpoint].responses).map(([status, response]: any) => (
                    <div key={status} className="bg-card-hover rounded-lg p-4">
                      <div className="flex items-center gap-3 mb-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          status === '200' ? 'bg-success/20 text-success' :
                          status === '400' ? 'bg-warning/20 text-warning' :
                          status === '404' ? 'bg-danger/20 text-danger' :
                          'bg-danger/20 text-danger'
                        }`}>
                          {status}
                        </span>
                        <span className="text-sm text-text-secondary">{response.description}</span>
                      </div>
                      {response.example && (
                        <pre className="bg-background rounded-lg p-3 overflow-x-auto">
                          <code className="text-xs text-text-primary">
                            {JSON.stringify(response.example, null, 2)}
                          </code>
                        </pre>
                      )}
                    </div>
                  ))
                }
              </div>
            </AnimatedCard>
          </div>
        </div>

        {/* Guides Section */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <AnimatedCard delay={0.5} className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/20 rounded-lg">
                <Key className="w-5 h-5 text-primary" />
              </div>
              <h3 className="text-sm font-semibold text-text-primary">Authentication</h3>
            </div>
            <p className="text-xs text-text-secondary mb-3">
              Learn how to authenticate your API requests using Bearer tokens.
            </p>
            <button className="text-xs text-primary hover:text-primary-hover flex items-center gap-1">
              Read guide <ChevronRight className="w-3 h-3" />
            </button>
          </AnimatedCard>

          <AnimatedCard delay={0.6} className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-secondary/20 rounded-lg">
                <Clock className="w-5 h-5 text-secondary" />
              </div>
              <h3 className="text-sm font-semibold text-text-primary">Rate Limiting</h3>
            </div>
            <p className="text-xs text-text-secondary mb-3">
              Understand rate limits and how to handle rate limit responses.
            </p>
            <button className="text-xs text-primary hover:text-primary-hover flex items-center gap-1">
              Read guide <ChevronRight className="w-3 h-3" />
            </button>
          </AnimatedCard>

          <AnimatedCard delay={0.7} className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-warning/20 rounded-lg">
                <AlertCircle className="w-5 h-5 text-warning" />
              </div>
              <h3 className="text-sm font-semibold text-text-primary">Error Handling</h3>
            </div>
            <p className="text-xs text-text-secondary mb-3">
              Best practices for handling API errors and implementing retries.
            </p>
            <button className="text-xs text-primary hover:text-primary-hover flex items-center gap-1">
              Read guide <ChevronRight className="w-3 h-3" />
            </button>
          </AnimatedCard>

          <AnimatedCard delay={0.8} className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-success/20 rounded-lg">
                <Zap className="w-5 h-5 text-success" />
              </div>
              <h3 className="text-sm font-semibold text-text-primary">Webhooks</h3>
            </div>
            <p className="text-xs text-text-secondary mb-3">
              Set up webhooks to receive real-time notifications for events.
            </p>
            <button className="text-xs text-primary hover:text-primary-hover flex items-center gap-1">
              Read guide <ChevronRight className="w-3 h-3" />
            </button>
          </AnimatedCard>
        </div>

        {/* SDK Downloads */}
        <AnimatedCard delay={0.9} className="p-6 mt-8">
          <h3 className="text-lg font-semibold text-text-primary mb-4">SDK Downloads</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-card-hover rounded-lg p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 bg-primary/20 rounded-lg">
                  <FileCode className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-text-primary">Python SDK</h4>
                  <p className="text-xs text-text-secondary">v2.3.1</p>
                </div>
              </div>
              <pre className="bg-background rounded p-2 mb-3">
                <code className="text-xs text-text-primary">pip install nba-ml-sdk</code>
              </pre>
              <div className="flex gap-2">
                <button className="flex-1 px-3 py-1.5 bg-primary/20 text-primary rounded text-xs hover:bg-primary/30 transition-colors">
                  <Github className="w-3 h-3 inline mr-1" />
                  GitHub
                </button>
                <button className="flex-1 px-3 py-1.5 bg-primary/20 text-primary rounded text-xs hover:bg-primary/30 transition-colors">
                  <Download className="w-3 h-3 inline mr-1" />
                  Download
                </button>
              </div>
            </div>

            <div className="bg-card-hover rounded-lg p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 bg-secondary/20 rounded-lg">
                  <Terminal className="w-5 h-5 text-secondary" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-text-primary">Node.js SDK</h4>
                  <p className="text-xs text-text-secondary">v2.3.1</p>
                </div>
              </div>
              <pre className="bg-background rounded p-2 mb-3">
                <code className="text-xs text-text-primary">npm install @nba-ml/sdk</code>
              </pre>
              <div className="flex gap-2">
                <button className="flex-1 px-3 py-1.5 bg-secondary/20 text-secondary rounded text-xs hover:bg-secondary/30 transition-colors">
                  <Github className="w-3 h-3 inline mr-1" />
                  GitHub
                </button>
                <button className="flex-1 px-3 py-1.5 bg-secondary/20 text-secondary rounded text-xs hover:bg-secondary/30 transition-colors">
                  <Download className="w-3 h-3 inline mr-1" />
                  Download
                </button>
              </div>
            </div>

            <div className="bg-card-hover rounded-lg p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 bg-warning/20 rounded-lg">
                  <Coffee className="w-5 h-5 text-warning" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-text-primary">Java SDK</h4>
                  <p className="text-xs text-text-secondary">v2.3.1</p>
                </div>
              </div>
              <pre className="bg-background rounded p-2 mb-3">
                <code className="text-xs text-text-primary">com.nbaml:sdk:2.3.1</code>
              </pre>
              <div className="flex gap-2">
                <button className="flex-1 px-3 py-1.5 bg-warning/20 text-warning rounded text-xs hover:bg-warning/30 transition-colors">
                  <Github className="w-3 h-3 inline mr-1" />
                  GitHub
                </button>
                <button className="flex-1 px-3 py-1.5 bg-warning/20 text-warning rounded text-xs hover:bg-warning/30 transition-colors">
                  <Download className="w-3 h-3 inline mr-1" />
                  Download
                </button>
              </div>
            </div>
          </div>
        </AnimatedCard>
      </div>
    </div>
  )
}