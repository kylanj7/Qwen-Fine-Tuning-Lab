import { useEffect, useState, useRef } from 'react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import Select from '../components/common/Select'
import { useModelStore } from '../store/modelStore'
import { inferenceApi, ChatMessage } from '../api/client'
import { useInferenceWebSocket } from '../hooks/useWebSocket'
import styles from './Chat.module.css'

interface StreamingState {
  isGenerating: boolean
  currentResponse: string
  tokensPerSec: number
  totalTokens: number
}

export default function Chat() {
  const { ggufModels, fetchGGUFModels } = useModelStore()

  // Chat state
  const [selectedModel, setSelectedModel] = useState('')
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')

  // Settings
  const [systemPrompt, setSystemPrompt] = useState(
    'You are a helpful AI assistant specializing in scientific topics.'
  )
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(40)
  const [maxTokens, setMaxTokens] = useState(2048)

  // Streaming state
  const [streaming, setStreaming] = useState<StreamingState>({
    isGenerating: false,
    currentResponse: '',
    tokensPerSec: 0,
    totalTokens: 0,
  })

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // WebSocket for streaming
  const { sendMessage, isConnected } = useInferenceWebSocket(sessionId, (data) => {
    const msg = data as {
      type: string
      content?: string
      token_count?: number
      tokens_per_sec?: number
      tokens?: number
      elapsed_sec?: number
      message?: string
    }

    switch (msg.type) {
      case 'start':
        setStreaming((prev) => ({ ...prev, isGenerating: true, currentResponse: '' }))
        break
      case 'token':
        setStreaming((prev) => ({
          ...prev,
          currentResponse: prev.currentResponse + (msg.content || ''),
          totalTokens: msg.token_count || prev.totalTokens,
        }))
        break
      case 'complete':
        setStreaming((prev) => {
          // Add completed message to history
          setMessages((msgs) => [
            ...msgs,
            {
              role: 'assistant',
              content: prev.currentResponse,
              tokens: msg.tokens,
              tokens_per_sec: msg.tokens_per_sec,
            },
          ])
          return {
            isGenerating: false,
            currentResponse: '',
            tokensPerSec: msg.tokens_per_sec || 0,
            totalTokens: msg.tokens || 0,
          }
        })
        break
      case 'error':
        setStreaming({ isGenerating: false, currentResponse: '', tokensPerSec: 0, totalTokens: 0 })
        alert(`Error: ${msg.message}`)
        break
    }
  })

  // Load models on mount
  useEffect(() => {
    fetchGGUFModels()
  }, [fetchGGUFModels])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming.currentResponse])

  // Create session when model is selected
  useEffect(() => {
    const createSession = async () => {
      if (!selectedModel) return

      const model = ggufModels.find((m) => m.name === selectedModel)
      if (!model) return

      const response = await inferenceApi.createSession(model.path, model.name)
      if (response.data) {
        setSessionId(response.data.id)
        setMessages([])
      }
    }

    createSession()
  }, [selectedModel, ggufModels])

  const handleSend = () => {
    if (!inputValue.trim() || !isConnected || streaming.isGenerating) return

    const model = ggufModels.find((m) => m.name === selectedModel)
    if (!model) return

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: inputValue.trim() }])

    // Send via WebSocket
    sendMessage({
      type: 'message',
      content: inputValue.trim(),
      model_path: model.path,
      system_prompt: systemPrompt,
      temperature,
      top_p: topP,
      top_k: topK,
      max_tokens: maxTokens,
    })

    setInputValue('')
    inputRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClear = async () => {
    if (sessionId) {
      await inferenceApi.clearSession(sessionId)
    }
    setMessages([])
    setStreaming({ isGenerating: false, currentResponse: '', tokensPerSec: 0, totalTokens: 0 })
  }

  const modelOptions = ggufModels.map((m) => ({
    value: m.name,
    label: `${m.name} (${m.size_mb.toFixed(1)} MB)`,
  }))

  return (
    <div className={styles.page}>
      <div className={styles.layout}>
        {/* Sidebar Settings */}
        <aside className={styles.sidebar}>
          <Card title="Model" padding="sm">
            <Select
              options={modelOptions}
              value={selectedModel}
              onChange={setSelectedModel}
              placeholder="Select a model..."
            />
          </Card>

          <Card title="System Prompt" padding="sm">
            <textarea
              className={styles.systemPrompt}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={4}
            />
          </Card>

          <Card title="Parameters" padding="sm">
            <div className={styles.paramSlider}>
              <label>Temperature: {temperature.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
              />
            </div>

            <div className={styles.paramSlider}>
              <label>Top P: {topP.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
              />
            </div>

            <div className={styles.paramSlider}>
              <label>Top K: {topK}</label>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
              />
            </div>

            <div className={styles.paramSlider}>
              <label>Max Tokens: {maxTokens}</label>
              <input
                type="range"
                min="256"
                max="4096"
                step="256"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              />
            </div>
          </Card>

          {/* Metrics */}
          {streaming.tokensPerSec > 0 && (
            <Card title="Metrics" padding="sm">
              <div className={styles.metrics}>
                <div className={styles.metricItem}>
                  <span className={styles.metricLabel}>Tokens/sec</span>
                  <span className={styles.metricValue}>{streaming.tokensPerSec.toFixed(1)}</span>
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricLabel}>Total Tokens</span>
                  <span className={styles.metricValue}>{streaming.totalTokens}</span>
                </div>
              </div>
            </Card>
          )}
        </aside>

        {/* Chat Area */}
        <main className={styles.chatArea}>
          <div className={styles.messagesContainer}>
            {messages.length === 0 && !streaming.currentResponse && (
              <div className={styles.emptyChat}>
                <h3>Start a conversation</h3>
                <p>Select a model and type a message to begin</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`${styles.message} ${styles[msg.role]}`}>
                <div className={styles.messageHeader}>
                  <span className={styles.messageRole}>
                    {msg.role === 'user' ? 'You' : 'Assistant'}
                  </span>
                  {msg.tokens_per_sec && (
                    <span className={styles.messageStats}>
                      {msg.tokens} tokens @ {msg.tokens_per_sec} t/s
                    </span>
                  )}
                </div>
                <div className={styles.messageContent}>{msg.content}</div>
              </div>
            ))}

            {streaming.currentResponse && (
              <div className={`${styles.message} ${styles.assistant}`}>
                <div className={styles.messageHeader}>
                  <span className={styles.messageRole}>Assistant</span>
                  <span className={styles.generating}>Generating...</span>
                </div>
                <div className={styles.messageContent}>{streaming.currentResponse}</div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className={styles.inputArea}>
            <div className={styles.inputContainer}>
              <textarea
                ref={inputRef}
                className={styles.input}
                placeholder={
                  isConnected ? 'Type a message...' : 'Select a model to start chatting'
                }
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!isConnected || streaming.isGenerating}
                rows={1}
              />
              <div className={styles.inputActions}>
                <Button variant="ghost" size="sm" onClick={handleClear} disabled={messages.length === 0}>
                  Clear
                </Button>
                <Button
                  onClick={handleSend}
                  disabled={!inputValue.trim() || !isConnected || streaming.isGenerating}
                  loading={streaming.isGenerating}
                >
                  Send
                </Button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
