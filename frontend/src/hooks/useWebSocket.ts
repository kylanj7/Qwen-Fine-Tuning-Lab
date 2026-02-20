import { useEffect, useRef, useState, useCallback } from 'react'

interface WebSocketOptions {
  onMessage?: (data: unknown) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  lastMessage: unknown | null
  sendMessage: (data: unknown) => void
  connect: () => void
  disconnect: () => void
}

export function useWebSocket(
  url: string | null,
  options: WebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnectAttempts = 5,
    reconnectInterval = 2000,
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Use refs for callbacks to avoid reconnection on callback changes
  const onMessageRef = useRef(onMessage)
  const onOpenRef = useRef(onOpen)
  const onCloseRef = useRef(onClose)
  const onErrorRef = useRef(onError)

  // Update refs when callbacks change
  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])
  useEffect(() => { onOpenRef.current = onOpen }, [onOpen])
  useEffect(() => { onCloseRef.current = onClose }, [onClose])
  useEffect(() => { onErrorRef.current = onError }, [onError])

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<unknown | null>(null)

  const connect = useCallback(() => {
    if (!url) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return

    try {
      console.log('[WebSocket] Connecting to:', url)
      const ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('[WebSocket] Connected')
        setIsConnected(true)
        reconnectCountRef.current = 0
        onOpenRef.current?.()
      }

      ws.onclose = (event) => {
        console.log('[WebSocket] Closed:', event.code, event.reason)
        setIsConnected(false)
        onCloseRef.current?.()

        // Attempt reconnect if not a normal closure
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++
          console.log(`[WebSocket] Reconnecting (${reconnectCountRef.current}/${reconnectAttempts})...`)
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      ws.onerror = (event) => {
        console.error('[WebSocket] Error:', event)
        onErrorRef.current?.(event)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
          onMessageRef.current?.(data)
        } catch {
          // Handle non-JSON messages (like "pong")
          setLastMessage(event.data)
          onMessageRef.current?.(event.data)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('[WebSocket] Connection error:', error)
    }
  }, [url, reconnectAttempts, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    reconnectCountRef.current = reconnectAttempts // Prevent reconnection
    wsRef.current?.close()
    wsRef.current = null
    setIsConnected(false)
  }, [reconnectAttempts])

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }, [])

  // Auto-connect when URL changes
  useEffect(() => {
    if (url) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [url, connect, disconnect])

  // Keepalive ping
  useEffect(() => {
    if (!isConnected) return

    const pingInterval = setInterval(() => {
      sendMessage('ping')
    }, 25000)

    return () => clearInterval(pingInterval)
  }, [isConnected, sendMessage])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
  }
}

// Helper to construct WebSocket URL that works with both development proxy and production
function getWebSocketUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}

// Specialized hooks for different WebSocket types
export function useTrainingWebSocket(runId: number | null, onUpdate?: (data: unknown) => void) {
  const url = runId ? getWebSocketUrl(`/ws/training/${runId}`) : null

  return useWebSocket(url, {
    onMessage: onUpdate,
  })
}

export function useEvaluationWebSocket(evalId: number | null, onUpdate?: (data: unknown) => void) {
  const url = evalId ? getWebSocketUrl(`/ws/evaluation/${evalId}`) : null

  return useWebSocket(url, {
    onMessage: onUpdate,
  })
}

export function useInferenceWebSocket(sessionId: number | null, onUpdate?: (data: unknown) => void) {
  const url = sessionId ? getWebSocketUrl(`/ws/inference/${sessionId}`) : null

  return useWebSocket(url, {
    onMessage: onUpdate,
  })
}
