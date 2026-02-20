const API_BASE = '/api'

interface ApiResponse<T> {
  data?: T
  error?: string
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }))
      return { error: error.detail || `HTTP ${response.status}` }
    }

    const data = await response.json()
    return { data }
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Network error' }
  }
}

// Config endpoints
export const configsApi = {
  getAll: () => request<{
    models: ConfigItem[]
    datasets: ConfigItem[]
    training: ConfigItem[]
  }>('/configs/all'),

  getModels: () => request<{ configs: ConfigItem[] }>('/configs/models'),
  getDatasets: () => request<{ configs: ConfigItem[] }>('/configs/datasets'),
  getTraining: () => request<{ configs: ConfigItem[] }>('/configs/training'),

  getDetail: (type: string, filename: string) =>
    request<ConfigDetail>(`/configs/${type}/${filename}`),
}

// Training endpoints
export const trainingApi = {
  start: (data: TrainingStartRequest) =>
    request<TrainingRun>('/training/start', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getStatus: (id: number) => request<TrainingRun>(`/training/status/${id}`),

  list: (params?: { skip?: number; limit?: number; status?: string }) => {
    const query = new URLSearchParams()
    if (params?.skip) query.set('skip', String(params.skip))
    if (params?.limit) query.set('limit', String(params.limit))
    if (params?.status) query.set('status', params.status)
    return request<{ runs: TrainingRun[]; total: number }>(`/training/list?${query}`)
  },

  cancel: (id: number) =>
    request<{ status: string }>(`/training/cancel/${id}`, { method: 'POST' }),

  getLogs: (id: number, lines = 100) =>
    request<{ logs: string[] }>(`/training/logs/${id}?lines=${lines}`),
}

// Evaluation endpoints
export const evaluationApi = {
  start: (data: EvaluationStartRequest) =>
    request<EvaluationRun>('/evaluation/start', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getStatus: (id: number) => request<EvaluationRun>(`/evaluation/status/${id}`),

  getResults: (id: number) => request<EvaluationResults>(`/evaluation/results/${id}`),

  list: (params?: { skip?: number; limit?: number; status?: string }) => {
    const query = new URLSearchParams()
    if (params?.skip) query.set('skip', String(params.skip))
    if (params?.limit) query.set('limit', String(params.limit))
    if (params?.status) query.set('status', params.status)
    return request<{ evaluations: EvaluationRun[]; total: number }>(`/evaluation/list?${query}`)
  },

  cancel: (id: number) =>
    request<{ status: string }>(`/evaluation/cancel/${id}`, { method: 'POST' }),

  clear: () =>
    request<{ status: string; deleted_count: number }>('/evaluation/clear', { method: 'DELETE' }),
}

// Model endpoints
export const modelsApi = {
  getAdapters: () => request<LoRAAdapter[]>('/models/adapters'),
  getGGUF: () => request<GGUFModel[]>('/models/gguf'),

  convert: (data: ConversionRequest) =>
    request<ConversionJob>('/models/convert', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getConversionStatus: (id: number) =>
    request<ConversionJob>(`/models/convert/status/${id}`),

  listConversions: (params?: { skip?: number; limit?: number }) => {
    const query = new URLSearchParams()
    if (params?.skip) query.set('skip', String(params.skip))
    if (params?.limit) query.set('limit', String(params.limit))
    return request<ConversionJob[]>(`/models/convert/list?${query}`)
  },

  deleteGGUF: (name: string) =>
    request<{ status: string }>(`/models/gguf/${name}`, { method: 'DELETE' }),

  deleteAdapter: (name: string) =>
    request<{ status: string }>(`/models/adapters/${name}`, { method: 'DELETE' }),
}

// Inference endpoints
export const inferenceApi = {
  createSession: (modelPath: string, modelName: string) =>
    request<ChatSession>(`/inference/session/create?model_path=${encodeURIComponent(modelPath)}&model_name=${encodeURIComponent(modelName)}`, {
      method: 'POST',
    }),

  getSession: (id: number) => request<ChatSession>(`/inference/session/${id}`),

  listSessions: (params?: { skip?: number; limit?: number }) => {
    const query = new URLSearchParams()
    if (params?.skip) query.set('skip', String(params.skip))
    if (params?.limit) query.set('limit', String(params.limit))
    return request<ChatSession[]>(`/inference/sessions?${query}`)
  },

  deleteSession: (id: number) =>
    request<{ status: string }>(`/inference/session/${id}`, { method: 'DELETE' }),

  clearSession: (id: number) =>
    request<{ status: string }>(`/inference/session/${id}/clear`, { method: 'POST' }),

  updateSettings: (id: number, settings: GenerationSettings) =>
    request<{ status: string }>(`/inference/session/${id}/settings`, {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
}

// Types
export interface ConfigItem {
  name: string
  filename: string
  path: string
  description?: string
}

export interface ConfigDetail {
  name: string
  filename: string
  path: string
  content: Record<string, unknown>
}

export interface TrainingStartRequest {
  model_config_name: string
  dataset_config_name: string
  training_config_name: string
  parameter_overrides?: Record<string, unknown>
}

export interface TrainingRun {
  id: number
  run_name: string
  model_config_name: string
  dataset_config_name: string
  training_config_name: string
  status: string
  current_step: number
  total_steps: number
  current_epoch: number
  total_epochs: number
  current_loss?: number
  best_loss?: number
  wandb_run_url?: string
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface EvaluationStartRequest {
  model_path: string
  model_name: string
  dataset_config_name: string
  sample_count?: number
  judge_model?: string
}

export interface EvaluationScores {
  factual_accuracy?: number
  completeness?: number
  technical_precision?: number
  overall_score?: number
}

export interface EvaluationRun {
  id: number
  model_path: string
  model_name: string
  dataset_config_name: string
  sample_count: number
  judge_model: string
  status: string
  current_sample: number
  total_samples: number
  scores?: EvaluationScores
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface EvaluationResults {
  id: number
  model_name: string
  status: string
  overall_scores?: EvaluationScores
  detailed_results: EvaluationDetailedResult[]
  results_file?: string
  articles_file?: string
}

export interface EvaluationDetailedResult {
  question_idx: number
  question: string
  model_response: string
  justification?: string
  scores: EvaluationScores
  rag_sources: RagSource[]
}

export interface RagSource {
  title: string
  authors?: string[]
  year?: number
  abstract?: string
  url?: string
  pdf_url?: string
  is_open_access?: boolean
}

export interface LoRAAdapter {
  name: string
  path: string
  base_model?: string
  dataset?: string
  created_at?: string
  size_mb: number
}

export interface GGUFModel {
  name: string
  path: string
  filename: string
  size_mb: number
  quantization?: string
  created_at?: string
}

export interface ConversionRequest {
  adapter_path: string
  adapter_name: string
  base_model?: string
  quantization_method?: string
  output_name?: string
}

export interface ConversionJob {
  id: number
  adapter_name: string
  base_model: string
  quantization_method: string
  status: string
  current_stage: string
  progress: number
  output_path?: string
  output_size_bytes?: number
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface ChatSession {
  id: number
  model_path: string
  model_name: string
  system_prompt: string
  messages: ChatMessage[]
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  tokens?: number
  tokens_per_sec?: number
}

export interface GenerationSettings {
  temperature?: number
  top_p?: number
  top_k?: number
  max_tokens?: number
  system_prompt?: string
}

export interface PaperDownloadRequest {
  paper_id: string
  title: string
  pdf_url: string
  authors?: string[]
  year?: number
  citation_count?: number
  semantic_scholar_url?: string
  evaluation_id?: number
}

export interface DownloadedPaper {
  id: number
  paper_id: string
  title: string
  authors: string[]
  year?: number
  citation_count: number
  semantic_scholar_url: string
  pdf_url: string
  status: string
  progress: number
  local_path?: string
  file_size_bytes?: number
  evaluation_id?: number
  created_at: string
  downloaded_at?: string
  error_message?: string
}

// Papers endpoints
export const papersApi = {
  download: (data: PaperDownloadRequest) =>
    request<DownloadedPaper>('/papers/download', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  list: (params?: { skip?: number; limit?: number; status?: string }) => {
    const query = new URLSearchParams()
    if (params?.skip) query.set('skip', String(params.skip))
    if (params?.limit) query.set('limit', String(params.limit))
    if (params?.status) query.set('status', params.status)
    return request<{ papers: DownloadedPaper[]; total: number }>(`/papers/list?${query}`)
  },

  get: (id: number) => request<DownloadedPaper>(`/papers/${id}`),

  getFileUrl: (id: number) => `${API_BASE}/papers/${id}/file`,

  delete: (id: number) =>
    request<{ status: string }>(`/papers/${id}`, { method: 'DELETE' }),

  retry: (id: number) =>
    request<DownloadedPaper>(`/papers/${id}/retry`, { method: 'POST' }),

  clearAll: () =>
    request<{ status: string; deleted_count: number }>('/papers/clear/all', { method: 'DELETE' }),
}
