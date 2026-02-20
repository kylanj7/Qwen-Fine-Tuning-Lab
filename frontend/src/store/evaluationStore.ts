import { create } from 'zustand'
import { evaluationApi, EvaluationRun, EvaluationResults, EvaluationDetailedResult } from '../api/client'

interface EvaluationState {
  evaluations: EvaluationRun[]
  currentEvaluation: EvaluationRun | null
  currentResults: EvaluationResults | null
  streamingResults: EvaluationDetailedResult[]
  loading: boolean
  error: string | null
  total: number

  fetchEvaluations: (params?: { skip?: number; limit?: number; status?: string }) => Promise<void>
  startEvaluation: (data: {
    model_path: string
    model_name: string
    dataset_config_name: string
    sample_count?: number
    judge_model?: string
  }) => Promise<EvaluationRun | null>
  getStatus: (id: number) => Promise<void>
  getResults: (id: number) => Promise<void>
  cancelEvaluation: (id: number) => Promise<void>
  updateCurrentEvaluation: (updates: Partial<EvaluationRun>) => void
  appendStreamingResult: (result: EvaluationDetailedResult) => void
  clearStreamingResults: () => void
  clearEvaluations: () => Promise<void>
  selectEvaluation: (id: number) => Promise<void>
}

export const useEvaluationStore = create<EvaluationState>((set, get) => ({
  evaluations: [],
  currentEvaluation: null,
  currentResults: null,
  streamingResults: [],
  loading: false,
  error: null,
  total: 0,

  fetchEvaluations: async (params) => {
    set({ loading: true, error: null })

    const response = await evaluationApi.list(params)

    if (response.error) {
      set({ loading: false, error: response.error })
      return
    }

    if (response.data) {
      set({
        evaluations: response.data.evaluations,
        total: response.data.total,
        loading: false,
      })
    }
  },

  startEvaluation: async (data) => {
    set({ loading: true, error: null })

    const response = await evaluationApi.start(data)

    if (response.error) {
      set({ loading: false, error: response.error })
      return null
    }

    if (response.data) {
      set({
        currentEvaluation: response.data,
        loading: false,
      })
      get().fetchEvaluations()
      return response.data
    }

    return null
  },

  getStatus: async (id) => {
    const response = await evaluationApi.getStatus(id)

    if (response.data) {
      set({ currentEvaluation: response.data })
    }
  },

  getResults: async (id) => {
    const response = await evaluationApi.getResults(id)

    if (response.data) {
      set({ currentResults: response.data })
    }
  },

  cancelEvaluation: async (id) => {
    const response = await evaluationApi.cancel(id)

    if (response.data) {
      get().getStatus(id)
      get().fetchEvaluations()
    }
  },

  updateCurrentEvaluation: (updates) => {
    const { currentEvaluation } = get()
    if (currentEvaluation) {
      set({ currentEvaluation: { ...currentEvaluation, ...updates } })
    }
  },

  appendStreamingResult: (result) => {
    set((state) => ({
      streamingResults: [...state.streamingResults, result],
    }))
  },

  clearStreamingResults: () => {
    set({ streamingResults: [] })
  },

  clearEvaluations: async () => {
    const response = await evaluationApi.clear()

    if (response.error) {
      set({ error: response.error })
      return
    }

    // Refresh the evaluations list
    get().fetchEvaluations()
  },

  selectEvaluation: async (id) => {
    // Find evaluation in list and set as current
    const { evaluations } = get()
    const evaluation = evaluations.find((e) => e.id === id)

    if (evaluation) {
      // Clear previous results and set the new current evaluation
      set({
        currentEvaluation: evaluation,
        currentResults: null,
        streamingResults: [],
      })
    }

    // Fetch full results including detailed_results
    const response = await evaluationApi.getResults(id)
    if (response.data) {
      set({ currentResults: response.data })

      // Also update currentEvaluation with scores from results if available
      if (response.data.overall_scores) {
        set((state) => ({
          currentEvaluation: state.currentEvaluation
            ? {
                ...state.currentEvaluation,
                scores: response.data!.overall_scores,
              }
            : null,
        }))
      }
    }
  },
}))
