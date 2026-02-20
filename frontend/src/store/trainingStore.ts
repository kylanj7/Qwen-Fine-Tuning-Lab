import { create } from 'zustand'
import { trainingApi, TrainingRun } from '../api/client'

interface TrainingState {
  runs: TrainingRun[]
  currentRun: TrainingRun | null
  loading: boolean
  error: string | null
  total: number

  fetchRuns: (params?: { skip?: number; limit?: number; status?: string }) => Promise<void>
  startTraining: (data: {
    model_config_name: string
    dataset_config_name: string
    training_config_name: string
    parameter_overrides?: Record<string, unknown>
  }) => Promise<TrainingRun | null>
  getStatus: (id: number) => Promise<void>
  cancelRun: (id: number) => Promise<void>
  updateCurrentRun: (updates: Partial<TrainingRun>) => void
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  runs: [],
  currentRun: null,
  loading: false,
  error: null,
  total: 0,

  fetchRuns: async (params) => {
    set({ loading: true, error: null })

    const response = await trainingApi.list(params)

    if (response.error) {
      set({ loading: false, error: response.error })
      return
    }

    if (response.data) {
      set({
        runs: response.data.runs,
        total: response.data.total,
        loading: false,
      })
    }
  },

  startTraining: async (data) => {
    set({ loading: true, error: null })

    const response = await trainingApi.start(data)

    if (response.error) {
      set({ loading: false, error: response.error })
      return null
    }

    if (response.data) {
      set({
        currentRun: response.data,
        loading: false,
      })
      // Refresh the runs list
      get().fetchRuns()
      return response.data
    }

    return null
  },

  getStatus: async (id) => {
    const response = await trainingApi.getStatus(id)

    if (response.data) {
      set({ currentRun: response.data })
    }
  },

  cancelRun: async (id) => {
    const response = await trainingApi.cancel(id)

    if (response.data) {
      get().getStatus(id)
      get().fetchRuns()
    }
  },

  updateCurrentRun: (updates) => {
    const { currentRun } = get()
    if (currentRun) {
      set({ currentRun: { ...currentRun, ...updates } })
    }
  },
}))
