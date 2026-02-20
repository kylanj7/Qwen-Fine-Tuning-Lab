import { create } from 'zustand'
import { configsApi, ConfigItem } from '../api/client'

interface ConfigState {
  models: ConfigItem[]
  datasets: ConfigItem[]
  training: ConfigItem[]
  loading: boolean
  error: string | null
  fetchConfigs: () => Promise<void>
}

export const useConfigStore = create<ConfigState>((set) => ({
  models: [],
  datasets: [],
  training: [],
  loading: false,
  error: null,

  fetchConfigs: async () => {
    set({ loading: true, error: null })

    const response = await configsApi.getAll()

    if (response.error) {
      set({ loading: false, error: response.error })
      return
    }

    if (response.data) {
      set({
        models: response.data.models,
        datasets: response.data.datasets,
        training: response.data.training,
        loading: false,
      })
    }
  },
}))
