import { create } from 'zustand'
import { modelsApi, LoRAAdapter, GGUFModel, ConversionJob } from '../api/client'

interface ModelState {
  adapters: LoRAAdapter[]
  ggufModels: GGUFModel[]
  conversions: ConversionJob[]
  currentConversion: ConversionJob | null
  loading: boolean
  error: string | null

  fetchAdapters: () => Promise<void>
  fetchGGUFModels: () => Promise<void>
  fetchConversions: () => Promise<void>
  startConversion: (data: {
    adapter_path: string
    adapter_name: string
    base_model?: string
    quantization_method?: string
    output_name?: string
  }) => Promise<ConversionJob | null>
  getConversionStatus: (id: number) => Promise<void>
  deleteAdapter: (name: string) => Promise<boolean>
  deleteGGUFModel: (name: string) => Promise<boolean>
  updateCurrentConversion: (updates: Partial<ConversionJob>) => void
}

export const useModelStore = create<ModelState>((set, get) => ({
  adapters: [],
  ggufModels: [],
  conversions: [],
  currentConversion: null,
  loading: false,
  error: null,

  fetchAdapters: async () => {
    set({ loading: true, error: null })

    const response = await modelsApi.getAdapters()

    if (response.error) {
      set({ loading: false, error: response.error })
      return
    }

    if (response.data) {
      set({ adapters: response.data, loading: false })
    }
  },

  fetchGGUFModels: async () => {
    set({ loading: true, error: null })

    const response = await modelsApi.getGGUF()

    if (response.error) {
      set({ loading: false, error: response.error })
      return
    }

    if (response.data) {
      set({ ggufModels: response.data, loading: false })
    }
  },

  fetchConversions: async () => {
    const response = await modelsApi.listConversions()

    if (response.data) {
      set({ conversions: response.data })
    }
  },

  startConversion: async (data) => {
    set({ loading: true, error: null })

    const response = await modelsApi.convert(data)

    if (response.error) {
      set({ loading: false, error: response.error })
      return null
    }

    if (response.data) {
      set({
        currentConversion: response.data,
        loading: false,
      })
      get().fetchConversions()
      return response.data
    }

    return null
  },

  getConversionStatus: async (id) => {
    const response = await modelsApi.getConversionStatus(id)

    if (response.data) {
      set({ currentConversion: response.data })
    }
  },

  deleteAdapter: async (name) => {
    const response = await modelsApi.deleteAdapter(name)

    if (response.data) {
      get().fetchAdapters()
      return true
    }

    return false
  },

  deleteGGUFModel: async (name) => {
    const response = await modelsApi.deleteGGUFModel(name)

    if (response.data) {
      get().fetchGGUFModels()
      return true
    }

    return false
  },

  updateCurrentConversion: (updates) => {
    const { currentConversion } = get()
    if (currentConversion) {
      set({ currentConversion: { ...currentConversion, ...updates } })
    }
  },
}))
