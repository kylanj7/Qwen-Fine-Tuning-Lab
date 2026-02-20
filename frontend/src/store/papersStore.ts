import { create } from 'zustand'
import { papersApi, DownloadedPaper, PaperDownloadRequest } from '../api/client'

interface PapersState {
  papers: DownloadedPaper[]
  loading: boolean
  error: string | null
  total: number

  fetchPapers: (params?: { skip?: number; limit?: number; status?: string }) => Promise<void>
  downloadPaper: (data: PaperDownloadRequest) => Promise<DownloadedPaper | null>
  deletePaper: (id: number) => Promise<void>
  retryDownload: (id: number) => Promise<void>
  clearAll: () => Promise<void>
}

export const usePapersStore = create<PapersState>((set, get) => ({
  papers: [],
  loading: false,
  error: null,
  total: 0,

  fetchPapers: async (params) => {
    set({ loading: true, error: null })

    const response = await papersApi.list(params)

    if (response.error) {
      set({ loading: false, error: response.error })
      return
    }

    if (response.data) {
      set({
        papers: response.data.papers,
        total: response.data.total,
        loading: false,
      })
    }
  },

  downloadPaper: async (data) => {
    const response = await papersApi.download(data)

    if (response.error) {
      set({ error: response.error })
      return null
    }

    if (response.data) {
      // Refresh the list
      get().fetchPapers()
      return response.data
    }

    return null
  },

  deletePaper: async (id) => {
    const response = await papersApi.delete(id)

    if (response.error) {
      set({ error: response.error })
      return
    }

    // Refresh the list
    get().fetchPapers()
  },

  retryDownload: async (id) => {
    const response = await papersApi.retry(id)

    if (response.error) {
      set({ error: response.error })
      return
    }

    // Refresh the list
    get().fetchPapers()
  },

  clearAll: async () => {
    const response = await papersApi.clearAll()

    if (response.error) {
      set({ error: response.error })
      return
    }

    // Refresh the list
    get().fetchPapers()
  },
}))
