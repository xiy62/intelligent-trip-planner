import axios from 'axios'
import type {
  MemoryClearResponse,
  ObservabilityResponse,
  ObservabilityRun,
  ObservabilityRunDetail,
  ObservabilitySummary,
  TripFormData,
  TripPlanResponse
} from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error)
)

export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || error.message || 'Failed to generate trip plan')
  }
}

export async function clearTripMemory(profileId: string): Promise<MemoryClearResponse> {
  try {
    const response = await apiClient.post<MemoryClearResponse>('/api/trip/memory/clear', {
      profile_id: profileId
    })
    return response.data
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || error.message || 'Failed to clear preference memory')
  }
}

export async function healthCheck(): Promise<any> {
  const response = await apiClient.get('/health')
  return response.data
}

export async function getObservabilitySummary(): Promise<ObservabilitySummary> {
  const response = await apiClient.get<ObservabilityResponse<ObservabilitySummary>>('/api/observability/summary')
  return response.data.data
}

export async function listObservabilityRuns(params: {
  limit?: number
  source?: string
  city?: string
  passed?: boolean
  failure_type?: string
} = {}): Promise<ObservabilityRun[]> {
  const response = await apiClient.get<ObservabilityResponse<ObservabilityRun[]>>('/api/observability/runs', {
    params
  })
  return response.data.data
}

export async function getObservabilityRunDetail(runId: string): Promise<ObservabilityRunDetail> {
  const response = await apiClient.get<ObservabilityResponse<ObservabilityRunDetail>>(`/api/observability/runs/${runId}`)
  return response.data.data
}

export async function clearObservabilityRuns(source?: string): Promise<{ success: boolean; deleted: number }> {
  const response = await apiClient.delete<{ success: boolean; deleted: number }>('/api/observability/runs', {
    params: source ? { source } : {}
  })
  return response.data
}

export default apiClient
