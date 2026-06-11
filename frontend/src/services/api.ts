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
  timeout: 120000, // 2分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 清除匿名偏好记忆
 */
export async function clearTripMemory(profileId: string): Promise<MemoryClearResponse> {
  try {
    const response = await apiClient.post<MemoryClearResponse>('/api/trip/memory/clear', {
      profile_id: profileId
    })
    return response.data
  } catch (error: any) {
    console.error('清除匿名偏好记忆失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '清除匿名偏好记忆失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
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
