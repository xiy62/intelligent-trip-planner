// 类型定义

export interface Location {
  longitude: number
  latitude: number
}

export interface Attraction {
  name: string
  address: string
  location: Location
  visit_duration: number
  description: string
  category?: string
  rating?: number
  image_url?: string
  ticket_price?: number
}

export interface Meal {
  type: 'breakfast' | 'lunch' | 'dinner' | 'snack'
  name: string
  address?: string
  location?: Location
  description?: string
  estimated_cost?: number
}

export interface Hotel {
  name: string
  address: string
  location?: Location
  price_range: string
  rating: string
  distance: string
  type: string
  estimated_cost?: number
}

export interface Budget {
  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total: number
}

export interface DayPlan {
  date: string
  day_index: number
  description: string
  transportation: string
  accommodation: string
  hotel?: Hotel
  attractions: Attraction[]
  meals: Meal[]
}

export interface WeatherInfo {
  date: string
  day_weather: string
  night_weather: string
  day_temp: number
  night_temp: number
  wind_direction: string
  wind_power: string
}

export interface TripPlan {
  city: string
  start_date: string
  end_date: string
  days: DayPlan[]
  weather_info: WeatherInfo[]
  overall_suggestions: string
  budget?: Budget
}

export interface MemoryProfile {
  profile_id: string
  transportation: string
  accommodation: string
  preferences: string[]
  recent_cities: string[]
  trip_count: number
  last_summary: string
  created_at?: number
  updated_at?: number
}

export interface TripFormData {
  city: string
  start_date: string
  end_date: string
  travel_days: number
  transportation: string
  accommodation: string
  preferences: string[]
  free_text_input: string
  profile_id?: string
  conversation_id?: string
}

export interface TripPlanResponse {
  success: boolean
  message: string
  data?: TripPlan
  conversation_id?: string
  memory_applied?: boolean
  memory_summary?: string
  memory_profile?: MemoryProfile
}

export interface MemoryClearResponse {
  success: boolean
  message: string
  profile_id: string
}

export interface ObservabilityRun {
  run_id: string
  conversation_id: string
  profile_id: string
  source: string
  city: string
  travel_days: number
  rag_mode: string
  started_at: number
  ended_at: number
  end_to_end_ms: number
  passed: boolean | null
  first_evaluation_pass: boolean | null
  final_evaluation_pass: boolean | null
  recovered_after_retry: boolean | null
  fallback: boolean | null
  evaluation_attempt_count: number
  hard_failures: string[]
  scores: Record<string, number>
  request: Record<string, any>
  final_plan_summary: Record<string, any>
  evaluation_report: Record<string, any>
  evaluation_history: Record<string, any>[]
  retry_counts: Record<string, number>
  retrieved_rag_sources: Record<string, any>[]
  benchmark_metadata: Record<string, any>
  created_at: number
}

export interface ObservabilityEvent {
  event_id: string
  run_id: string
  node_name: string
  attempt: number
  latency_ms: number
  event_type: string
  message: string
  payload: Record<string, any>
  created_at: number
}

export interface ObservabilityRunDetail extends ObservabilityRun {
  events: ObservabilityEvent[]
}

export interface ObservabilitySummary {
  total_runs: number
  pass_rate: number
  fallback_rate: number
  recovery_rate: number
  avg_latency_ms: number
  avg_evaluation_attempts: number
  failure_category_counts: Record<string, number>
  avg_grounding_score: number
  avg_pacing_score: number
  avg_route_coherence_score: number
  avg_preference_match_score: number
  avg_attribution_coverage_score: number
  quality_warning_rate: number
  attribution_coverage_rate: number
  trace_coverage: number
  failure_categorization_coverage: number
}

export interface ObservabilityResponse<T> {
  success: boolean
  data: T
}
