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
  photos?: string[]
  poi_id?: string
  image_url?: string
  maps_url?: string
  website_url?: string
  ticket_price?: number
  cost_status?: 'known' | 'estimated' | 'unknown'
}

export interface Meal {
  type: 'breakfast' | 'lunch' | 'dinner' | 'snack'
  name: string
  address?: string
  location?: Location
  description?: string
  estimated_cost?: number
  image_url?: string
  maps_url?: string
  website_url?: string
  poi_id?: string
  cost_status?: 'known' | 'estimated' | 'unknown'
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
  image_url?: string
  maps_url?: string
  website_url?: string
  poi_id?: string
  cost_status?: 'known' | 'estimated' | 'unknown'
}

export interface Budget {
  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total: number
  estimate_incomplete?: boolean
  currency?: string
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

export interface ValidationSummary {
  validated: boolean
  fallback_used: boolean
  date_coverage_passed: boolean
  budget_consistency_passed: boolean
  grounding_score?: number | null
  attribution_coverage_score?: number | null
  pacing_score?: number | null
  route_coherence_score?: number | null
  quality_warnings: string[]
  grounded_entity_count?: number | null
  checked_entity_count?: number | null
  evidence_summary?: string | null
}

export interface MemoryProfile {
  profile_id: string
  transportation: string
  accommodation: string
  preferences: string[]
  recent_cities: string[]
  preference_metadata?: Record<string, MemoryPreferenceMetadata[]>
  trip_count: number
  last_summary: string
  created_at?: number
  updated_at?: number
}

export interface MemoryPreferenceMetadata {
  value: string
  count: number
  last_seen_at?: number
  source_type: string
}

export interface MemoryConflictExplanation {
  field: string
  remembered_value: string
  current_value: string
  resolution: string
  count: number
  last_seen_at?: number
  source_type: string
  explanation: string
}

export interface TripFormData {
  city: string
  start_date: string
  end_date: string
  travel_days: number
  transportation: string
  accommodation: string
  preferences: string[]
  country_code?: string
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
  memory_conflicts?: MemoryConflictExplanation[]
  validation_summary?: ValidationSummary
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
  workflow_name: string
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
  agent_metrics: {
    by_agent?: Record<string, { attempts: number; latency_ms: number; token_usage: number; tool_calls: Record<string, number> }>
    targeted_retries?: string[]
    invalid_source_ids?: string[]
    handoff_trace?: Record<string, any>[]
  }
  proposal_versions: Record<string, number | null>
  tool_usage: Record<string, Record<string, number>>
  handoff_trace: Record<string, any>[]
  materialization_failures: Record<string, any>[]
  invalid_source_ids: string[]
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

export interface RAGDraft {
  doc_id: string
  country: string
  city: string
  district: string
  theme: string[]
  poi_names: string[]
  best_for: string[]
  recommended_duration: string
  seasonality: string[]
  transport_advice: string[]
  planning_tips: string[]
  source_type: string
  source_url: string
  language: string
  last_verified_at: string
  title: string
  content: string
  review_status: string
  reviewer: string
  review_notes: string
  source_id: string
  raw_html_path: string
  raw_text_path: string
  fetched_at: string
}

export interface RAGDraftSummary {
  draft_id: string
  doc_id: string
  country: string
  city: string
  title: string
  source_type: string
  source_url: string
  review_status: string
  promoted: boolean
  corpus_status: string
  reviewer: string
  updated_path: string
  fetched_at: string
}

export interface RAGDraftDetail extends RAGDraftSummary {
  draft: RAGDraft
  extracted_text: string
}

export interface RAGPrefillEvidence {
  field: string
  suggestion: string
  evidence: string
}

export interface RAGPrefillSuggestion {
  field: string
  value: string
  source_quote: string
  section_id: string
  section_heading: string
  time_sensitive: boolean
  confidence?: number | null
  status: 'accepted' | 'review_required' | 'rejected'
  reason: string
}

export interface RAGPrefillResponse {
  suggested_draft: RAGDraft
  suggestions: RAGPrefillSuggestion[]
  warnings: string[]
  source_char_count: number
  used_char_count: number
  section_count: number
  selected_section_count: number
  discarded_section_count: number
  accepted_suggestion_count: number
  review_required_suggestion_count: number
  rejected_suggestion_count: number
}

export interface RAGUrlIngestionRequest {
  source_id: string
  country: string
  city: string
  source_url: string
  source_type: string
  title: string
  theme: string[]
  poi_names: string[]
  district: string
  language: string
  best_for: string[]
  recommended_duration: string
  css_selector?: string
}

export interface RAGPromoteResult {
  country: string
  scanned: number
  approved: number
  promoted: number
  skipped_existing: number
}

export interface RAGIngestionJob {
  job_id: string
  job_type: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  started_at: number
  finished_at: number
  message: string
  error: string
  created_at: number
}
