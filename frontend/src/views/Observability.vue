<template>
  <div class="observability-page">
    <div class="page-heading">
      <div>
        <p class="eyebrow">Agent Reliability</p>
        <h1>Evaluation Observability</h1>
        <p class="subtitle">
          Local dashboard for planner traces, retry paths, evaluation failures, and RAG grounding evidence.
        </p>
      </div>
      <a-space>
        <a-button @click="loadData" :loading="loading">刷新</a-button>
        <a-button danger @click="handleClearBenchmark">清理 Benchmark Runs</a-button>
      </a-space>
    </div>

    <a-row :gutter="[16, 16]" class="summary-grid">
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Total Runs" :value="summary.total_runs" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Pass Rate" :value="toPercent(summary.pass_rate)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Recovery Rate" :value="toPercent(summary.recovery_rate)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Trace Coverage" :value="toPercent(summary.trace_coverage)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Failure Categorization" :value="toPercent(summary.failure_categorization_coverage)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Fallback Rate" :value="toPercent(summary.fallback_rate)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Avg Latency" :value="Math.round(summary.avg_latency_ms)" suffix="ms" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Avg Grounding" :value="toPercent(summary.avg_grounding_score)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Avg Attribution" :value="toPercent(summary.avg_attribution_coverage_score)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Avg Route Coherence" :value="toPercent(summary.avg_route_coherence_score)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Avg Pacing" :value="toPercent(summary.avg_pacing_score)" suffix="%" />
        </a-card>
      </a-col>
      <a-col :xs="24" :sm="12" :lg="6">
        <a-card>
          <a-statistic title="Quality Warning Rate" :value="toPercent(summary.quality_warning_rate)" suffix="%" />
        </a-card>
      </a-col>
    </a-row>

    <a-card class="runs-card" title="Recent Planner Runs" :bordered="false">
      <div class="filters">
        <a-select v-model:value="filters.source" allow-clear placeholder="source" style="width: 160px">
          <a-select-option value="runtime">runtime</a-select-option>
          <a-select-option value="benchmark">benchmark</a-select-option>
        </a-select>
        <a-input v-model:value="filters.city" allow-clear placeholder="city" style="width: 160px" />
        <a-select v-model:value="filters.passed" allow-clear placeholder="passed" style="width: 160px">
          <a-select-option :value="true">passed</a-select-option>
          <a-select-option :value="false">failed</a-select-option>
        </a-select>
        <a-input v-model:value="filters.failure_type" allow-clear placeholder="failure type" style="width: 220px" />
        <a-button type="primary" @click="loadData">Apply</a-button>
      </div>

      <a-table
        row-key="run_id"
        :columns="columns"
        :data-source="runs"
        :loading="loading"
        :pagination="{ pageSize: 10 }"
        size="middle"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'status'">
            <a-tag :color="record.passed ? 'green' : record.fallback ? 'orange' : 'red'">
              {{ record.passed ? 'passed' : record.fallback ? 'fallback' : 'failed' }}
            </a-tag>
          </template>
          <template v-else-if="column.key === 'hard_failures'">
            <a-space wrap>
              <a-tag
                v-for="failure in record.hard_failures"
                :key="failure"
                color="volcano"
              >
                {{ failure }}
              </a-tag>
              <span v-if="!record.hard_failures.length" class="muted">none</span>
            </a-space>
          </template>
          <template v-else-if="column.key === 'latency'">
            {{ Math.round(record.end_to_end_ms) }}ms
          </template>
          <template v-else-if="column.key === 'created_at'">
            {{ formatTimestamp(record.created_at) }}
          </template>
          <template v-else-if="column.key === 'action'">
            <a-button size="small" @click="openRun(record.run_id)">Inspect</a-button>
          </template>
        </template>
      </a-table>
    </a-card>

    <a-drawer
      v-model:open="drawerOpen"
      width="760"
      title="Run Detail"
      placement="right"
    >
      <a-spin :spinning="detailLoading">
        <div v-if="selectedRun" class="detail-stack">
          <a-card size="small" title="Request">
            <div class="kv-grid">
              <span>Run</span><code>{{ selectedRun.run_id }}</code>
              <span>Conversation</span><code>{{ selectedRun.conversation_id || 'n/a' }}</code>
              <span>Source</span><strong>{{ selectedRun.source }}</strong>
              <span>City</span><strong>{{ selectedRun.city }}</strong>
              <span>RAG Mode</span><strong>{{ selectedRun.rag_mode || 'n/a' }}</strong>
              <span>Latency</span><strong>{{ Math.round(selectedRun.end_to_end_ms) }}ms</strong>
            </div>
          </a-card>

          <a-card size="small" title="Evaluation">
            <a-space wrap class="tag-row">
              <a-tag :color="selectedRun.passed ? 'green' : 'red'">
                {{ selectedRun.passed ? 'passed' : 'failed' }}
              </a-tag>
              <a-tag v-if="selectedRun.recovered_after_retry" color="blue">recovered</a-tag>
              <a-tag v-if="selectedRun.fallback" color="orange">fallback</a-tag>
            </a-space>
            <pre>{{ pretty(selectedRun.evaluation_report) }}</pre>
          </a-card>

          <a-card size="small" title="Quality Diagnostics">
            <div class="quality-grid">
              <span>Attribution</span>
              <strong>{{ toPercent(selectedRun.scores.attribution_coverage_score || 0) }}%</strong>
              <span>Route Coherence</span>
              <strong>{{ toPercent(selectedRun.scores.route_coherence_score || 0) }}%</strong>
              <span>Pacing</span>
              <strong>{{ toPercent(selectedRun.scores.pacing_score || 0) }}%</strong>
              <span>Preference Match</span>
              <strong>{{ toPercent(selectedRun.scores.preference_match_score || 0) }}%</strong>
            </div>
            <div class="quality-warnings">
              <a-tag
                v-for="warning in selectedRun.evaluation_report.quality_warnings || []"
                :key="warning"
                color="gold"
              >
                {{ warning }}
              </a-tag>
              <span v-if="!(selectedRun.evaluation_report.quality_warnings || []).length" class="muted">
                no quality warnings
              </span>
            </div>
          </a-card>

          <a-card size="small" title="Evidence Links">
            <a-table
              row-key="entity_name"
              :columns="evidenceColumns"
              :data-source="selectedRun.evaluation_report.evidence_links || []"
              :pagination="false"
              size="small"
            />
          </a-card>

          <a-card size="small" title="Evaluation History">
            <a-timeline>
              <a-timeline-item
                v-for="(item, index) in selectedRun.evaluation_history"
                :key="index"
                :color="item.passed ? 'green' : 'red'"
              >
                <strong>Attempt {{ index + 1 }}:</strong>
                {{ item.passed ? 'passed' : 'failed' }}
                <span v-if="item.next_action"> -> {{ item.next_action }}</span>
                <div class="history-failures">
                  <a-tag
                    v-for="failure in item.hard_failures || []"
                    :key="failure"
                    color="volcano"
                  >
                    {{ failure }}
                  </a-tag>
                  <span v-if="!(item.hard_failures || []).length" class="muted">no hard failures</span>
                </div>
              </a-timeline-item>
            </a-timeline>
          </a-card>

          <a-card size="small" title="Node Latency + Retry Events">
            <a-table
              row-key="event_id"
              :columns="eventColumns"
              :data-source="selectedRun.events"
              :pagination="false"
              size="small"
            />
          </a-card>

          <a-card size="small" title="Decision Trace">
            <ol class="trace-list">
              <li
                v-for="event in selectedRun.events.filter(item => item.event_type === 'routing')"
                :key="event.event_id"
              >
                {{ event.message }}
              </li>
            </ol>
          </a-card>

          <a-card size="small" title="Retrieved RAG Sources">
            <a-table
              row-key="chunk_id"
              :columns="ragColumns"
              :data-source="selectedRun.retrieved_rag_sources"
              :pagination="false"
              size="small"
            />
          </a-card>
        </div>
        <a-empty v-else description="No run selected" />
      </a-spin>
    </a-drawer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { message } from 'ant-design-vue'
import {
  clearObservabilityRuns,
  getObservabilityRunDetail,
  getObservabilitySummary,
  listObservabilityRuns
} from '@/services/api'
import type { ObservabilityRun, ObservabilityRunDetail, ObservabilitySummary } from '@/types'

const emptySummary: ObservabilitySummary = {
  total_runs: 0,
  pass_rate: 0,
  fallback_rate: 0,
  recovery_rate: 0,
  avg_latency_ms: 0,
  avg_evaluation_attempts: 0,
  failure_category_counts: {},
  avg_grounding_score: 0,
  avg_pacing_score: 0,
  avg_route_coherence_score: 0,
  avg_preference_match_score: 0,
  avg_attribution_coverage_score: 0,
  quality_warning_rate: 0,
  attribution_coverage_rate: 0,
  trace_coverage: 0,
  failure_categorization_coverage: 0
}

const loading = ref(false)
const detailLoading = ref(false)
const drawerOpen = ref(false)
const runs = ref<ObservabilityRun[]>([])
const selectedRun = ref<ObservabilityRunDetail | null>(null)
const summary = ref<ObservabilitySummary>({ ...emptySummary })
const filters = reactive<{
  source?: string
  city?: string
  passed?: boolean
  failure_type?: string
}>({})

const columns = computed(() => [
  { title: 'Created', key: 'created_at', dataIndex: 'created_at' },
  { title: 'Source', key: 'source', dataIndex: 'source' },
  { title: 'City', key: 'city', dataIndex: 'city' },
  { title: 'Status', key: 'status' },
  { title: 'Failures', key: 'hard_failures' },
  { title: 'Latency', key: 'latency' },
  { title: 'Actions', key: 'action' }
])

const eventColumns = [
  { title: 'Type', dataIndex: 'event_type', key: 'event_type' },
  { title: 'Node', dataIndex: 'node_name', key: 'node_name' },
  { title: 'Attempt', dataIndex: 'attempt', key: 'attempt' },
  { title: 'Latency', dataIndex: 'latency_ms', key: 'latency_ms' },
  { title: 'Message', dataIndex: 'message', key: 'message' }
]

const ragColumns = [
  { title: 'Doc', dataIndex: 'doc_id', key: 'doc_id' },
  { title: 'Chunk', dataIndex: 'chunk_id', key: 'chunk_id' },
  { title: 'Section', dataIndex: 'section', key: 'section' },
  { title: 'Theme', dataIndex: 'theme', key: 'theme' }
]

const evidenceColumns = [
  { title: 'Entity', dataIndex: 'entity_name', key: 'entity_name' },
  { title: 'Type', dataIndex: 'entity_type', key: 'entity_type' },
  { title: 'Evidence', dataIndex: 'evidence_type', key: 'evidence_type' },
  { title: 'Evidence ID', dataIndex: 'evidence_id', key: 'evidence_id' },
  { title: 'Confidence', dataIndex: 'confidence', key: 'confidence' },
  { title: 'Reason', dataIndex: 'match_reason', key: 'match_reason' }
]

function toPercent(value: number): number {
  return Math.round((value || 0) * 1000) / 10
}

function formatTimestamp(value: number): string {
  if (!value) return 'n/a'
  return new Date(value * 1000).toLocaleString()
}

function pretty(value: any): string {
  return JSON.stringify(value || {}, null, 2)
}

async function loadData() {
  loading.value = true
  try {
    const params = {
      limit: 100,
      source: filters.source || undefined,
      city: filters.city || undefined,
      passed: filters.passed,
      failure_type: filters.failure_type || undefined
    }
    const [summaryData, runData] = await Promise.all([
      getObservabilitySummary(),
      listObservabilityRuns(params)
    ])
    summary.value = summaryData
    runs.value = runData
  } catch (error: any) {
    message.error(error.message || '加载observability数据失败')
  } finally {
    loading.value = false
  }
}

async function openRun(runId: string) {
  drawerOpen.value = true
  detailLoading.value = true
  try {
    selectedRun.value = await getObservabilityRunDetail(runId)
  } catch (error: any) {
    message.error(error.message || '读取run详情失败')
  } finally {
    detailLoading.value = false
  }
}

async function handleClearBenchmark() {
  loading.value = true
  try {
    const result = await clearObservabilityRuns('benchmark')
    message.success(`已清理 ${result.deleted} 条benchmark traces`)
    await loadData()
  } catch (error: any) {
    message.error(error.message || '清理失败')
  } finally {
    loading.value = false
  }
}

onMounted(loadData)
</script>

<style scoped>
.observability-page {
  max-width: 1440px;
  margin: 0 auto;
}

.page-heading {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 24px;
}

.eyebrow {
  margin: 0 0 6px;
  color: #0f766e;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: 34px;
}

.subtitle {
  max-width: 760px;
  margin: 8px 0 0;
  color: #64748b;
}

.summary-grid {
  margin-bottom: 20px;
}

.runs-card {
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}

.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
}

.muted {
  color: #94a3b8;
}

.detail-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.kv-grid {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: 8px 12px;
}

.kv-grid span {
  color: #64748b;
}

code {
  white-space: pre-wrap;
  word-break: break-all;
}

pre {
  max-height: 320px;
  overflow: auto;
  padding: 12px;
  margin: 12px 0 0;
  background: #0f172a;
  border-radius: 8px;
  color: #e2e8f0;
}

.tag-row {
  margin-bottom: 8px;
}

.trace-list {
  padding-left: 20px;
  margin: 0;
}

.trace-list li {
  margin-bottom: 8px;
}

.history-failures {
  margin-top: 8px;
}

.quality-grid {
  display: grid;
  grid-template-columns: 160px minmax(0, 1fr);
  gap: 8px 12px;
}

.quality-grid span {
  color: #64748b;
}

.quality-warnings {
  margin-top: 12px;
}

@media (max-width: 768px) {
  .page-heading {
    flex-direction: column;
  }
}
</style>
