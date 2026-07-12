import { Button, Card, Col, Descriptions, Drawer, Empty, Input, message, Row, Select, Space, Statistic, Table, Tag, Timeline } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useEffect, useMemo, useState } from 'react'
import {
  clearObservabilityRuns,
  getObservabilityRunDetail,
  getObservabilitySummary,
  listObservabilityRuns
} from '@/services/api'
import type { ObservabilityEvent, ObservabilityRun, ObservabilityRunDetail, ObservabilitySummary } from '@/types'

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

function toPercent(value: number): number {
  return Math.round((value || 0) * 1000) / 10
}

function formatTimestamp(value: number): string {
  if (!value) return 'n/a'
  return new Date(value * 1000).toLocaleString()
}

export default function Observability() {
  const [summary, setSummary] = useState<ObservabilitySummary>(emptySummary)
  const [runs, setRuns] = useState<ObservabilityRun[]>([])
  const [selectedRun, setSelectedRun] = useState<ObservabilityRunDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [filters, setFilters] = useState<{
    source?: string
    city?: string
    passed?: boolean
    failure_type?: string
  }>({})

  const columns: ColumnsType<ObservabilityRun> = useMemo(() => [
    {
      title: 'Created',
      dataIndex: 'created_at',
      render: (value: number) => formatTimestamp(value)
    },
    { title: 'Source', dataIndex: 'source' },
    { title: 'City', dataIndex: 'city' },
    {
      title: 'Status',
      render: (_, record) => (
        <Tag color={record.passed ? 'green' : record.fallback ? 'orange' : 'red'}>
          {record.passed ? 'passed' : record.fallback ? 'fallback' : 'failed'}
        </Tag>
      )
    },
    {
      title: 'Failures',
      dataIndex: 'hard_failures',
      render: (failures: string[]) => failures?.length ? failures.map((item) => <Tag color="volcano" key={item}>{item}</Tag>) : <span className="muted">none</span>
    },
    {
      title: 'Latency',
      dataIndex: 'end_to_end_ms',
      render: (value: number) => `${Math.round(value || 0)}ms`
    },
    {
      title: 'Action',
      render: (_, record) => <Button size="small" onClick={() => openRun(record.run_id)}>Inspect</Button>
    }
  ], [])

  const eventColumns: ColumnsType<ObservabilityEvent> = [
    { title: 'Type', dataIndex: 'event_type' },
    { title: 'Node', dataIndex: 'node_name' },
    { title: 'Attempt', dataIndex: 'attempt' },
    { title: 'Latency', dataIndex: 'latency_ms' },
    { title: 'Message', dataIndex: 'message' }
  ]

  async function loadData() {
    setLoading(true)
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
      setSummary(summaryData || emptySummary)
      setRuns(runData || [])
    } catch (error: any) {
      message.error(error.message || 'Failed to load observability data.')
    } finally {
      setLoading(false)
    }
  }

  async function openRun(runId: string) {
    setDrawerOpen(true)
    setSelectedRun(null)
    setDetailLoading(true)
    try {
      setSelectedRun(await getObservabilityRunDetail(runId))
    } catch (error: any) {
      message.error(error.message || 'Failed to load run detail.')
    } finally {
      setDetailLoading(false)
    }
  }

  async function handleClearBenchmark() {
    try {
      const result = await clearObservabilityRuns('benchmark')
      message.success(`Deleted ${result.deleted} benchmark runs.`)
      await loadData()
    } catch (error: any) {
      message.error(error.message || 'Failed to clear benchmark runs.')
    }
  }

  useEffect(() => {
    void loadData()
  }, [])

  return (
    <main className="observability-page">
      <section className="page-heading">
        <div>
          <Tag color="blue">Agent Reliability</Tag>
          <h1>Evaluation Observability</h1>
          <p>Inspect planner traces, retry paths, validation failures, quality warnings, and RAG evidence.</p>
        </div>
        <Space wrap>
          <Button onClick={loadData} loading={loading}>Refresh</Button>
          <Button danger onClick={handleClearBenchmark}>Clear benchmark runs</Button>
        </Space>
      </section>

      <Row gutter={[16, 16]} className="summary-grid">
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Total Runs" value={summary.total_runs} /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Pass Rate" value={toPercent(summary.pass_rate)} suffix="%" /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Recovery Rate" value={toPercent(summary.recovery_rate)} suffix="%" /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Trace Coverage" value={toPercent(summary.trace_coverage)} suffix="%" /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Fallback Rate" value={toPercent(summary.fallback_rate)} suffix="%" /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Avg Latency" value={Math.round(summary.avg_latency_ms || 0)} suffix="ms" /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Avg Attribution" value={toPercent(summary.avg_attribution_coverage_score)} suffix="%" /></Card></Col>
        <Col xs={24} sm={12} lg={6}><Card><Statistic title="Quality Warning Rate" value={toPercent(summary.quality_warning_rate)} suffix="%" /></Card></Col>
      </Row>

      <Card className="runs-card" title="Recent Planner Runs" variant="borderless">
        <Space wrap className="filters">
          <Select
            allowClear
            placeholder="source"
            style={{ width: 160 }}
            value={filters.source}
            onChange={(value) => setFilters((current) => ({ ...current, source: value }))}
            options={[{ value: 'runtime' }, { value: 'benchmark' }]}
          />
          <Input
            allowClear
            placeholder="city"
            style={{ width: 160 }}
            value={filters.city}
            onChange={(event) => setFilters((current) => ({ ...current, city: event.target.value }))}
          />
          <Select
            allowClear
            placeholder="passed"
            style={{ width: 160 }}
            value={filters.passed}
            onChange={(value) => setFilters((current) => ({ ...current, passed: value }))}
            options={[{ value: true, label: 'passed' }, { value: false, label: 'failed' }]}
          />
          <Input
            allowClear
            placeholder="failure type"
            style={{ width: 220 }}
            value={filters.failure_type}
            onChange={(event) => setFilters((current) => ({ ...current, failure_type: event.target.value }))}
          />
          <Button type="primary" onClick={loadData}>Apply</Button>
        </Space>

        <Table rowKey="run_id" columns={columns} dataSource={runs} loading={loading} pagination={{ pageSize: 10 }} />
      </Card>

      <Drawer open={drawerOpen} onClose={() => setDrawerOpen(false)} width={760} title="Run Detail" loading={detailLoading}>
        {selectedRun ? (
          <div className="detail-stack">
            <Card size="small" title="Request">
              <Descriptions column={1} size="small">
                <Descriptions.Item label="Run">{selectedRun.run_id}</Descriptions.Item>
                <Descriptions.Item label="Source">{selectedRun.source}</Descriptions.Item>
                <Descriptions.Item label="Workflow">{selectedRun.workflow_name || 'legacy'}</Descriptions.Item>
                <Descriptions.Item label="City">{selectedRun.city}</Descriptions.Item>
                <Descriptions.Item label="Latency">{Math.round(selectedRun.end_to_end_ms || 0)}ms</Descriptions.Item>
              </Descriptions>
            </Card>

            <Card size="small" title="Agent Collaboration Timeline">
              <Timeline
                items={(selectedRun.events || [])
                  .filter((event) => ['agent_start', 'agent_tool', 'agent_handoff', 'agent_retry', 'materialization'].includes(event.event_type))
                  .map((event) => ({
                    color: event.event_type === 'agent_retry' ? 'orange' : event.event_type === 'materialization' ? 'purple' : 'blue',
                    children: <div><strong>{event.event_type}</strong> · {event.message}</div>
                  }))}
              />
              {Object.entries(selectedRun.agent_metrics?.by_agent || {}).map(([role, metric]) => (
                <Descriptions key={role} column={3} size="small" bordered>
                  <Descriptions.Item label="Agent">{role}</Descriptions.Item>
                  <Descriptions.Item label="Attempts">{metric.attempts}</Descriptions.Item>
                  <Descriptions.Item label="Latency">{Math.round(metric.latency_ms || 0)}ms</Descriptions.Item>
                </Descriptions>
              ))}
              <p><strong>Proposal versions:</strong> {JSON.stringify(selectedRun.proposal_versions || {})}</p>
              {(selectedRun.agent_metrics?.targeted_retries || []).length > 0 && (
                <p><strong>Retry owners:</strong> {selectedRun.agent_metrics.targeted_retries?.join(', ')}</p>
              )}
              {(selectedRun.materialization_failures || []).length > 0 && (
                <pre>{JSON.stringify(selectedRun.materialization_failures, null, 2)}</pre>
              )}
            </Card>

            <Card size="small" title="Evaluation">
              <Space wrap>
                <Tag color={selectedRun.passed ? 'green' : 'red'}>{selectedRun.passed ? 'passed' : 'failed'}</Tag>
                {selectedRun.recovered_after_retry && <Tag color="blue">recovered</Tag>}
                {selectedRun.fallback && <Tag color="orange">fallback</Tag>}
              </Space>
              <pre>{JSON.stringify(selectedRun.evaluation_report || {}, null, 2)}</pre>
            </Card>

            <Card size="small" title="Quality Scores">
              <Descriptions column={1} size="small">
                <Descriptions.Item label="Attribution">{toPercent(selectedRun.scores.attribution_coverage_score || 0)}%</Descriptions.Item>
                <Descriptions.Item label="Route Coherence">{toPercent(selectedRun.scores.route_coherence_score || 0)}%</Descriptions.Item>
                <Descriptions.Item label="Pacing">{toPercent(selectedRun.scores.pacing_score || 0)}%</Descriptions.Item>
                <Descriptions.Item label="Preference Match">{toPercent(selectedRun.scores.preference_match_score || 0)}%</Descriptions.Item>
              </Descriptions>
            </Card>

            <Card size="small" title="Evaluation History">
              <Timeline
                items={(selectedRun.evaluation_history || []).map((item, index) => ({
                  color: item.passed ? 'green' : 'red',
                  children: (
                    <div>
                      <strong>Attempt {index + 1}:</strong> {item.passed ? 'passed' : 'failed'}
                      {item.next_action ? ` -> ${item.next_action}` : ''}
                    </div>
                  )
                }))}
              />
            </Card>

            <Card size="small" title="Node Events">
              <Table rowKey="event_id" columns={eventColumns} dataSource={selectedRun.events} pagination={false} size="small" />
            </Card>
          </div>
        ) : (
          <Empty description="No run selected" />
        )}
      </Drawer>
    </main>
  )
}
