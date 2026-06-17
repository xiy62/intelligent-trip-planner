import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  Upload,
  message
} from 'antd'
import type { UploadFile } from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import {
  aiPrefillRagDraft,
  approveRagDraft,
  getRagDraft,
  getRagIngestionJob,
  listRagDrafts,
  promoteRagDrafts,
  rebuildRagIndex,
  updateRagDraft,
  uploadRagSource
} from '@/services/api'
import type { RAGDraft, RAGDraftDetail, RAGDraftSummary, RAGIngestionJob, RAGPrefillResponse } from '@/types'

const { Title, Paragraph, Text } = Typography
const { TextArea } = Input

const arrayFields: Array<keyof RAGDraft> = [
  'theme',
  'poi_names',
  'best_for',
  'seasonality',
  'transport_advice',
  'planning_tips'
]

const emptyUpload = {
  source_id: '',
  country: 'US',
  city: '',
  source_url: '',
  source_type: 'official_tourism_portal',
  title: '',
  theme: [] as string[],
  poi_names: [] as string[],
  district: '',
  language: 'en',
  best_for: [] as string[],
  recommended_duration: ''
}

function statusColor(status: string) {
  if (status === 'approved') return 'green'
  if (status === 'failed') return 'red'
  if (status === 'running') return 'blue'
  if (status === 'succeeded') return 'green'
  return 'gold'
}

function normalizedDraft(base: RAGDraft, values: Partial<RAGDraft>): RAGDraft {
  const merged = { ...base, ...values }
  for (const field of arrayFields) {
    const value = merged[field]
    ;(merged as any)[field] = Array.isArray(value)
      ? value.map((item) => String(item).trim()).filter(Boolean)
      : []
  }
  return {
    ...merged,
    country: merged.country || 'US',
    language: merged.language || 'en',
    review_status: merged.review_status || 'draft',
    reviewer: merged.reviewer || '',
    review_notes: merged.review_notes || ''
  }
}

function cleanExtractedTextForContent(text: string, maxChars?: number) {
  const cleaned = (text || '')
    .replace(/\r/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !/^https?:\/\//i.test(line) && !/^\d+\s*\/\s*\d+$/.test(line))
    .join('\n')
    .trim()

  if (!maxChars || cleaned.length <= maxChars) {
    return cleaned
  }
  return cleaned.slice(0, maxChars).trim()
}

export default function RAGIngestion() {
  const [uploadForm] = Form.useForm()
  const [draftForm] = Form.useForm<RAGDraft>()
  const [drafts, setDrafts] = useState<RAGDraftSummary[]>([])
  const [selectedDetail, setSelectedDetail] = useState<RAGDraftDetail | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [loadingDrafts, setLoadingDrafts] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [prefilling, setPrefilling] = useState(false)
  const [promoting, setPromoting] = useState(false)
  const [job, setJob] = useState<RAGIngestionJob | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [previewJson, setPreviewJson] = useState('')
  const [aiPrefill, setAiPrefill] = useState<RAGPrefillResponse | null>(null)
  const watchedContent = Form.useWatch('content', draftForm) || ''

  const reviewDrafts = useMemo(
    () => drafts.filter((draft) => !draft.promoted),
    [drafts]
  )
  const promotedDrafts = useMemo(
    () => drafts.filter((draft) => draft.promoted),
    [drafts]
  )
  const readyToPromoteCount = useMemo(
    () => reviewDrafts.filter((draft) => draft.review_status === 'approved').length,
    [reviewDrafts]
  )
  const promotedCount = useMemo(
    () => promotedDrafts.length,
    [promotedDrafts]
  )

  async function loadDrafts() {
    setLoadingDrafts(true)
    try {
      setDrafts(await listRagDrafts({ country: 'US' }))
    } catch (error: any) {
      message.error(error.message || 'Failed to load RAG drafts')
    } finally {
      setLoadingDrafts(false)
    }
  }

  useEffect(() => {
    loadDrafts()
  }, [])

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    const timer = window.setInterval(async () => {
      try {
        setJob(await getRagIngestionJob(job.job_id))
      } catch (error: any) {
        message.error(error.message || 'Failed to poll rebuild job')
        window.clearInterval(timer)
      }
    }, 1600)
    return () => window.clearInterval(timer)
  }, [job])

  async function handleUpload(values: typeof emptyUpload) {
    if (!fileList[0]?.originFileObj) {
      message.warning('Choose a PDF, Markdown, or text file first')
      return
    }
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', fileList[0].originFileObj)
      Object.entries(values).forEach(([key, value]) => {
        formData.append(key, Array.isArray(value) ? value.join(',') : String(value ?? ''))
      })
      const detail = await uploadRagSource(formData)
      setFileList([])
      uploadForm.resetFields()
      uploadForm.setFieldsValue(emptyUpload)
      await loadDrafts()
      await openDraft(detail.draft_id)
      message.success('Draft generated from uploaded source')
    } catch (error: any) {
      message.error(error.response?.data?.detail || error.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  async function openDraft(draftId: string) {
    try {
      const detail = await getRagDraft(draftId)
      setSelectedDetail(detail)
      draftForm.setFieldsValue(detail.draft)
      setAiPrefill(null)
      setDrawerOpen(true)
    } catch (error: any) {
      message.error(error.message || 'Failed to open draft')
    }
  }

  function buildCurrentDraftPayload(): RAGDraft {
    if (!selectedDetail) {
      throw new Error('No draft selected')
    }
    return normalizedDraft(selectedDetail.draft, draftForm.getFieldsValue(true))
  }

  async function saveDraft() {
    if (!selectedDetail) return null
    const values = await draftForm.validateFields()
    const payload = normalizedDraft(selectedDetail.draft, values)
    setSaving(true)
    try {
      const detail = await updateRagDraft(selectedDetail.draft_id, payload)
      setSelectedDetail(detail)
      draftForm.setFieldsValue(detail.draft)
      await loadDrafts()
      message.success('Draft saved')
      return detail
    } catch (error: any) {
      message.error(error.response?.data?.detail || error.message || 'Failed to save draft')
      return null
    } finally {
      setSaving(false)
    }
  }

  async function approveDraft() {
    if (!selectedDetail) return
    const saved = await saveDraft()
    if (!saved) return
    setSaving(true)
    try {
      const current = draftForm.getFieldsValue(true)
      const detail = await approveRagDraft(saved.draft_id, {
        reviewer: current.reviewer || 'local-admin',
        review_notes: current.review_notes || ''
      })
      setSelectedDetail(detail)
      draftForm.setFieldsValue(detail.draft)
      await loadDrafts()
      message.success('Draft approved')
    } catch (error: any) {
      message.error(error.response?.data?.detail || error.message || 'Failed to approve draft')
    } finally {
      setSaving(false)
    }
  }

  function previewCurrentJson() {
    const payload = buildCurrentDraftPayload()
    setPreviewJson(JSON.stringify(payload, null, 2))
    setPreviewOpen(true)
  }

  function fillContentFromExtractedText(maxChars?: number) {
    if (!selectedDetail?.extracted_text) {
      message.warning('No extracted text is available for this draft')
      return
    }
    const content = cleanExtractedTextForContent(selectedDetail.extracted_text, maxChars)
    draftForm.setFieldValue('content', content)
    message.success(
      maxChars
        ? `Knowledge content filled with the first ${content.length.toLocaleString()} cleaned characters`
        : `Knowledge content filled with ${content.length.toLocaleString()} cleaned characters`
    )
  }

  async function applyAiPrefill() {
    if (!selectedDetail) return
    setPrefilling(true)
    try {
      const result = await aiPrefillRagDraft(selectedDetail.draft_id)
      setAiPrefill(result)
      draftForm.setFieldsValue(result.suggested_draft)
      message.success('AI suggestions applied. Review before saving.')
    } catch (error: any) {
      message.error(error.response?.data?.detail || error.message || 'AI prefill failed')
    } finally {
      setPrefilling(false)
    }
  }

  async function promoteApproved() {
    if (readyToPromoteCount === 0) {
      message.info('No approved drafts are ready to promote')
      return
    }
    setPromoting(true)
    try {
      const result = await promoteRagDrafts({ country: 'US', overwrite: false })
      await loadDrafts()
      if (result.promoted > 0) {
        message.success(`Promoted ${result.promoted} approved draft(s) into the ${result.country.toUpperCase()} corpus`)
      } else if (result.skipped_existing > 0) {
        message.info(`${result.skipped_existing} approved draft(s) were already promoted; no new docs written`)
      } else {
        message.info('No approved drafts are ready to promote')
      }
    } catch (error: any) {
      message.error(error.response?.data?.detail || error.message || 'Promotion failed')
    } finally {
      setPromoting(false)
    }
  }

  async function startRebuild() {
    try {
      const queued = await rebuildRagIndex()
      setJob(queued)
      message.success('Queued Chroma index rebuild')
    } catch (error: any) {
      message.error(error.response?.data?.detail || error.message || 'Failed to queue rebuild')
    }
  }

  function renderDraftTitle(value: string, record: RAGDraftSummary) {
    return (
      <Space direction="vertical" size={0}>
        <Button type="link" className="link-button" onClick={() => openDraft(record.draft_id)}>
          {value}
        </Button>
        <Text type="secondary">{record.doc_id}</Text>
      </Space>
    )
  }

  const reviewColumns: ColumnsType<RAGDraftSummary> = [
    {
      title: 'Title',
      dataIndex: 'title',
      render: renderDraftTitle
    },
    { title: 'City', dataIndex: 'city', width: 130 },
    {
      title: 'Status',
      dataIndex: 'review_status',
      width: 110,
      render: (status) => <Tag color={statusColor(status)}>{status}</Tag>
    },
    { title: 'Reviewer', dataIndex: 'reviewer', width: 140 },
    {
      title: 'Action',
      width: 120,
      render: (_, record) => <Button onClick={() => openDraft(record.draft_id)}>Review</Button>
    }
  ]

  const promotedColumns: ColumnsType<RAGDraftSummary> = [
    {
      title: 'Title',
      dataIndex: 'title',
      render: renderDraftTitle
    },
    { title: 'City', dataIndex: 'city', width: 130 },
    { title: 'Reviewer', dataIndex: 'reviewer', width: 140 },
    {
      title: 'Source URL',
      dataIndex: 'source_url',
      width: 240,
      render: (value: string) =>
        value ? (
          <a href={value} target="_blank" rel="noreferrer">
            Source
          </a>
        ) : (
          <Text type="secondary">Not provided</Text>
        )
    },
    {
      title: 'Action',
      width: 100,
      render: (_, record) => <Button onClick={() => openDraft(record.draft_id)}>View</Button>
    }
  ]

  return (
    <div className="rag-ingestion-page">
      <section className="page-heading">
        <Text className="section-kicker">Local admin workflow</Text>
        <Title level={1}>RAG Ingestion Console</Title>
        <Paragraph>
          Upload source-backed travel documents, edit structured draft fields, preview the JSON contract,
          approve reviewed knowledge, promote it into the corpus, and rebuild the Chroma index asynchronously.
        </Paragraph>
      </section>

      <Row gutter={[18, 18]} className="rag-stats">
        <Col xs={24} md={6}>
          <Card>
            <Statistic title="Drafts" value={drafts.length} />
          </Card>
        </Col>
        <Col xs={24} md={6}>
          <Card>
            <Statistic title="Ready to promote" value={readyToPromoteCount} />
          </Card>
        </Col>
        <Col xs={24} md={6}>
          <Card>
            <Statistic title="Promoted" value={promotedCount} />
          </Card>
        </Col>
        <Col xs={24} md={6}>
          <Card>
            <Statistic
              title="Index rebuild"
              value={job?.status || 'idle'}
              valueStyle={{ color: job ? undefined : '#64748b' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[18, 18]}>
        <Col xs={24} xl={9}>
          <Card title="Upload source" className="rag-card">
            <Alert
              type="info"
              showIcon
              className="rag-alert"
              message="Supported files: PDF, Markdown, and plain text."
              description="The uploaded file is extracted into draft knowledge only. Nothing enters the production RAG corpus until a reviewer approves and promotes it."
            />
            <Form form={uploadForm} layout="vertical" initialValues={emptyUpload} onFinish={handleUpload}>
              <Form.Item
                name="source_id"
                label="Source ID"
                rules={[{ required: true, message: 'Source ID is required' }]}
              >
                <Input placeholder="nyc-central-park-official-001" />
              </Form.Item>
              <Row gutter={12}>
                <Col span={10}>
                  <Form.Item name="country" label="Country" rules={[{ required: true }]}>
                    <Select options={[{ value: 'US', label: 'US' }, { value: 'CN', label: 'CN' }]} />
                  </Form.Item>
                </Col>
                <Col span={14}>
                  <Form.Item name="city" label="City" rules={[{ required: true }]}>
                    <Input placeholder="New York" />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="title" label="Title" rules={[{ required: true }]}>
                <Input placeholder="Official Central Park visitor guide" />
              </Form.Item>
              <Form.Item name="source_url" label="Source URL" rules={[{ required: true, type: 'url' }]}>
                <Input placeholder="https://..." />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item name="source_type" label="Source type" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="language" label="Language" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="theme" label="Themes">
                <Select mode="tags" tokenSeparators={[',']} placeholder="parks, family, skyline" />
              </Form.Item>
              <Form.Item name="poi_names" label="POI names">
                <Select mode="tags" tokenSeparators={[',']} placeholder="Central Park, Bethesda Terrace" />
              </Form.Item>
              <Form.Item name="best_for" label="Best for">
                <Select mode="tags" tokenSeparators={[',']} placeholder="families, first-time visitors" />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item name="district" label="District">
                    <Input placeholder="Manhattan" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="recommended_duration" label="Recommended duration">
                    <Input placeholder="2-3 hours" />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item label="Source file" required>
                <Upload
                  accept=".pdf,.md,.markdown,.txt"
                  maxCount={1}
                  fileList={fileList}
                  beforeUpload={() => false}
                  onChange={({ fileList }) => setFileList(fileList)}
                >
                  <Button icon={<UploadOutlined />}>Choose file</Button>
                </Upload>
              </Form.Item>
              <Button type="primary" htmlType="submit" loading={uploading} block>
                Generate draft
              </Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={15}>
          <Card
            title="Review Drafts"
            className="rag-card"
            extra={
              <Space wrap>
                <Button onClick={loadDrafts}>Refresh</Button>
                <Button onClick={promoteApproved} loading={promoting} disabled={readyToPromoteCount === 0}>
                  Promote approved
                </Button>
              </Space>
            }
          >
            <Table
              rowKey="draft_id"
              loading={loadingDrafts}
              columns={reviewColumns}
              dataSource={reviewDrafts}
              pagination={{ pageSize: 6 }}
              scroll={{ x: 720 }}
              locale={{ emptyText: 'No drafts awaiting review or promotion' }}
            />
          </Card>

          <Card
            title="Promoted Corpus"
            className="rag-card"
            extra={
              <Button type="primary" onClick={startRebuild} disabled={promotedCount === 0}>
                Rebuild index
              </Button>
            }
          >
            {job && (
              <Alert
                className="rag-alert"
                type={job.status === 'failed' ? 'error' : job.status === 'succeeded' ? 'success' : 'info'}
                showIcon
                message={`Rebuild job: ${job.status}`}
                description={job.error || job.message || job.job_id}
              />
            )}
            <Table
              rowKey="draft_id"
              loading={loadingDrafts}
              columns={promotedColumns}
              dataSource={promotedDrafts}
              pagination={{ pageSize: 6 }}
              scroll={{ x: 760 }}
              locale={{ emptyText: 'No promoted corpus documents yet' }}
            />
          </Card>
        </Col>
      </Row>

      <Drawer
        title={selectedDetail?.title || 'Review draft'}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        width={820}
        destroyOnClose={false}
        extra={
          <Space>
            <Button onClick={applyAiPrefill} loading={prefilling} disabled={!selectedDetail}>
              AI Prefill Fields
            </Button>
            <Button onClick={previewCurrentJson} disabled={!selectedDetail}>
              Preview JSON
            </Button>
            <Button onClick={saveDraft} loading={saving} disabled={!selectedDetail}>
              Save Draft
            </Button>
            <Button type="primary" onClick={approveDraft} loading={saving} disabled={!selectedDetail}>
              Approve
            </Button>
          </Space>
        }
      >
        {selectedDetail && (
          <div className="draft-editor-grid">
            <Card size="small" title="Draft metadata">
              <Descriptions size="small" column={1}>
                <Descriptions.Item label="Doc ID">{selectedDetail.doc_id}</Descriptions.Item>
                <Descriptions.Item label="Corpus">
                  <Tag color={selectedDetail.promoted ? 'green' : 'default'}>
                    {selectedDetail.promoted ? 'promoted' : 'not promoted'}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Path">{selectedDetail.updated_path}</Descriptions.Item>
                <Descriptions.Item label="Fetched">{selectedDetail.fetched_at}</Descriptions.Item>
              </Descriptions>
            </Card>

            {aiPrefill && (
              <Card
                size="small"
                title="AI Evidence"
                extra={
                  <Text type="secondary">
                    Used {aiPrefill.used_char_count.toLocaleString()} / {aiPrefill.source_char_count.toLocaleString()} chars
                  </Text>
                }
              >
                <Alert
                  className="rag-alert"
                  type="warning"
                  showIcon
                  message="Review before saving"
                  description="AI suggestions are only applied to the unsaved form state. They are not saved, approved, promoted, or indexed until you explicitly complete those steps."
                />
                {aiPrefill.warnings.length > 0 && (
                  <div className="ai-prefill-section">
                    <Text strong>Warnings</Text>
                    <ul>
                      {aiPrefill.warnings.map((warning, index) => (
                        <li key={`${warning}-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {aiPrefill.field_evidence.length > 0 ? (
                  <div className="ai-evidence-list">
                    {aiPrefill.field_evidence.map((item, index) => (
                      <div className="ai-evidence-item" key={`${item.field}-${index}`}>
                        <Tag color="teal">{item.field}</Tag>
                        <Text strong>{item.suggestion}</Text>
                        <Paragraph className="ai-evidence-quote">{item.evidence}</Paragraph>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Text type="secondary">No field-level evidence returned.</Text>
                )}
              </Card>
            )}

            <Form form={draftForm} layout="vertical">
              <Form.Item name="doc_id" label="Doc ID" rules={[{ required: true }]}>
                <Input disabled />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item name="country" label="Country" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="city" label="City" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="title" label="Title" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
              <Form.Item name="source_url" label="Source URL" rules={[{ required: true, type: 'url' }]}>
                <Input />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item name="source_type" label="Source type" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="language" label="Language" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item name="district" label="District">
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="recommended_duration" label="Recommended duration">
                    <Input />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="theme" label="Themes">
                <Select mode="tags" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item name="poi_names" label="POI names">
                <Select mode="tags" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item name="best_for" label="Best for">
                <Select mode="tags" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item name="seasonality" label="Seasonality">
                <Select mode="tags" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item name="transport_advice" label="Transport advice">
                <Select mode="tags" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item name="planning_tips" label="Planning tips">
                <Select mode="tags" tokenSeparators={[',']} />
              </Form.Item>
              <Form.Item
                name="content"
                label="Knowledge content indexed by RAG"
                extra="This is the curated text promoted into the knowledge corpus. The extracted text below is source evidence and is not indexed unless you copy or summarize it here."
                rules={[{ required: true }]}
              >
                <TextArea rows={10} />
              </Form.Item>
              <Space className="content-fill-button" wrap>
                <Button onClick={() => fillContentFromExtractedText(8000)}>
                  Fill first 8k chars
                </Button>
                <Button onClick={() => fillContentFromExtractedText()}>
                  Fill full extracted text
                </Button>
                <Text type="secondary">
                  Current content: {watchedContent.length.toLocaleString()} chars
                </Text>
              </Space>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item name="review_status" label="Review status">
                    <Select
                      disabled
                      options={[
                        { value: 'draft', label: 'draft' },
                        { value: 'approved', label: 'approved' },
                        { value: 'rejected', label: 'rejected' }
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="reviewer" label="Reviewer">
                    <Input placeholder="local-admin" />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="review_notes" label="Review notes">
                <TextArea rows={3} />
              </Form.Item>
              <Form.Item name="last_verified_at" label="Last verified date" rules={[{ required: true }]}>
                <Input placeholder="2026-06-16" />
              </Form.Item>
              <Form.Item name="source_id" hidden>
                <Input />
              </Form.Item>
              <Form.Item name="raw_html_path" hidden>
                <Input />
              </Form.Item>
              <Form.Item name="raw_text_path" hidden>
                <Input />
              </Form.Item>
              <Form.Item name="fetched_at" hidden>
                <Input />
              </Form.Item>
            </Form>

            <Card
              size="small"
              title="Extracted text preview"
              extra={<Text type="secondary">Source evidence, not directly indexed</Text>}
            >
              <pre className="extracted-text-preview">{selectedDetail.extracted_text || 'No extracted text available.'}</pre>
            </Card>
          </div>
        )}
      </Drawer>

      <Modal
        title="Preview JSON"
        open={previewOpen}
        onCancel={() => setPreviewOpen(false)}
        footer={<Button onClick={() => setPreviewOpen(false)}>Close</Button>}
        width={860}
      >
        <Alert
          className="rag-alert"
          type="warning"
          showIcon
          message="Preview only"
          description="This JSON is generated from the current unsaved form state. Closing this modal does not save or approve anything."
        />
        <pre className="json-preview">{previewJson}</pre>
      </Modal>
    </div>
  )
}
