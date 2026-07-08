import { DeleteOutlined } from '@ant-design/icons'
import { Button, Card, Checkbox, Col, DatePicker, Form, Input, message, Progress, Row, Select, Space, Tag } from 'antd'
import type { Dayjs } from 'dayjs'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { clearTripMemory, generateTripPlan } from '@/services/api'
import type { TripFormData } from '@/types'

const PROFILE_STORAGE_KEY = 'trip_profile_id'
const CONVERSATION_STORAGE_KEY = 'trip_conversation_id'

const preferenceOptions = [
  'Museums',
  'Food',
  'Architecture',
  'Parks',
  'Night views',
  'Shopping',
  'Relaxed pace'
]

type FormValues = {
  city: string
  date_range: [Dayjs, Dayjs]
  transportation: string
  accommodation: string
  preferences: string[]
  free_text_input?: string
}

function createProfileId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `trip_${crypto.randomUUID()}`
  }
  return `trip_${Date.now()}_${Math.random().toString(36).slice(2, 12)}`
}

function getOrCreateProfileId() {
  const existing = localStorage.getItem(PROFILE_STORAGE_KEY)
  if (existing) return existing
  const profileId = createProfileId()
  localStorage.setItem(PROFILE_STORAGE_KEY, profileId)
  return profileId
}

export default function Home() {
  const [form] = Form.useForm<FormValues>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('')
  const dates = Form.useWatch('date_range', form)

  const travelDays = useMemo(() => {
    if (!dates?.[0] || !dates?.[1]) return 1
    return Math.max(1, dates[1].diff(dates[0], 'day') + 1)
  }, [dates])

  useEffect(() => {
    if (!loading) return
    const interval = window.setInterval(() => {
      setProgress((current) => {
        const next = Math.min(90, current + 10)
        if (next <= 30) setStatus('Retrieving attractions...')
        else if (next <= 50) setStatus('Checking weather...')
        else if (next <= 70) setStatus('Finding hotels...')
        else setStatus('Generating itinerary...')
        return next
      })
    }, 550)
    return () => window.clearInterval(interval)
  }, [loading])

  async function handleSubmit(values: FormValues) {
    const [start, end] = values.date_range
    const days = end.diff(start, 'day') + 1
    if (days <= 0 || days > 30) {
      message.error('Travel dates must cover 1-30 days.')
      return
    }

    setLoading(true)
    setProgress(5)
    setStatus('Initializing graph workflow...')
    try {
      const profileId = getOrCreateProfileId()
      const requestData: TripFormData = {
        city: values.city,
        start_date: start.format('YYYY-MM-DD'),
        end_date: end.format('YYYY-MM-DD'),
        travel_days: days,
        transportation: values.transportation,
        accommodation: values.accommodation,
        preferences: values.preferences.map(String),
        country_code: 'US',
        free_text_input: values.free_text_input || '',
        profile_id: profileId,
        conversation_id: sessionStorage.getItem(CONVERSATION_STORAGE_KEY) || undefined
      }

      const response = await generateTripPlan(requestData)
      setProgress(100)
      setStatus('Complete')

      if (response.success && response.data) {
        sessionStorage.setItem('tripPlan', JSON.stringify(response.data))
        if (response.validation_summary) {
          sessionStorage.setItem('tripValidationSummary', JSON.stringify(response.validation_summary))
        } else {
          sessionStorage.removeItem('tripValidationSummary')
        }
        if (response.conversation_id) {
          sessionStorage.setItem(CONVERSATION_STORAGE_KEY, response.conversation_id)
        }
        if (response.memory_summary) {
          sessionStorage.setItem('tripMemorySummary', response.memory_summary)
        } else {
          sessionStorage.removeItem('tripMemorySummary')
        }
        if (response.memory_applied && response.memory_profile) {
          sessionStorage.setItem('tripMemoryApplied', 'true')
          sessionStorage.setItem('tripMemoryProfile', JSON.stringify(response.memory_profile))
        } else {
          sessionStorage.removeItem('tripMemoryApplied')
          sessionStorage.removeItem('tripMemoryProfile')
        }
        message.success('Trip plan generated.')
        window.setTimeout(() => navigate('/result'), 450)
      } else {
        message.error(response.message || 'Trip planning failed.')
      }
    } catch (error: any) {
      message.error(error.message || 'Trip planning failed.')
    } finally {
      window.setTimeout(() => {
        setLoading(false)
        setProgress(0)
        setStatus('')
      }, 700)
    }
  }

  async function handleClearMemory() {
    const profileId = localStorage.getItem(PROFILE_STORAGE_KEY)
    if (!profileId) {
      message.info('No local preference memory exists.')
      return
    }
    try {
      await clearTripMemory(profileId)
      localStorage.removeItem(PROFILE_STORAGE_KEY)
      sessionStorage.removeItem(CONVERSATION_STORAGE_KEY)
      sessionStorage.removeItem('tripMemorySummary')
      sessionStorage.removeItem('tripMemoryApplied')
      sessionStorage.removeItem('tripMemoryProfile')
      message.success('Preference memory cleared.')
    } catch (error: any) {
      message.error(error.message || 'Failed to clear preference memory.')
    }
  }

  return (
    <main className="home-page">
      <section className="hero">
        <Tag color="blue">Grounded planning</Tag>
        <h1>Plan a grounded city itinerary</h1>
        <p>
          Retrieve places, hotels, weather, and travel context, then validate the itinerary before it reaches the user.
        </p>
      </section>

      <Card className="planner-card" variant="borderless">
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            city: 'New York',
            transportation: 'Public transit',
            accommodation: 'Mid-range hotel',
            preferences: ['Museums', 'Food']
          }}
          onFinish={handleSubmit}
        >
          <Row gutter={[20, 4]}>
            <Col xs={24} lg={8}>
              <Form.Item name="city" label="Destination city" rules={[{ required: true, message: 'Enter a city.' }]}>
                <Input size="large" placeholder="New York" />
              </Form.Item>
            </Col>
            <Col xs={24} lg={10}>
              <Form.Item name="date_range" label="Travel dates" rules={[{ required: true, message: 'Select dates.' }]}>
                <DatePicker.RangePicker size="large" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} lg={6}>
              <Form.Item label="Travel days">
                <div className="days-pill">{travelDays} day{travelDays === 1 ? '' : 's'}</div>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={[20, 4]}>
            <Col xs={24} md={8}>
              <Form.Item name="transportation" label="Transportation">
                <Select
                  size="large"
                  options={[
                    { value: 'Public transit', label: 'Public transit' },
                    { value: 'Walking', label: 'Walking' },
                    { value: 'Rideshare / taxi', label: 'Rideshare / taxi' },
                    { value: 'Driving', label: 'Driving' }
                  ]}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="accommodation" label="Accommodation">
                <Select
                  size="large"
                  options={[
                    { value: 'Budget hotel', label: 'Budget hotel' },
                    { value: 'Mid-range hotel', label: 'Mid-range hotel' },
                    { value: 'Luxury hotel', label: 'Luxury hotel' },
                    { value: 'Boutique hotel', label: 'Boutique hotel' }
                  ]}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="preferences" label="Preferences">
                <Checkbox.Group options={preferenceOptions} className="preference-grid" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="free_text_input" label="Additional requirements">
            <Input.TextArea
              rows={3}
              placeholder="Example: keep the pace relaxed, avoid long transfers, include one museum per day."
            />
          </Form.Item>

          {loading && (
            <div className="progress-panel">
              <Progress percent={progress} status="active" />
              <span>{status}</span>
            </div>
          )}

          <Space wrap>
            <Button type="primary" htmlType="submit" size="large" loading={loading}>
              Generate itinerary
            </Button>
            <Button icon={<DeleteOutlined />} onClick={handleClearMemory} disabled={loading}>
              Clear preference memory
            </Button>
          </Space>
        </Form>
      </Card>
    </main>
  )
}
