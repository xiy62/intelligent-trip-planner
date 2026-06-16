import { ArrowLeftOutlined, DownOutlined, EnvironmentOutlined, GlobalOutlined } from '@ant-design/icons'
import { APIProvider, Map, Marker, useMap } from '@vis.gl/react-google-maps'
import { Button, Card, Collapse, Descriptions, Dropdown, Empty, List, message, Space, Tag } from 'antd'
import type { MenuProps } from 'antd'
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Attraction, DayPlan, MemoryProfile, TripPlan } from '@/types'

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || ''
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

type MapAttraction = Attraction & {
  dayIndex: number
  attractionIndex: number
}

function collectAttractions(plan: TripPlan): MapAttraction[] {
  return plan.days.flatMap((day) =>
    day.attractions
      .filter((attraction) => attraction.location)
      .map((attraction, attractionIndex) => ({
        ...attraction,
        dayIndex: day.day_index,
        attractionIndex
      }))
  )
}

function centerFor(attractions: MapAttraction[]) {
  if (attractions.length > 0) {
    return {
      lat: attractions[0].location.latitude,
      lng: attractions[0].location.longitude
    }
  }
  return { lat: 40.7128, lng: -74.006 }
}

function DayPolylines({ days }: { days: DayPlan[] }) {
  const map = useMap()

  useEffect(() => {
    if (!map || !(window as any).google?.maps) return
    const polylines = days
      .map((day) => {
        const path = day.attractions
          .filter((attraction) => attraction.location)
          .map((attraction) => ({
            lat: attraction.location.latitude,
            lng: attraction.location.longitude
          }))
        if (path.length < 2) return null
        return new (window as any).google.maps.Polyline({
          path,
          geodesic: true,
          strokeColor: '#2563eb',
          strokeOpacity: 0.75,
          strokeWeight: 3,
          map
        })
      })
      .filter(Boolean)

    return () => {
      polylines.forEach((polyline: any) => polyline?.setMap(null))
    }
  }, [days, map])

  return null
}

function TripMap({ plan }: { plan: TripPlan }) {
  const attractions = useMemo(() => collectAttractions(plan), [plan])
  const center = centerFor(attractions)

  if (!GOOGLE_MAPS_API_KEY) {
    return (
      <div className="map-fallback">
        <h3>Map preview unavailable</h3>
        <p>Set VITE_GOOGLE_MAPS_API_KEY to render Google Maps markers for this itinerary.</p>
      </div>
    )
  }

  return (
    <APIProvider apiKey={GOOGLE_MAPS_API_KEY} language="en" region="US">
      <Map defaultZoom={12} defaultCenter={center} gestureHandling="greedy" disableDefaultUI={false}>
        {attractions.map((attraction) => (
          <Marker
            key={`${attraction.dayIndex}-${attraction.attractionIndex}-${attraction.name}`}
            position={{ lat: attraction.location.latitude, lng: attraction.location.longitude }}
            title={attraction.name}
          />
        ))}
        <DayPolylines days={plan.days} />
      </Map>
    </APIProvider>
  )
}

function readStoredPlan(): TripPlan | null {
  const raw = sessionStorage.getItem('tripPlan')
  if (!raw) return null
  try {
    return JSON.parse(raw) as TripPlan
  } catch {
    return null
  }
}

function readMemoryProfile(): MemoryProfile | null {
  const raw = sessionStorage.getItem('tripMemoryProfile')
  if (!raw) return null
  try {
    return JSON.parse(raw) as MemoryProfile
  } catch {
    return null
  }
}

function isForecastUnavailable(item: TripPlan['weather_info'][number]) {
  const dayUnknown = !item.day_weather || item.day_weather.toLowerCase() === 'unknown'
  const nightUnknown = !item.night_weather || item.night_weather.toLowerCase() === 'unknown'
  return dayUnknown && nightUnknown && Number(item.day_temp || 0) === 0 && Number(item.night_temp || 0) === 0
}

function resolveMediaUrl(url?: string | null) {
  if (!url) return ''
  if (url.startsWith('/api/')) return `${API_BASE_URL}${url}`
  return url
}

function googleMapsSearchUrl(query: string, city: string) {
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${query} ${city}`)}`
}

export default function Result() {
  const navigate = useNavigate()
  const exportRef = useRef<HTMLDivElement>(null)
  const [tripPlan, setTripPlan] = useState<TripPlan | null>(() => readStoredPlan())
  const [memoryProfile] = useState<MemoryProfile | null>(() => readMemoryProfile())
  const [memoryApplied] = useState(() => sessionStorage.getItem('tripMemoryApplied') === 'true')
  const [editMode, setEditMode] = useState(false)

  function updatePlan(nextPlan: TripPlan) {
    setTripPlan(nextPlan)
    sessionStorage.setItem('tripPlan', JSON.stringify(nextPlan))
  }

  function deleteAttraction(dayIndex: number, attractionIndex: number) {
    if (!tripPlan) return
    const next = structuredClone(tripPlan)
    const day = next.days.find((item) => item.day_index === dayIndex)
    if (!day || day.attractions.length <= 1) {
      message.warning('Each day should keep at least one attraction.')
      return
    }
    day.attractions.splice(attractionIndex, 1)
    updatePlan(next)
  }

  function moveAttraction(dayIndex: number, attractionIndex: number, direction: 'up' | 'down') {
    if (!tripPlan) return
    const next = structuredClone(tripPlan)
    const day = next.days.find((item) => item.day_index === dayIndex)
    if (!day) return
    const target = direction === 'up' ? attractionIndex - 1 : attractionIndex + 1
    if (target < 0 || target >= day.attractions.length) return
    const [item] = day.attractions.splice(attractionIndex, 1)
    day.attractions.splice(target, 0, item)
    updatePlan(next)
  }

  async function exportImage() {
    if (!exportRef.current) return
    const canvas = await html2canvas(exportRef.current, { scale: 2, useCORS: true })
    const link = document.createElement('a')
    link.download = `trip-plan-${tripPlan?.city || 'itinerary'}-${Date.now()}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  }

  async function exportPdf() {
    if (!exportRef.current) return
    const canvas = await html2canvas(exportRef.current, { scale: 2, useCORS: true })
    const imgData = canvas.toDataURL('image/png')
    const pdf = new jsPDF('p', 'mm', 'a4')
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    const imgHeight = (canvas.height * pageWidth) / canvas.width
    let heightLeft = imgHeight
    let position = 0
    pdf.addImage(imgData, 'PNG', 0, position, pageWidth, imgHeight)
    heightLeft -= pageHeight
    while (heightLeft > 0) {
      position = heightLeft - imgHeight
      pdf.addPage()
      pdf.addImage(imgData, 'PNG', 0, position, pageWidth, imgHeight)
      heightLeft -= pageHeight
    }
    pdf.save(`trip-plan-${tripPlan?.city || 'itinerary'}-${Date.now()}.pdf`)
  }

  const exportItems: MenuProps['items'] = [
    { key: 'image', label: 'Export as image', onClick: exportImage },
    { key: 'pdf', label: 'Export as PDF', onClick: exportPdf }
  ]

  if (!tripPlan) {
    return (
      <Empty className="empty-state" description="No trip plan found.">
        <Button type="primary" onClick={() => navigate('/')}>
          Create a trip plan
        </Button>
      </Empty>
    )
  }

  return (
    <main className="result-page">
      <div className="result-actions">
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/')}>
          Back to planner
        </Button>
        <Space wrap>
          <Button onClick={() => setEditMode((value) => !value)}>
            {editMode ? 'Finish editing' : 'Edit itinerary'}
          </Button>
          <Dropdown menu={{ items: exportItems }}>
            <Button>
              Export <DownOutlined />
            </Button>
          </Dropdown>
        </Space>
      </div>

      <div ref={exportRef} className="result-grid">
        <section className="result-main">
          <Card variant="borderless" className="overview-card">
            <div className="eyebrow-row">
              <Tag color="cyan">{tripPlan.city}</Tag>
              <Tag color="green">{tripPlan.days.length} day{tripPlan.days.length === 1 ? '' : 's'}</Tag>
              <Tag color="blue">Validated itinerary</Tag>
            </div>
            <h1>{tripPlan.city} itinerary</h1>
            <p className="date-range">{tripPlan.start_date} to {tripPlan.end_date}</p>
            <p>{tripPlan.overall_suggestions}</p>
          </Card>

          {memoryApplied && memoryProfile && (
            <Card variant="borderless" title="Preference memory applied">
              <Space wrap>
                <Tag color="blue">{memoryProfile.transportation}</Tag>
                <Tag color="purple">{memoryProfile.accommodation}</Tag>
                {memoryProfile.preferences.map((preference) => (
                  <Tag color="green" key={preference}>{preference}</Tag>
                ))}
                {memoryProfile.recent_cities.map((city) => (
                  <Tag color="cyan" key={city}>{city}</Tag>
                ))}
              </Space>
            </Card>
          )}

          {tripPlan.budget && (
            <Card variant="borderless" className="budget-card" title="Budget estimate">
              <div className="budget-grid">
                <span>Attractions</span><strong>${tripPlan.budget.total_attractions}</strong>
                <span>Hotels</span><strong>${tripPlan.budget.total_hotels}</strong>
                <span>Meals</span><strong>${tripPlan.budget.total_meals}</strong>
                <span>Transportation</span><strong>${tripPlan.budget.total_transportation}</strong>
                <span>Total</span><strong>${tripPlan.budget.total}</strong>
              </div>
            </Card>
          )}

          <Card variant="borderless" title="Daily itinerary">
            <Collapse
              defaultActiveKey={['0']}
              items={tripPlan.days.map((day) => ({
                key: String(day.day_index),
                label: `Day ${day.day_index + 1} · ${day.date}`,
                children: (
                  <div className="day-panel">
                    <Descriptions column={1} size="small" bordered>
                      <Descriptions.Item label="Description">{day.description}</Descriptions.Item>
                      <Descriptions.Item label="Transportation">{day.transportation}</Descriptions.Item>
                      <Descriptions.Item label="Accommodation">{day.accommodation}</Descriptions.Item>
                    </Descriptions>

                    <h3>Attractions</h3>
                    <List
                      grid={{ gutter: 16, xs: 1, md: 2 }}
                      dataSource={day.attractions}
                      renderItem={(attraction, attractionIndex) => (
                        <List.Item>
                          <Card
                            size="small"
                            className="place-card"
                            title={attraction.name}
                            extra={
                              editMode ? (
                                <Space>
                                  <Button size="small" onClick={() => moveAttraction(day.day_index, attractionIndex, 'up')} disabled={attractionIndex === 0}>Up</Button>
                                  <Button size="small" onClick={() => moveAttraction(day.day_index, attractionIndex, 'down')} disabled={attractionIndex === day.attractions.length - 1}>Down</Button>
                                  <Button size="small" danger onClick={() => deleteAttraction(day.day_index, attractionIndex)}>Delete</Button>
                                </Space>
                              ) : null
                            }
                          >
                            <div className="place-media-row">
                              {attraction.image_url && (
                                <img
                                  className="place-image"
                                  src={resolveMediaUrl(attraction.image_url)}
                                  alt={attraction.name}
                                  loading="lazy"
                                />
                              )}
                              <div className="place-copy">
                                <p className="detail-line"><strong>Address:</strong> {attraction.address}</p>
                                <p className="detail-line"><strong>Duration:</strong> {attraction.visit_duration} minutes</p>
                                <p>{attraction.description}</p>
                                {attraction.rating && <p><strong>Rating:</strong> {attraction.rating}</p>}
                                <Space wrap className="place-actions">
                                  <Button
                                    size="small"
                                    icon={<EnvironmentOutlined />}
                                    href={attraction.maps_url || googleMapsSearchUrl(attraction.name, tripPlan.city)}
                                    target="_blank"
                                  >
                                    Maps
                                  </Button>
                                  {attraction.website_url && (
                                    <Button size="small" icon={<GlobalOutlined />} href={attraction.website_url} target="_blank">
                                      Website
                                    </Button>
                                  )}
                                </Space>
                              </div>
                            </div>
                          </Card>
                        </List.Item>
                      )}
                    />

                    {day.hotel && (
                      <>
                        <h3>Hotel</h3>
                        <Card size="small">
                          <div className="place-media-row hotel-media-row">
                            {day.hotel.image_url && (
                              <img
                                className="place-image"
                                src={resolveMediaUrl(day.hotel.image_url)}
                                alt={day.hotel.name}
                                loading="lazy"
                              />
                            )}
                            <div className="place-copy">
                              <Descriptions column={1} size="small">
                                <Descriptions.Item label="Name">{day.hotel.name}</Descriptions.Item>
                                <Descriptions.Item label="Address">{day.hotel.address}</Descriptions.Item>
                                <Descriptions.Item label="Type">{day.hotel.type || 'Hotel'}</Descriptions.Item>
                                <Descriptions.Item label="Price">{day.hotel.price_range || `$${day.hotel.estimated_cost || 0}`}</Descriptions.Item>
                              </Descriptions>
                              <Space wrap className="place-actions">
                                <Button
                                  size="small"
                                  icon={<EnvironmentOutlined />}
                                  href={day.hotel.maps_url || googleMapsSearchUrl(day.hotel.name, tripPlan.city)}
                                  target="_blank"
                                >
                                  Maps
                                </Button>
                                {day.hotel.website_url && (
                                  <Button size="small" icon={<GlobalOutlined />} href={day.hotel.website_url} target="_blank">
                                    Website
                                  </Button>
                                )}
                              </Space>
                            </div>
                          </div>
                        </Card>
                      </>
                    )}

                    <h3>Meals</h3>
                    <List
                      size="small"
                      dataSource={day.meals}
                      renderItem={(meal) => (
                        <List.Item>
                          <div className="meal-row">
                            {meal.image_url && (
                              <img
                                className="meal-image"
                                src={resolveMediaUrl(meal.image_url)}
                                alt={meal.name}
                                loading="lazy"
                              />
                            )}
                            <div>
                              <strong>{meal.type}:</strong>&nbsp;{meal.name}
                              {meal.description ? ` - ${meal.description}` : ''}
                            </div>
                            <Button
                              size="small"
                              icon={<EnvironmentOutlined />}
                              href={meal.maps_url || googleMapsSearchUrl(`${meal.name} ${meal.address || ''}`, tripPlan.city)}
                              target="_blank"
                            >
                              Maps
                            </Button>
                            {meal.website_url && (
                              <Button size="small" icon={<GlobalOutlined />} href={meal.website_url} target="_blank">
                                Website
                              </Button>
                            )}
                          </div>
                        </List.Item>
                      )}
                    />
                  </div>
                )
              }))}
            />
          </Card>

          <Card variant="borderless" title="Weather">
            <List
              grid={{ gutter: 16, xs: 1, md: 3 }}
              dataSource={tripPlan.weather_info || []}
              renderItem={(item) => (
                <List.Item>
                  <Card size="small" className={isForecastUnavailable(item) ? 'weather-card unavailable' : 'weather-card'}>
                    <strong>{item.date}</strong>
                    {isForecastUnavailable(item) ? (
                      <>
                        <p className="weather-unavailable">Forecast unavailable</p>
                        <p className="muted">This date is outside the reliable forecast window or the provider returned incomplete data.</p>
                      </>
                    ) : (
                      <>
                        <p>Day: {item.day_weather} · {item.day_temp}°C</p>
                        <p>Night: {item.night_weather} · {item.night_temp}°C</p>
                        <p>Wind: {item.wind_direction} {item.wind_power}</p>
                      </>
                    )}
                  </Card>
                </List.Item>
              )}
            />
          </Card>
        </section>

        <aside className="result-map">
          <Card variant="borderless" title="Itinerary map">
            <div className="map-container">
              <TripMap plan={tripPlan} />
            </div>
          </Card>
        </aside>
      </div>
    </main>
  )
}
