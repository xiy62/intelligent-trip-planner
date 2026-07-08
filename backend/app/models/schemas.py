"""Public API and itinerary data models."""

from datetime import datetime
import re
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Request models

class TripRequest(BaseModel):
    """Structured trip-planning request."""
    city: str = Field(..., min_length=1, max_length=100, description="Destination city", example="New York")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format", example="2026-07-01")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format", example="2026-07-03")
    travel_days: int = Field(..., description="Number of travel days", ge=1, le=30, example=3)
    transportation: str = Field(
        ..., min_length=1, max_length=100, description="Preferred transportation", example="Public transit"
    )
    accommodation: str = Field(
        ..., min_length=1, max_length=100, description="Accommodation preference", example="Mid-range hotel"
    )
    preferences: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Travel preference labels",
        example=["Museums", "Food"],
    )
    country_code: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description="2-letter ISO-style country code used for map search region bias",
        example="US",
    )
    free_text_input: Optional[str] = Field(
        default="", max_length=1000, description="Additional user requirements", example="Keep the itinerary relaxed and include museums."
    )
    profile_id: Optional[str] = Field(default=None, max_length=128, description="Anonymous preference-memory profile ID")
    conversation_id: Optional[str] = Field(default=None, max_length=128, description="Trip-planning conversation ID")

    @field_validator("city", "transportation", "accommodation", mode="before")
    @classmethod
    def normalize_required_text(cls, value):
        """Trim required text fields and reject blank values."""
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("free_text_input", "profile_id", "conversation_id", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        """Trim optional text while preserving omitted values."""
        if value is None:
            return value
        return value.strip() if isinstance(value, str) else value

    @field_validator("country_code", mode="before")
    @classmethod
    def normalize_country_code(cls, value):
        """Normalize 2-letter country codes used for map region bias."""
        if value is None:
            return "US"
        if not isinstance(value, str):
            return value
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("country_code must not be blank")
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("country_code must be a 2-letter country code")
        return normalized

    @field_validator("preferences", mode="before")
    @classmethod
    def normalize_preferences(cls, value):
        """Trim, deduplicate, and validate preference labels."""
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        normalized: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("preference labels must be strings")
            label = item.strip()
            if not label:
                raise ValueError("preference labels must not be blank")
            if len(label) > 50:
                raise ValueError("preference labels must be at most 50 characters")
            if label not in normalized:
                normalized.append(label)
        return normalized

    @model_validator(mode="after")
    def validate_date_range(self):
        """Require exact ISO dates and inclusive range consistency."""
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        if not date_pattern.fullmatch(self.start_date) or not date_pattern.fullmatch(self.end_date):
            raise ValueError("start_date and end_date must use YYYY-MM-DD format")
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("start_date and end_date must use YYYY-MM-DD format") from exc
        if end < start:
            raise ValueError("end_date must be on or after start_date")
        inclusive_days = (end - start).days + 1
        if inclusive_days != self.travel_days:
            raise ValueError(
                f"travel_days must equal the inclusive date range ({inclusive_days})"
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "city": "New York",
                "start_date": "2026-07-01",
                "end_date": "2026-07-03",
                "travel_days": 3,
                "transportation": "Public transit",
                "accommodation": "Mid-range hotel",
                "preferences": ["Museums", "Food"],
                "country_code": "US",
                "free_text_input": "Keep the itinerary relaxed."
            }
        }


class POISearchRequest(BaseModel):
    """POI search request."""
    keywords: str = Field(..., description="Search keywords", example="museum")
    city: str = Field(..., description="City", example="New York")
    citylimit: bool = Field(default=True, description="Whether to restrict results to the requested city")
    country_code: str = Field(default="US", min_length=2, max_length=2, description="2-letter region bias code")


class RouteRequest(BaseModel):
    """Route-planning request."""
    origin_address: str = Field(..., description="Origin address", example="Times Square")
    destination_address: str = Field(..., description="Destination address", example="Central Park")
    origin_city: Optional[str] = Field(default=None, description="Origin city")
    destination_city: Optional[str] = Field(default=None, description="Destination city")
    route_type: str = Field(default="walking", description="Route type: walking/driving/transit/bicycling")


# Response and domain models

class Location(BaseModel):
    """Geographic coordinates."""
    longitude: float = Field(..., description="Longitude")
    latitude: float = Field(..., description="Latitude")


class Attraction(BaseModel):
    """Attraction recommendation."""
    name: str = Field(..., description="Attraction name")
    address: str = Field(..., description="Address")
    location: Location = Field(..., description="Coordinates")
    visit_duration: int = Field(..., description="Recommended visit duration in minutes")
    description: str = Field(..., description="Attraction description")
    category: Optional[str] = Field(default="Attraction", description="Attraction category")
    rating: Optional[float] = Field(default=None, description="Rating")
    photos: Optional[List[str]] = Field(default_factory=list, description="Photo URL list")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    maps_url: Optional[str] = Field(default=None, description="Map provider URL")
    website_url: Optional[str] = Field(default=None, description="Official website URL")
    ticket_price: int = Field(default=0, description="Ticket price")


class Meal(BaseModel):
    """Meal recommendation."""
    type: str = Field(..., description="Meal type: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="Meal recommendation name")
    address: Optional[str] = Field(default=None, description="Address")
    location: Optional[Location] = Field(default=None, description="Coordinates")
    description: Optional[str] = Field(default=None, description="Description")
    estimated_cost: int = Field(default=0, description="Estimated cost")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    maps_url: Optional[str] = Field(default=None, description="Map provider URL")
    website_url: Optional[str] = Field(default=None, description="Official website URL")
    poi_id: Optional[str] = Field(default="", description="POI ID")


class Hotel(BaseModel):
    """Hotel recommendation."""
    name: str = Field(..., description="Hotel name")
    address: str = Field(default="", description="Hotel address")
    location: Optional[Location] = Field(default=None, description="Hotel coordinates")
    price_range: str = Field(default="", description="Price range")
    rating: str = Field(default="", description="Rating")
    distance: str = Field(default="", description="Distance to attractions")
    type: str = Field(default="", description="Hotel type")
    estimated_cost: int = Field(default=0, description="Estimated nightly cost")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    maps_url: Optional[str] = Field(default=None, description="Map provider URL")
    website_url: Optional[str] = Field(default=None, description="Official website URL")
    poi_id: Optional[str] = Field(default="", description="POI ID")


class DayPlan(BaseModel):
    """One day of an itinerary."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    day_index: int = Field(..., description="Zero-based day index")
    description: str = Field(..., description="Daily itinerary description")
    transportation: str = Field(..., description="Transportation")
    accommodation: str = Field(..., description="Accommodation preference")
    hotel: Optional[Hotel] = Field(default=None, description="Recommended hotel")
    attractions: List[Attraction] = Field(default=[], description="Attraction list")
    meals: List[Meal] = Field(default=[], description="Meal list")


class WeatherInfo(BaseModel):
    """Weather aligned to one trip date."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    day_weather: str = Field(default="", description="Daytime weather")
    night_weather: str = Field(default="", description="Nighttime weather")
    day_temp: Union[int, str] = Field(default=0, description="Daytime temperature")
    night_temp: Union[int, str] = Field(default=0, description="Nighttime temperature")
    wind_direction: str = Field(default="", description="Wind direction")
    wind_power: str = Field(default="", description="Wind speed")

    @field_validator('day_temp', 'night_temp', mode='before')
    @classmethod
    def parse_temperature(cls, v):
        """Parse temperature values and remove unit suffixes."""
        if isinstance(v, str):
            # Remove common Celsius suffixes before numeric parsing.
            v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
            try:
                return int(v)
            except ValueError:
                return 0
        return v


class Budget(BaseModel):
    """Itemized trip budget."""
    total_attractions: int = Field(default=0, description="Total attraction ticket cost")
    total_hotels: int = Field(default=0, description="Total hotel cost")
    total_meals: int = Field(default=0, description="Total meal cost")
    total_transportation: int = Field(default=0, description="Total transportation cost")
    total: int = Field(default=0, description="Total estimated cost")


class TripPlan(BaseModel):
    """Validated multi-day trip plan."""
    city: str = Field(..., description="Destination city")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    days: List[DayPlan] = Field(..., description="Daily itinerary")
    weather_info: List[WeatherInfo] = Field(default=[], description="Weather information")
    overall_suggestions: str = Field(..., description="Overall suggestions")
    budget: Optional[Budget] = Field(default=None, description="Budget information")


class MemoryPreferenceMetadata(BaseModel):
    """Frequency metadata for one remembered preference value."""
    value: str = Field(..., description="Remembered preference value")
    count: int = Field(default=1, description="Number of successful plans where this value appeared")
    last_seen_at: Optional[float] = Field(default=None, description="Unix timestamp for latest successful observation")
    source_type: str = Field(default="explicit_request", description="Where the preference came from")


class MemoryConflictExplanation(BaseModel):
    """Explanation for a memory/current-request conflict."""
    field: str = Field(..., description="Request field with a memory conflict")
    remembered_value: str = Field(..., description="Value from anonymous profile memory")
    current_value: str = Field(..., description="Explicit value from the current request")
    resolution: str = Field(default="current_request_used", description="How the conflict was resolved")
    count: int = Field(default=0, description="Observed count for the remembered value")
    last_seen_at: Optional[float] = Field(default=None, description="Latest timestamp for the remembered value")
    source_type: str = Field(default="explicit_request", description="Source type for the remembered value")
    explanation: str = Field(..., description="Human-readable conflict explanation")


class MemoryProfile(BaseModel):
    """Anonymous preference-memory summary."""
    profile_id: str = Field(..., description="Anonymous preference-memory profile ID")
    transportation: str = Field(default="", description="Previously preferred transportation")
    accommodation: str = Field(default="", description="Previously preferred accommodation")
    preferences: List[str] = Field(default_factory=list, description="Historical preference labels")
    recent_cities: List[str] = Field(default_factory=list, description="Recently planned destinations")
    preference_metadata: Dict[str, List[MemoryPreferenceMetadata]] = Field(
        default_factory=dict,
        description="Frequency and recency metadata for remembered preference values",
    )
    trip_count: int = Field(default=0, description="Number of successful memory writes")
    last_summary: str = Field(default="", description="Latest memory summary")
    created_at: Optional[float] = Field(default=None, description="Created timestamp")
    updated_at: Optional[float] = Field(default=None, description="Updated timestamp")


class ValidationSummary(BaseModel):
    """Sanitized user-facing validation summary for public trip responses."""

    validated: bool = Field(default=False, description="Whether the final itinerary passed validation without fallback")
    fallback_used: bool = Field(default=False, description="Whether a fallback itinerary was returned")
    date_coverage_passed: bool = Field(default=False, description="Whether itinerary dates cover the requested trip")
    budget_consistency_passed: bool = Field(default=False, description="Whether budget totals are internally consistent")
    grounding_score: Optional[float] = Field(default=None, description="Share of checked entities grounded to evidence")
    attribution_coverage_score: Optional[float] = Field(default=None, description="Share of checked entities with attribution evidence")
    pacing_score: Optional[float] = Field(default=None, description="Soft pacing quality score")
    route_coherence_score: Optional[float] = Field(default=None, description="Soft same-day route coherence score")
    quality_warnings: List[str] = Field(default_factory=list, description="Sanitized quality warning codes")
    grounded_entity_count: Optional[int] = Field(default=None, description="Count of checked entities with supporting evidence")
    checked_entity_count: Optional[int] = Field(default=None, description="Count of generated entities checked for evidence")
    evidence_summary: Optional[str] = Field(default=None, description="Concise user-facing explanation of evidence checks")


class TripPlanResponse(BaseModel):
    """Trip-planning API response."""
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(default="", description="Response message")
    data: Optional[TripPlan] = Field(default=None, description="Trip plan data")
    conversation_id: Optional[str] = Field(default=None, description="Backend conversation ID")
    memory_applied: bool = Field(default=False, description="Whether historical preference memory was applied")
    memory_summary: Optional[str] = Field(default=None, description="Historical preference summary applied to this request")
    memory_profile: Optional[MemoryProfile] = Field(default=None, description="Structured anonymous preference memory")
    memory_conflicts: List[MemoryConflictExplanation] = Field(
        default_factory=list,
        description="Explicit explanations when current request overrides historical memory",
    )
    validation_summary: Optional[ValidationSummary] = Field(
        default=None,
        description="Sanitized validation metadata for user-facing result display",
    )


class MemoryClearRequest(BaseModel):
    """Anonymous memory cleanup request."""
    profile_id: str = Field(..., description="Anonymous preference-memory profile ID")


class MemoryClearResponse(BaseModel):
    """Anonymous memory cleanup response."""
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(default="", description="Response message")
    profile_id: str = Field(..., description="Anonymous preference-memory profile ID")


class POIInfo(BaseModel):
    """Normalized POI record."""
    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="Name")
    type: str = Field(..., description="Type")
    address: str = Field(..., description="Address")
    location: Location = Field(..., description="Coordinates")
    tel: Optional[str] = Field(default=None, description="Phone number")
    rating: Optional[float] = Field(default=None, description="Rating")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    maps_url: Optional[str] = Field(default=None, description="Map provider URL")
    website_url: Optional[str] = Field(default=None, description="Official website URL")


class POISearchResponse(BaseModel):
    """POI search response."""
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(default="", description="Response message")
    data: List[POIInfo] = Field(default_factory=list, description="POI list")


class RouteInfo(BaseModel):
    """Normalized route result."""
    distance: float = Field(..., description="Distance in meters")
    duration: int = Field(..., description="Duration in seconds")
    route_type: str = Field(..., description="Route type")
    description: str = Field(..., description="Route description")


class RouteResponse(BaseModel):
    """Route-planning response."""
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(default="", description="Response message")
    data: Optional[RouteInfo] = Field(default=None, description="Route information")


class WeatherResponse(BaseModel):
    """Weather lookup response."""
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(default="", description="Response message")
    data: List[WeatherInfo] = Field(default_factory=list, description="Weather information")


# Error response

class ErrorResponse(BaseModel):
    """Standard API error response."""
    success: bool = Field(default=False, description="Whether the request succeeded")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
