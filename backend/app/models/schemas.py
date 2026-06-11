"""Public API and itinerary data models."""

from datetime import datetime
import re
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Request models

class TripRequest(BaseModel):
    """Structured trip-planning request."""
    city: str = Field(..., min_length=1, max_length=100, description="目的地城市", example="北京")
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", example="2025-06-01")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", example="2025-06-03")
    travel_days: int = Field(..., description="旅行天数", ge=1, le=30, example=3)
    transportation: str = Field(
        ..., min_length=1, max_length=100, description="交通方式", example="公共交通"
    )
    accommodation: str = Field(
        ..., min_length=1, max_length=100, description="住宿偏好", example="经济型酒店"
    )
    preferences: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="旅行偏好标签",
        example=["历史文化", "美食"],
    )
    free_text_input: Optional[str] = Field(
        default="", max_length=1000, description="额外要求", example="希望多安排一些博物馆"
    )
    profile_id: Optional[str] = Field(default=None, max_length=128, description="匿名设备偏好记忆ID")
    conversation_id: Optional[str] = Field(default=None, max_length=128, description="旅行规划会话ID")

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
                "city": "北京",
                "start_date": "2025-06-01",
                "end_date": "2025-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化", "美食"],
                "free_text_input": "希望多安排一些博物馆"
            }
        }


class POISearchRequest(BaseModel):
    """POI search request."""
    keywords: str = Field(..., description="搜索关键词", example="故宫")
    city: str = Field(..., description="城市", example="北京")
    citylimit: bool = Field(default=True, description="是否限制在城市范围内")


class RouteRequest(BaseModel):
    """Route-planning request."""
    origin_address: str = Field(..., description="起点地址", example="北京市朝阳区阜通东大街6号")
    destination_address: str = Field(..., description="终点地址", example="北京市海淀区上地十街10号")
    origin_city: Optional[str] = Field(default=None, description="起点城市")
    destination_city: Optional[str] = Field(default=None, description="终点城市")
    route_type: str = Field(default="walking", description="路线类型: walking/driving/transit")


# Response and domain models

class Location(BaseModel):
    """Geographic coordinates."""
    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")


class Attraction(BaseModel):
    """Attraction recommendation."""
    name: str = Field(..., description="景点名称")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    visit_duration: int = Field(..., description="建议游览时间(分钟)")
    description: str = Field(..., description="景点描述")
    category: Optional[str] = Field(default="景点", description="景点类别")
    rating: Optional[float] = Field(default=None, description="评分")
    photos: Optional[List[str]] = Field(default_factory=list, description="景点图片URL列表")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    ticket_price: int = Field(default=0, description="门票价格(元)")


class Meal(BaseModel):
    """Meal recommendation."""
    type: str = Field(..., description="餐饮类型: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="餐饮名称")
    address: Optional[str] = Field(default=None, description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    description: Optional[str] = Field(default=None, description="描述")
    estimated_cost: int = Field(default=0, description="预估费用(元)")


class Hotel(BaseModel):
    """Hotel recommendation."""
    name: str = Field(..., description="酒店名称")
    address: str = Field(default="", description="酒店地址")
    location: Optional[Location] = Field(default=None, description="酒店位置")
    price_range: str = Field(default="", description="价格范围")
    rating: str = Field(default="", description="评分")
    distance: str = Field(default="", description="距离景点距离")
    type: str = Field(default="", description="酒店类型")
    estimated_cost: int = Field(default=0, description="预估费用(元/晚)")


class DayPlan(BaseModel):
    """One day of an itinerary."""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_index: int = Field(..., description="第几天(从0开始)")
    description: str = Field(..., description="当日行程描述")
    transportation: str = Field(..., description="交通方式")
    accommodation: str = Field(..., description="住宿")
    hotel: Optional[Hotel] = Field(default=None, description="推荐酒店")
    attractions: List[Attraction] = Field(default=[], description="景点列表")
    meals: List[Meal] = Field(default=[], description="餐饮列表")


class WeatherInfo(BaseModel):
    """Weather aligned to one trip date."""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_weather: str = Field(default="", description="白天天气")
    night_weather: str = Field(default="", description="夜间天气")
    day_temp: Union[int, str] = Field(default=0, description="白天温度")
    night_temp: Union[int, str] = Field(default=0, description="夜间温度")
    wind_direction: str = Field(default="", description="风向")
    wind_power: str = Field(default="", description="风力")

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
    total_attractions: int = Field(default=0, description="景点门票总费用")
    total_hotels: int = Field(default=0, description="酒店总费用")
    total_meals: int = Field(default=0, description="餐饮总费用")
    total_transportation: int = Field(default=0, description="交通总费用")
    total: int = Field(default=0, description="总费用")


class TripPlan(BaseModel):
    """Validated multi-day trip plan."""
    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    days: List[DayPlan] = Field(..., description="每日行程")
    weather_info: List[WeatherInfo] = Field(default=[], description="天气信息")
    overall_suggestions: str = Field(..., description="总体建议")
    budget: Optional[Budget] = Field(default=None, description="预算信息")


class MemoryProfile(BaseModel):
    """Anonymous preference-memory summary."""
    profile_id: str = Field(..., description="匿名设备偏好记忆ID")
    transportation: str = Field(default="", description="历史常用交通方式")
    accommodation: str = Field(default="", description="历史常用住宿偏好")
    preferences: List[str] = Field(default_factory=list, description="历史偏好标签")
    recent_cities: List[str] = Field(default_factory=list, description="最近规划过的目的地")
    trip_count: int = Field(default=0, description="成功写入记忆的规划次数")
    last_summary: str = Field(default="", description="最近记忆摘要")
    created_at: Optional[float] = Field(default=None, description="创建时间戳")
    updated_at: Optional[float] = Field(default=None, description="更新时间戳")


class TripPlanResponse(BaseModel):
    """Trip-planning API response."""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[TripPlan] = Field(default=None, description="旅行计划数据")
    conversation_id: Optional[str] = Field(default=None, description="后端会话ID")
    memory_applied: bool = Field(default=False, description="是否应用了历史偏好记忆")
    memory_summary: Optional[str] = Field(default=None, description="本次应用的历史偏好摘要")
    memory_profile: Optional[MemoryProfile] = Field(default=None, description="结构化匿名偏好记忆")


class MemoryClearRequest(BaseModel):
    """Anonymous memory cleanup request."""
    profile_id: str = Field(..., description="匿名设备偏好记忆ID")


class MemoryClearResponse(BaseModel):
    """Anonymous memory cleanup response."""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    profile_id: str = Field(..., description="匿名设备偏好记忆ID")


class POIInfo(BaseModel):
    """Normalized POI record."""
    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="名称")
    type: str = Field(..., description="类型")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    tel: Optional[str] = Field(default=None, description="电话")


class POISearchResponse(BaseModel):
    """POI search response."""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[POIInfo] = Field(default_factory=list, description="POI列表")


class RouteInfo(BaseModel):
    """Normalized route result."""
    distance: float = Field(..., description="距离(米)")
    duration: int = Field(..., description="时间(秒)")
    route_type: str = Field(..., description="路线类型")
    description: str = Field(..., description="路线描述")


class RouteResponse(BaseModel):
    """Route-planning response."""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[RouteInfo] = Field(default=None, description="路线信息")


class WeatherResponse(BaseModel):
    """Weather lookup response."""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[WeatherInfo] = Field(default_factory=list, description="天气信息")


# Error response

class ErrorResponse(BaseModel):
    """Standard API error response."""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")
