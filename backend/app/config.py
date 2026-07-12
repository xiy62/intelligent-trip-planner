"""Application configuration."""

import os
from typing import Dict, List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Environment-backed application settings."""

    app_name: str = "Intelligent Trip Planner"
    app_version: str = "1.0.0"
    app_env: str = "local"
    debug: bool = False
    admin_api_token: str = ""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    google_maps_api_key: str = ""
    map_provider: str = "google"

    weather_provider: str = "openmeteo"
    openweather_api_key: str = ""

    rag_mode: str = "chroma_retrieval"
    quality_retry_enabled: bool = False
    min_pacing_score: float = 0.75
    min_route_coherence_score: float = 0.75
    min_preference_match_score: float = 0.60
    route_time_evaluation_enabled: bool = False
    max_route_time_evaluations_per_trip: int = 12
    max_segment_minutes_by_mode: Dict[str, int] = {
        "walking": 30,
        "transit": 45,
        "driving": 35,
        "bicycling": 30,
    }
    max_daily_transit_minutes_by_mode: Dict[str, int] = {
        "walking": 90,
        "transit": 150,
        "driving": 120,
        "bicycling": 120,
    }

    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    experience_model: str = ""
    logistics_model: str = ""
    composer_model: str = ""

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins_list(self) -> List[str]:
        """Return configured CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()


def get_settings() -> Settings:
    """Return the shared settings instance."""
    return settings


def validate_config():
    """Validate configuration required by the active runtime."""
    errors = []
    warnings = []

    if settings.map_provider == "google" and not settings.google_maps_api_key:
        errors.append("GOOGLE_MAPS_API_KEY is not configured")
    elif settings.map_provider != "google":
        errors.append(f"Unsupported MAP_PROVIDER: {settings.map_provider}")

    if settings.weather_provider == "openweather" and not settings.openweather_api_key:
        warnings.append("OPENWEATHER_API_KEY is not configured; OpenWeather will be unavailable")

    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        warnings.append("LLM_API_KEY or OPENAI_API_KEY is not configured; planning will be unavailable")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {error}" for error in errors))

    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return True


def print_config():
    """Print non-secret configuration for local debugging."""
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL") or settings.openai_base_url
    llm_model = os.getenv("LLM_MODEL_ID") or settings.openai_model

    print(f"Application: {settings.app_name}")
    print(f"Version: {settings.app_version}")
    print(f"Environment: {settings.app_env}")
    print(f"Server: {settings.host}:{settings.port}")
    print(f"Map provider: {settings.map_provider}")
    print(f"Weather provider: {settings.weather_provider}")
    print(f"RAG mode: {settings.rag_mode}")
    print(f"Quality retry enabled: {settings.quality_retry_enabled}")
    print(f"Route-time evaluation enabled: {settings.route_time_evaluation_enabled}")
    print(f"Google Maps API key: {'configured' if settings.google_maps_api_key else 'not configured'}")
    print(f"Admin API token: {'configured' if settings.admin_api_token else 'not configured'}")
    print(f"LLM API key: {'configured' if llm_api_key else 'not configured'}")
    print(f"LLM base URL: {llm_base_url}")
    print(f"LLM model: {llm_model}")
    print(f"Log level: {settings.log_level}")
