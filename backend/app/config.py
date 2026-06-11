"""Application configuration."""

import os
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Environment-backed application settings."""

    app_name: str = "Intelligent Trip Planner"
    app_version: str = "1.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    amap_api_key: str = ""
    map_provider: str = "amap"

    weather_provider: str = "openmeteo"
    openweather_api_key: str = ""

    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

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

    if settings.map_provider == "amap" and not settings.amap_api_key:
        errors.append("AMAP_API_KEY is not configured")

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
    print(f"Server: {settings.host}:{settings.port}")
    print(f"Map provider: {settings.map_provider}")
    print(f"Weather provider: {settings.weather_provider}")
    print(f"AMap API key: {'configured' if settings.amap_api_key else 'not configured'}")
    print(f"LLM API key: {'configured' if llm_api_key else 'not configured'}")
    print(f"LLM base URL: {llm_base_url}")
    print(f"LLM model: {llm_model}")
    print(f"Log level: {settings.log_level}")
