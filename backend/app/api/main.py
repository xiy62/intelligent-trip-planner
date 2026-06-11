"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings, print_config, validate_config
from .routes import map as map_routes
from .routes import observability, poi, trip

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Stateful AI trip-planning API built with LangChain and LangGraph.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(map_routes.router, prefix="/api")
app.include_router(observability.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Validate configuration when the API starts."""
    print("\n" + "=" * 60)
    print(f"{settings.app_name} v{settings.app_version}")
    print("=" * 60)
    print_config()
    validate_config()
    print("\nAPI docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    print("Intelligent Trip Planner is shutting down")


@app.get("/")
async def root():
    """Return service metadata."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health():
    """Return service health."""
    return {"status": "healthy", "service": settings.app_name, "version": settings.app_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host=settings.host, port=settings.port, reload=True)
