"""Shared FastAPI dependencies."""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, status

from ..config import get_settings


PRODUCTION_ENVS = {"prod", "production"}


def require_admin_token(x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")) -> None:
    """Require a lightweight admin token for local-admin API routes."""
    settings = get_settings()
    configured_token = (settings.admin_api_token or "").strip()
    app_env = (settings.app_env or "").strip().lower()

    if not configured_token:
        if app_env in PRODUCTION_ENVS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin API token is required",
            )
        return

    provided_token = (x_admin_token or "").strip()
    if not provided_token or not hmac.compare_digest(provided_token, configured_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token",
        )
