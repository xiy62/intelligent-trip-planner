"""LLM service built on LangChain chat models."""

from __future__ import annotations

import os

from ..config import get_settings

_llm_instance = None


def get_llm():
    """Return a singleton LangChain chat model."""
    global _llm_instance

    if _llm_instance is None:
        from langchain_openai import ChatOpenAI

        settings = get_settings()
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or settings.openai_base_url
        model = os.getenv("LLM_MODEL_ID") or os.getenv("OPENAI_MODEL") or settings.openai_model
        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
        )
        print("LLM service initialized")
        print("  Provider: OpenAI-compatible")
        print(f"  Model: {model}")

    return _llm_instance


def get_role_llm(role: str):
    """Return an optional role-specific model, falling back to the shared deterministic model."""
    settings = get_settings()
    model = getattr(settings, f"{role}_model", "")
    if not model:
        return get_llm()
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or settings.openai_base_url
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0)


def reset_llm():
    """Reset the shared LLM instance."""
    global _llm_instance
    _llm_instance = None
