"""Small adapter for typed agent outputs across real and fake chat models."""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def invoke_structured(llm: Any, schema: Type[T], prompt: str) -> T:
    """Invoke a model with a Pydantic contract, with a JSON fallback for test doubles."""
    if hasattr(llm, "with_structured_output"):
        value = llm.with_structured_output(schema, method="function_calling").invoke(prompt)
        return value if isinstance(value, schema) else schema.model_validate(value)
    response = llm.invoke(prompt)
    value = getattr(response, "content", response)
    if isinstance(value, schema):
        return value
    if isinstance(value, dict):
        return schema.model_validate(value)
    text = str(value).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return schema.model_validate(json.loads(text))
