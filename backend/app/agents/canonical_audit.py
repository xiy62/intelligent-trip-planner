"""Independent audit of provider-owned fields in a materialized trip plan."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from pydantic import BaseModel, Field

from ..models.multi_agent import CandidateRegistry, RegistryEntity
from ..models.schemas import TripPlan


class CanonicalFieldMismatch(BaseModel):
    path: str
    entity_type: str
    provider_id: str
    field: str
    expected: Any = None
    actual: Any = None


class CanonicalAuditResult(BaseModel):
    mismatches: List[CanonicalFieldMismatch] = Field(default_factory=list)
    audited_entity_count: int = 0
    skipped_generic_meal_count: int = 0

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)


def audit_canonical_fields(plan: Optional[TripPlan], registry: CandidateRegistry) -> CanonicalAuditResult:
    """Compare final provider-owned fields with their request-local registry entities."""
    result = CanonicalAuditResult()
    if plan is None:
        return result

    for day_index, day in enumerate(plan.days):
        for item_index, item in enumerate(day.attractions):
            _audit_entity(
                result,
                registry,
                entity_type="attraction",
                provider_id=item.poi_id or "",
                path=f"days.{day_index}.attractions.{item_index}",
                actual=item,
                fields=("poi_id", "name", "address", "location", "rating", "photos", "category",
                        "image_url", "maps_url", "website_url"),
            )
        if day.hotel is not None:
            _audit_entity(
                result,
                registry,
                entity_type="hotel",
                provider_id=day.hotel.poi_id or "",
                path=f"days.{day_index}.hotel",
                actual=day.hotel,
                fields=("poi_id", "name", "address", "location", "rating", "image_url", "maps_url",
                        "website_url"),
            )
        for item_index, item in enumerate(day.meals):
            if not item.poi_id:
                result.skipped_generic_meal_count += 1
                continue
            _audit_entity(
                result,
                registry,
                entity_type="meal",
                provider_id=item.poi_id,
                path=f"days.{day_index}.meals.{item_index}",
                actual=item,
                fields=("poi_id", "name", "address", "location", "image_url", "maps_url", "website_url"),
            )
    return result


def _audit_entity(
    result: CanonicalAuditResult,
    registry: CandidateRegistry,
    *,
    entity_type: str,
    provider_id: str,
    path: str,
    actual: Any,
    fields: Iterable[str],
) -> None:
    entity = _find_entity(registry, entity_type, provider_id)
    if entity is None:
        wrong_types = sorted({item.entity_type for item in registry.entities.values()
                              if _public_provider_id(item) == provider_id})
        result.mismatches.append(CanonicalFieldMismatch(
            path=path,
            entity_type=entity_type,
            provider_id=provider_id,
            field="identity_type" if wrong_types else "identity",
            expected=entity_type if wrong_types else "request-local registry entity",
            actual=wrong_types or provider_id,
        ))
        return

    result.audited_entity_count += 1
    expected = _canonical_values(entity)
    for field_name in fields:
        expected_value = _json_value(expected[field_name])
        actual_value = _json_value(getattr(actual, field_name))
        if expected_value != actual_value:
            result.mismatches.append(CanonicalFieldMismatch(
                path=path,
                entity_type=entity_type,
                provider_id=provider_id,
                field=field_name,
                expected=expected_value,
                actual=actual_value,
            ))


def _find_entity(registry: CandidateRegistry, entity_type: str, provider_id: str) -> Optional[RegistryEntity]:
    for entity in registry.entities.values():
        if entity.entity_type == entity_type and _public_provider_id(entity) == provider_id:
            return entity
    return None


def _public_provider_id(entity: RegistryEntity) -> str:
    return entity.provider_id or entity.source_id


def _canonical_values(entity: RegistryEntity) -> dict[str, Any]:
    values = {
        "poi_id": _public_provider_id(entity),
        "name": entity.name,
        "address": entity.address,
        "location": entity.location,
        "rating": entity.rating,
        "photos": list(entity.photo_names),
        "category": str(entity.metadata.get("category") or "Attraction"),
        "image_url": entity.image_url,
        "maps_url": entity.maps_url,
        "website_url": entity.website_url,
    }
    if entity.entity_type == "hotel":
        values["rating"] = str(entity.rating or "")
    return values


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return list(value)
    return value
