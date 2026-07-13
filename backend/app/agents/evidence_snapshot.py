"""Normalized evidence snapshots used only by the stability benchmark."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..models.langgraph_state import RAGChunk
from ..models.multi_agent import RegistryEntity
from ..models.schemas import TripRequest, WeatherInfo


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_REQUEST_FIELDS = (
    "city", "start_date", "end_date", "travel_days", "transportation",
    "accommodation", "preferences", "free_text_input", "country_code",
)


class EvidenceSnapshotMismatch(ValueError):
    pass


class ReplayProviderEscape(RuntimeError):
    pass


class AgentEvidenceSnapshot(BaseModel):
    entities: List[RegistryEntity] = Field(default_factory=list)


class ExperienceEvidenceSnapshot(AgentEvidenceSnapshot):
    rag_chunks: List[RAGChunk] = Field(default_factory=list)


class WorkflowEvidenceSnapshot(BaseModel):
    request_fingerprint: str
    request: Dict[str, Any]
    experience: ExperienceEvidenceSnapshot
    logistics: AgentEvidenceSnapshot
    weather_info: List[WeatherInfo] = Field(default_factory=list)


class EvidenceSnapshotFile(BaseModel):
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cases: Dict[str, WorkflowEvidenceSnapshot] = Field(default_factory=dict)
    snapshot_hash: str = ""

    def with_hash(self) -> "EvidenceSnapshotFile":
        value = self.model_copy(deep=True)
        value.snapshot_hash = snapshot_payload_hash(value)
        return value

    def validate_integrity(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise EvidenceSnapshotMismatch(
                f"snapshot schema mismatch: expected {SNAPSHOT_SCHEMA_VERSION}, got {self.schema_version}"
            )
        expected = snapshot_payload_hash(self)
        if not self.snapshot_hash or self.snapshot_hash != expected:
            raise EvidenceSnapshotMismatch("snapshot hash mismatch")


def request_snapshot(request: TripRequest) -> Dict[str, Any]:
    payload = request.model_dump(mode="json")
    return {key: payload.get(key) for key in SNAPSHOT_REQUEST_FIELDS}


def request_fingerprint(request: TripRequest) -> str:
    encoded = json.dumps(request_snapshot(request), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def snapshot_payload_hash(snapshot: EvidenceSnapshotFile) -> str:
    payload = snapshot.model_dump(mode="json", exclude={"snapshot_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
