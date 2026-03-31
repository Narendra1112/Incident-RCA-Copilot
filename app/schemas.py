from typing import List, Literal, Optional

from pydantic import BaseModel, Field


DocType = Literal["incident", "log", "runbook"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Incident question from user")
    service: Optional[str] = Field(default=None, description="Optional service filter")
    top_k: int = Field(default=8, ge=3, le=15, description="Number of evidence chunks to return")


class EvidenceItem(BaseModel):
    source: str
    doc_id: str
    chunk_id: str
    doc_type: DocType
    service: str
    component: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    relevance_score: float
    snippet: str


class AskResponse(BaseModel):
    issue_type: str
    likely_root_cause: str
    supporting_evidence: List[EvidenceItem]
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_next_actions: List[str]
    insufficient_data_flag: bool
    stale_evidence_warning: bool
    answer_summary: str


class BaseDocument(BaseModel):
    doc_id: str
    doc_type: DocType
    source: str
    timestamp: str
    service: str
    component: str
    severity: str
    text: str


class IncidentDocument(BaseDocument):
    doc_type: Literal["incident"] = "incident"
    source: Literal["incident_report"] = "incident_report"
    title: str
    summary: str
    symptoms: List[str]
    suspected_causes: List[str]
    resolution: str


class LogDocument(BaseDocument):
    doc_type: Literal["log"] = "log"
    source: Literal["application_log"] = "application_log"
    log_level: str
    event_type: str
    message: str


class RunbookDocument(BaseDocument):
    doc_type: Literal["runbook"] = "runbook"
    source: Literal["runbook"] = "runbook"
    title: str
    preconditions: List[str]
    steps: List[str]