from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_MIN_RESULTS = 3
DEFAULT_MIN_SCORE = 0.45
DEFAULT_MIN_DISTINCT_SOURCES = 2
DEFAULT_STALE_AFTER_HOURS = 72


@dataclass
class GuardrailDecision:
    allow_reasoning: bool
    insufficient_data_flag: bool
    stale_evidence_warning: bool
    reason: str
    evidence_summary: dict[str, Any]


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def has_meaningful_text(value: str | None) -> bool:
    text = (value or "").strip()
    return len(text) >= 20


def is_stale(
    timestamp_value: str | None,
    stale_after_hours: int = DEFAULT_STALE_AFTER_HOURS,
    reference_time: datetime | None = None,
) -> bool:
    ts = parse_timestamp(timestamp_value)
    if ts is None:
        return True

    if reference_time is None:
        reference_time = utc_now()

    return ts < (reference_time - timedelta(hours=stale_after_hours))


def collect_distinct(values: list[str | None]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value.strip())  # type: ignore[union-attr]
    return output


def summarize_evidence(
    retrieved_chunks: list[dict[str, Any]],
    stale_after_hours: int = DEFAULT_STALE_AFTER_HOURS,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    if reference_time is None:
        reference_time = utc_now()

    doc_types: dict[str, int] = {}
    services: list[str | None] = []
    components: list[str | None] = []
    sources: list[str | None] = []
    scores: list[float] = []
    timestamps: list[datetime] = []
    stale_count = 0
    valid_text_count = 0

    for item in retrieved_chunks:
        doc_type = item.get("doc_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        services.append(item.get("service"))
        components.append(item.get("component"))
        sources.append(item.get("source"))

        score = item.get("score")
        if isinstance(score, (float, int)):
            scores.append(float(score))

        chunk_text = item.get("chunk_text")
        if has_meaningful_text(chunk_text):
            valid_text_count += 1

        ts_start = parse_timestamp(item.get("timestamp_start"))
        ts_end = parse_timestamp(item.get("timestamp_end"))
        if ts_start:
            timestamps.append(ts_start)
        if ts_end:
            timestamps.append(ts_end)

        if is_stale(
            item.get("timestamp_end") or item.get("timestamp_start"),
            stale_after_hours=stale_after_hours,
            reference_time=reference_time,
        ):
            stale_count += 1

    distinct_sources = collect_distinct(sources)
    distinct_services = collect_distinct(services)
    distinct_components = collect_distinct(components)

    summary = {
        "result_count": len(retrieved_chunks),
        "valid_text_count": valid_text_count,
        "doc_type_counts": doc_types,
        "distinct_source_count": len(distinct_sources),
        "distinct_sources": distinct_sources,
        "distinct_services": distinct_services,
        "distinct_components": distinct_components,
        "max_score": max(scores) if scores else 0.0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
        "stale_count": stale_count,
        "fresh_count": len(retrieved_chunks) - stale_count,
        "evidence_time_min": min(timestamps).isoformat().replace("+00:00", "Z") if timestamps else None,
        "evidence_time_max": max(timestamps).isoformat().replace("+00:00", "Z") if timestamps else None,
    }
    return summary


def evaluate_retrieval_quality(
    retrieved_chunks: list[dict[str, Any]],
    min_results: int = DEFAULT_MIN_RESULTS,
    min_score: float = DEFAULT_MIN_SCORE,
    min_distinct_sources: int = DEFAULT_MIN_DISTINCT_SOURCES,
    stale_after_hours: int = DEFAULT_STALE_AFTER_HOURS,
    reference_time: datetime | None = None,
) -> GuardrailDecision:
    summary = summarize_evidence(
        retrieved_chunks=retrieved_chunks,
        stale_after_hours=stale_after_hours,
        reference_time=reference_time,
    )

    result_count = summary["result_count"]
    valid_text_count = summary["valid_text_count"]
    max_score = summary["max_score"]
    distinct_source_count = summary["distinct_source_count"]
    stale_count = summary["stale_count"]
    fresh_count = summary["fresh_count"]

    if result_count == 0:
        return GuardrailDecision(
            allow_reasoning=False,
            insufficient_data_flag=True,
            stale_evidence_warning=False,
            reason="No evidence retrieved.",
            evidence_summary=summary,
        )

    if valid_text_count == 0:
        return GuardrailDecision(
            allow_reasoning=False,
            insufficient_data_flag=True,
            stale_evidence_warning=False,
            reason="Retrieved chunks do not contain meaningful evidence text.",
            evidence_summary=summary,
        )

    if result_count < min_results:
        return GuardrailDecision(
            allow_reasoning=False,
            insufficient_data_flag=True,
            stale_evidence_warning=stale_count > 0,
            reason=f"Too few evidence chunks retrieved. Required at least {min_results}, got {result_count}.",
            evidence_summary=summary,
        )

    if max_score < min_score:
        return GuardrailDecision(
            allow_reasoning=False,
            insufficient_data_flag=True,
            stale_evidence_warning=stale_count > 0,
            reason=f"Top retrieval score below threshold. Required at least {min_score:.2f}, got {max_score:.4f}.",
            evidence_summary=summary,
        )

    if distinct_source_count < min_distinct_sources:
        return GuardrailDecision(
            allow_reasoning=False,
            insufficient_data_flag=True,
            stale_evidence_warning=stale_count > 0,
            reason=(
                f"Insufficient source diversity. Required at least {min_distinct_sources} "
                f"distinct sources, got {distinct_source_count}."
            ),
            evidence_summary=summary,
        )

    if fresh_count == 0:
        return GuardrailDecision(
            allow_reasoning=False,
            insufficient_data_flag=True,
            stale_evidence_warning=True,
            reason="All retrieved evidence is stale.",
            evidence_summary=summary,
        )

    return GuardrailDecision(
        allow_reasoning=True,
        insufficient_data_flag=False,
        stale_evidence_warning=stale_count > 0,
        reason="Evidence quality is sufficient for grounded reasoning.",
        evidence_summary=summary,
    )


def build_refusal_response(decision: GuardrailDecision) -> dict[str, Any]:
    return {
        "issue_type": "unknown",
        "likely_root_cause": "Insufficient evidence to determine a grounded root cause.",
        "supporting_evidence": [],
        "confidence": "low",
        "recommended_next_actions": [
            "Retrieve additional logs, incidents, or runbooks for the affected time window.",
            "Narrow the query to a specific service, component, or incident time range.",
            "Re-run retrieval after new evidence is indexed.",
        ],
        "insufficient_data_flag": True,
        "stale_evidence_warning": decision.stale_evidence_warning,
        "guardrail_reason": decision.reason,
        "evidence_summary": decision.evidence_summary,
    }


def validate_supporting_evidence_items(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False

    required_keys = {
        "source",
        "doc_id",
        "doc_type",
        "timestamp",
        "evidence_text",
    }

    for item in items:
        if not isinstance(item, dict):
            return False
        if not required_keys.issubset(item.keys()):
            return False
        if not has_meaningful_text(item.get("evidence_text")):
            return False

    return True


def validate_reasoning_output(output: dict[str, Any]) -> tuple[bool, str]:
    required_keys = {
        "issue_type",
        "likely_root_cause",
        "supporting_evidence",
        "confidence",
        "recommended_next_actions",
        "insufficient_data_flag",
        "stale_evidence_warning",
    }

    missing = required_keys - set(output.keys())
    if missing:
        return False, f"Missing required output fields: {sorted(missing)}"

    if not validate_supporting_evidence_items(output.get("supporting_evidence", [])):
        return False, "Supporting evidence is missing or invalid."

    if not isinstance(output.get("recommended_next_actions"), list) or not output["recommended_next_actions"]:
        return False, "recommended_next_actions must be a non-empty list."

    if not isinstance(output.get("insufficient_data_flag"), bool):
        return False, "insufficient_data_flag must be boolean."

    if not isinstance(output.get("stale_evidence_warning"), bool):
        return False, "stale_evidence_warning must be boolean."

    return True, "Output passed guardrail validation."


if __name__ == "__main__":
    sample_chunks = [
        {
            "score": 0.62,
            "chunk_id": "incident_incident_010_001",
            "doc_id": "incident_010",
            "doc_type": "incident",
            "source": "incident_report",
            "service": "inference-api",
            "component": "feature-fetcher",
            "timestamp_start": "2026-03-29T15:06:06Z",
            "timestamp_end": "2026-03-29T15:06:06Z",
            "chunk_text": "Summary: Customer-facing latency rose sharply in inference-api during a short but severe event.",
        },
        {
            "score": 0.58,
            "chunk_id": "runbook_runbook_001_001",
            "doc_id": "runbook_001_1",
            "doc_type": "runbook",
            "source": "runbook",
            "service": "inference-api",
            "component": "feature-fetcher",
            "timestamp_start": "2026-03-01T14:46:06Z",
            "timestamp_end": "2026-03-01T14:46:06Z",
            "chunk_text": "Preconditions: timeout rate above threshold\nSteps:\n1. Check feature-store dependency latency for the affected window.",
        },
        {
            "score": 0.55,
            "chunk_id": "log_log_001_001",
            "doc_id": "log_001",
            "doc_type": "log",
            "source": "application_log",
            "service": "inference-api",
            "component": "feature-fetcher",
            "timestamp_start": "2026-03-29T15:01:10Z",
            "timestamp_end": "2026-03-29T15:06:00Z",
            "chunk_text": "2026-03-29T15:03:00Z ERROR timeout_spike Feature fetch duration crossed threshold",
        },
    ]

    decision = evaluate_retrieval_quality(sample_chunks)
    print("allow_reasoning:", decision.allow_reasoning)
    print("insufficient_data_flag:", decision.insufficient_data_flag)
    print("stale_evidence_warning:", decision.stale_evidence_warning)
    print("reason:", decision.reason)
    print("evidence_summary:", decision.evidence_summary)