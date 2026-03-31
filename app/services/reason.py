import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

PROMPT_PATH = Path("app/prompts/ground_rca_prompt.txt")
DEFAULT_MODEL = "gpt-4.1-mini"


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        return (
            "You are a grounded incident and root-cause analysis assistant. "
            "You must answer only from the provided evidence. "
            "Do not invent facts. "
            "If evidence is insufficient, say so explicitly. "
            "Return valid JSON only."
        )
    return path.read_text(encoding="utf-8").strip()


def build_evidence_block(retrieved_chunks: list[dict[str, Any]]) -> str:
    lines = []

    for idx, item in enumerate(retrieved_chunks, start=1):
        lines.append(f"[Evidence {idx}]")
        lines.append(f"score: {item.get('score')}")
        lines.append(f"chunk_id: {item.get('chunk_id')}")
        lines.append(f"doc_id: {item.get('doc_id')}")
        lines.append(f"doc_type: {item.get('doc_type')}")
        lines.append(f"source: {item.get('source')}")
        lines.append(f"service: {item.get('service')}")
        lines.append(f"component: {item.get('component')}")
        lines.append(f"timestamp_start: {item.get('timestamp_start')}")
        lines.append(f"timestamp_end: {item.get('timestamp_end')}")
        lines.append("chunk_text:")
        lines.append(item.get("chunk_text", ""))
        lines.append("")

    return "\n".join(lines).strip()


def build_user_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    stale_evidence_warning: bool,
) -> str:
    evidence_block = build_evidence_block(retrieved_chunks)

    schema = {
        "issue_type": "string",
        "likely_root_cause": "string",
        "supporting_evidence": [
            {
                "source": "string",
                "doc_id": "string",
                "doc_type": "string",
                "timestamp": "string",
                "evidence_text": "string",
            }
        ],
        "confidence": "low | medium | high",
        "recommended_next_actions": ["string"],
        "insufficient_data_flag": "boolean",
        "stale_evidence_warning": "boolean",
    }

    instructions = f"""
User question:
{query}

You must follow these rules strictly:
1. Use only the provided evidence.
2. Do not infer facts that are not directly supported by the evidence.
3. If evidence is insufficient for a grounded root cause, set:
   - "likely_root_cause" to "Insufficient evidence to determine a grounded root cause."
   - "insufficient_data_flag" to true
4. Always include supporting_evidence entries taken from the retrieved evidence.
5. supporting_evidence must contain concise evidence snippets copied or tightly quoted from the evidence.
6. Set stale_evidence_warning to {str(stale_evidence_warning).lower()}.
7. Return valid JSON only. No markdown. No prose outside JSON.
8. issue_type should be short and operational, such as:
   - latency_spike
   - timeout_spike
   - worker_saturation
   - cache_issue
   - retry_storm
   - dependency_failure
   - unknown
9. Confidence must reflect evidence strength:
   - high: strong direct evidence from multiple chunks
   - medium: plausible grounded evidence but some uncertainty
   - low: weak or incomplete evidence
10. recommended_next_actions must be concrete operational steps, not generic advice.

Required JSON schema:
{json.dumps(schema, indent=2)}

Retrieved evidence:
{evidence_block}
""".strip()

    return instructions


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def call_reasoning_model(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    stale_evidence_warning: bool,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")

    client = get_openai_client()
    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        stale_evidence_warning=stale_evidence_warning,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty response")

    return json.loads(content)


if __name__ == "__main__":
    sample_query = "Why did inference latency spike?"
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
            "chunk_text": (
                "Summary: Customer-facing latency rose sharply in inference-api during a short but severe event.\n"
                "Symptoms: p95 latency climbed sharply; feature fetch duration crossed threshold; request timeouts increased\n"
                "Causes: feature store latency; stale client connections\n"
                "Resolution: Restart stale feature-fetcher clients and reduce fetch concurrency."
            ),
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
            "chunk_text": (
                "Preconditions: timeout rate above threshold; dependency latency elevated for feature path; "
                "queue depth rising in inference-api\n"
                "Steps:\n"
                "1. Check feature-store dependency latency for the affected window.\n"
                "2. Inspect client connection pool health and stale connections."
            ),
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
            "chunk_text": (
                "2026-03-29T15:03:00Z ERROR timeout_spike Feature fetch duration crossed threshold\n"
                "2026-03-29T15:04:10Z WARN dependency_latency Feature-store latency elevated above baseline\n"
                "2026-03-29T15:05:12Z WARN stale_connections Connection pool contained stale clients"
            ),
        },
    ]

    result = call_reasoning_model(
        query=sample_query,
        retrieved_chunks=sample_chunks,
        stale_evidence_warning=True,
    )
    print(json.dumps(result, indent=2))