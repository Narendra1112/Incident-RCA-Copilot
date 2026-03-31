import json
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

RAW_DIR = Path("app/data/raw")
OUTPUT_PATH = Path("app/data/processed/chunks.jsonl")

INCIDENTS_PATH = RAW_DIR / "incidents.jsonl"
RUNBOOKS_PATH = RAW_DIR / "runbooks.jsonl"
LOGS_PATH = RAW_DIR / "logs.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def make_chunk_id(doc_type: str, doc_id: str, suffix: str) -> str:
    return f"{doc_type}_{doc_id}_{suffix}"


def chunk_incidents(incidents: list[dict]) -> list[dict]:
    chunks = []

    for record in incidents:
        summary = record.get("summary", "").strip()
        symptoms = record.get("symptoms", [])
        suspected_causes = record.get("suspected_causes", [])
        resolution = record.get("resolution", "").strip()

        parts = []
        if summary:
            parts.append(f"Summary: {summary}")
        if symptoms:
            parts.append(f"Symptoms: {'; '.join(symptoms)}")
        if suspected_causes:
            parts.append(f"Causes: {'; '.join(suspected_causes)}")
        if resolution:
            parts.append(f"Resolution: {resolution}")

        chunks.append(
            {
                "chunk_id": make_chunk_id("incident", record["doc_id"], "001"),
                "doc_id": record["doc_id"],
                "doc_type": record["doc_type"],
                "source": record["source"],
                "service": record.get("service"),
                "component": record.get("component"),
                "timestamp_start": record.get("timestamp"),
                "timestamp_end": record.get("timestamp"),
                "chunk_text": "\n".join(parts),
            }
        )

    return chunks


def chunk_runbooks(runbooks: list[dict]) -> list[dict]:
    chunks = []

    for record in runbooks:
        preconditions = record.get("preconditions", [])
        steps = record.get("steps", [])

        parts = []
        if preconditions:
            parts.append(f"Preconditions: {'; '.join(preconditions)}")
        if steps:
            numbered_steps = [f"{idx}. {step}" for idx, step in enumerate(steps, start=1)]
            parts.append("Steps:\n" + "\n".join(numbered_steps))

        chunks.append(
            {
                "chunk_id": make_chunk_id("runbook", record["doc_id"], "001"),
                "doc_id": record["doc_id"],
                "doc_type": record["doc_type"],
                "source": record["source"],
                "service": record.get("service"),
                "component": record.get("component"),
                "timestamp_start": record.get("timestamp"),
                "timestamp_end": record.get("timestamp"),
                "chunk_text": "\n".join(parts),
            }
        )

    return chunks


def split_batches(records: list[dict], min_size: int = 4, max_size: int = 12) -> list[list[dict]]:
    n = len(records)
    if n < min_size:
        return []

    batch_count = math.ceil(n / max_size)

    while batch_count <= n:
        base = n // batch_count
        remainder = n % batch_count
        sizes = [base + (1 if i < remainder else 0) for i in range(batch_count)]

        if all(min_size <= size <= max_size for size in sizes):
            batches = []
            start = 0
            for size in sizes:
                end = start + size
                batches.append(records[start:end])
                start = end
            return batches

        batch_count += 1

    return []


def build_log_chunk(batch: list[dict], chunk_index: int) -> dict:
    first = batch[0]
    start_ts = batch[0]["timestamp"]
    end_ts = batch[-1]["timestamp"]

    lines = []
    for log in batch:
        lines.append(
            f"{log['timestamp']} {log.get('log_level', '')} {log.get('event_type', '')} {log.get('message', '')}".strip()
        )

    return {
        "chunk_id": make_chunk_id("log", first["doc_id"], f"{chunk_index:03d}"),
        "doc_id": first["doc_id"],
        "doc_type": "log",
        "source": first["source"],
        "service": first.get("service"),
        "component": first.get("component"),
        "timestamp_start": start_ts,
        "timestamp_end": end_ts,
        "chunk_text": "\n".join(lines),
    }


def chunk_logs(logs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in logs:
        key = (record.get("service"), record.get("component"))
        grouped[key].append(record)

    chunks = []

    for (_, _), group_logs in grouped.items():
        group_logs.sort(key=lambda x: parse_timestamp(x["timestamp"]))

        minute_buckets = defaultdict(list)
        for log in group_logs:
            ts = parse_timestamp(log["timestamp"])
            minute_buckets[floor_to_minute(ts)].append(log)

        sorted_minutes = sorted(minute_buckets.keys())
        chunk_index = 1
        i = 0

        while i < len(sorted_minutes):
            collected = []
            window_start = sorted_minutes[i]
            best_batches = []
            best_end_index = None

            for j in range(i, len(sorted_minutes)):
                minute_key = sorted_minutes[j]
                if minute_key - window_start > timedelta(minutes=10):
                    break

                collected.extend(minute_buckets[minute_key])

                candidate_batches = split_batches(collected, min_size=4, max_size=12)
                if candidate_batches:
                    best_batches = candidate_batches
                    best_end_index = j

            if best_batches:
                for batch in best_batches:
                    chunks.append(build_log_chunk(batch, chunk_index))
                    chunk_index += 1
                i = best_end_index + 1
            else:
                i += 1

    return chunks


def main() -> None:
    incidents = load_jsonl(INCIDENTS_PATH)
    runbooks = load_jsonl(RUNBOOKS_PATH)
    logs = load_jsonl(LOGS_PATH)

    all_chunks = []
    all_chunks.extend(chunk_incidents(incidents))
    all_chunks.extend(chunk_runbooks(runbooks))
    all_chunks.extend(chunk_logs(logs))

    write_jsonl(OUTPUT_PATH, all_chunks)
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()