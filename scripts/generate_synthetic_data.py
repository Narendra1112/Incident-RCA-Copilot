import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "app" / "data" / "raw"

INCIDENTS_PATH = RAW_DIR / "incidents.jsonl"
LOGS_PATH = RAW_DIR / "logs.jsonl"
RUNBOOKS_PATH = RAW_DIR / "runbooks.jsonl"

SEVERITIES = ["low", "medium", "high", "critical"]

SERVICES = [
    ("inference-api", "request-router"),
    ("inference-api", "feature-fetcher"),
    ("model-server", "predictor"),
    ("model-server", "batcher"),
    ("feature-store", "redis-cache"),
    ("gateway", "traffic-manager"),
]

IRRELEVANT_LOG_MESSAGES = [
    "Background metric flush completed successfully",
    "Health check passed for dependency connection pool",
    "Periodic config refresh completed",
    "Cache warmup completed for low-traffic segment",
    "No-op deployment validation finished",
    "Service heartbeat emitted",
]

NOISE_EVENT_TYPES = [
    "health_check",
    "config_refresh",
    "cache_maintenance",
    "heartbeat",
]


@dataclass(frozen=True)
class RootCauseTemplate:
    name: str
    service: str
    component: str
    symptom_phrases: list[str]
    suspected_causes: list[str]
    resolution: str
    runbook_title: str
    runbook_preconditions: list[str]
    runbook_steps: list[str]
    log_templates: list[str]
    event_type: str
    issue_type: str


ROOT_CAUSES = [
    RootCauseTemplate(
        name="feature_store_latency",
        service="inference-api",
        component="feature-fetcher",
        symptom_phrases=[
            "request timeouts increased",
            "queue depth increased",
            "feature fetch duration crossed threshold",
            "p95 latency climbed sharply",
        ],
        suspected_causes=[
            "feature store latency",
            "stale client connections",
        ],
        resolution="Restart stale feature-fetcher clients and reduce fetch concurrency.",
        runbook_title="Troubleshoot timeout spike in feature fetch path",
        runbook_preconditions=[
            "timeout rate above threshold",
            "dependency latency elevated for feature path",
            "queue depth rising in inference-api",
        ],
        runbook_steps=[
            "Check feature-store dependency latency for the affected window.",
            "Inspect client connection pool health and stale connections.",
            "Review request queue depth and timeout rate together.",
            "Restart stale clients if the pool is exhausted.",
        ],
        log_templates=[
            "Feature store fetch exceeded timeout threshold: {ms}ms",
            "Dependency latency elevated for feature-store: {ms}ms",
            "Feature fetch retries increased for trace window",
            "Queue wait time increased after slow feature fetch path",
        ],
        event_type="dependency_latency",
        issue_type="timeout_spike",
    ),
    RootCauseTemplate(
        name="worker_pool_saturation",
        service="model-server",
        component="batcher",
        symptom_phrases=[
            "p95 latency increased",
            "requests were shed under burst traffic",
            "inflight requests neared limit",
            "queue wait time increased",
        ],
        suspected_causes=[
            "worker saturation",
            "burst traffic exceeded safe concurrency",
        ],
        resolution="Increase worker capacity and tune concurrency limits for burst traffic.",
        runbook_title="Investigate worker saturation and queue backlog",
        runbook_preconditions=[
            "active workers consistently near max",
            "queue backlog increasing",
            "429 or request shedding observed",
        ],
        runbook_steps=[
            "Inspect worker utilization and inflight request count.",
            "Check queue backlog and wait time trend.",
            "Validate whether a traffic burst exceeded configured concurrency.",
            "Adjust worker count or reduce burst concurrency.",
        ],
        log_templates=[
            "Worker pool saturation detected: active_workers={count}",
            "Rejected request due to inflight limit",
            "Queue wait time exceeded threshold: {ms}ms",
            "Burst traffic detected beyond safe concurrency envelope",
        ],
        event_type="worker_saturation",
        issue_type="latency_spike",
    ),
    RootCauseTemplate(
        name="db_pool_exhaustion",
        service="inference-api",
        component="request-router",
        symptom_phrases=[
            "timeouts increased across read path",
            "database acquire wait time increased",
            "error rate rose for metadata lookups",
            "latency became unstable",
        ],
        suspected_causes=[
            "database connection pool exhaustion",
            "connection leak in request path",
        ],
        resolution="Recycle leaking workers and increase DB pool safety margin.",
        runbook_title="Investigate DB pool exhaustion in request path",
        runbook_preconditions=[
            "DB connection acquire wait increased",
            "timeouts affected request router",
            "metadata lookup latency elevated",
        ],
        runbook_steps=[
            "Inspect DB connection utilization and pending acquires.",
            "Check request-router error logs for connection timeout patterns.",
            "Review recent changes in connection handling.",
            "Restart leaking workers and retune pool size if needed.",
        ],
        log_templates=[
            "DB connection acquire wait exceeded threshold: {ms}ms",
            "Connection pool exhausted for metadata lookup",
            "Timed out waiting for DB connection from pool",
            "Request path blocked on DB acquire operation",
        ],
        event_type="db_pool_exhaustion",
        issue_type="timeout_spike",
    ),
    RootCauseTemplate(
        name="retry_storm",
        service="gateway",
        component="traffic-manager",
        symptom_phrases=[
            "request volume surged unexpectedly",
            "duplicate retries amplified backend load",
            "latency and timeout rates both increased",
            "downstream services showed cascading stress",
        ],
        suspected_causes=[
            "retry storm from gateway",
            "aggressive retry policy amplified failures",
        ],
        resolution="Throttle retries and cap retry fan-out at gateway.",
        runbook_title="Mitigate retry storm from gateway",
        runbook_preconditions=[
            "retry volume surged abruptly",
            "backend timeout rate increased with traffic amplification",
            "gateway emitted repeated retry events",
        ],
        runbook_steps=[
            "Inspect retry volume and retry fan-out ratio.",
            "Check whether a downstream dependency was already degraded.",
            "Reduce retry concurrency and backoff aggressiveness.",
            "Confirm backend load normalizes after retry cap is applied.",
        ],
        log_templates=[
            "Retry burst detected for upstream timeout condition",
            "Gateway retry fan-out exceeded safe threshold",
            "Amplified request volume observed after dependency slowdown",
            "Retry policy generated repeated requests for same trace class",
        ],
        event_type="retry_storm",
        issue_type="timeout_spike",
    ),
    RootCauseTemplate(
        name="bad_model_rollout",
        service="model-server",
        component="predictor",
        symptom_phrases=[
            "latency increased after rollout",
            "CPU usage climbed after model switch",
            "throughput dropped for online inference",
            "request timeout rate rose after deployment",
        ],
        suspected_causes=[
            "inefficient new model version",
            "bad rollout introduced slower inference path",
        ],
        resolution="Rollback model version and validate inference latency before re-rollout.",
        runbook_title="Validate model rollout after latency regression",
        runbook_preconditions=[
            "latency increased after deployment",
            "throughput dropped post-rollout",
            "model-server logs show slower inference path",
        ],
        runbook_steps=[
            "Compare latency before and after model rollout.",
            "Check whether model version changed execution path or batch behavior.",
            "Rollback if latency regression is confirmed.",
            "Revalidate rollout with canary traffic before promotion.",
        ],
        log_templates=[
            "Inference duration exceeded baseline after rollout: {ms}ms",
            "Model version switch correlated with slower prediction path",
            "Batch execution time increased after deployment",
            "Predictor throughput degraded following model update",
        ],
        event_type="rollout_regression",
        issue_type="latency_spike",
    ),
    RootCauseTemplate(
        name="cache_miss_spike",
        service="feature-store",
        component="redis-cache",
        symptom_phrases=[
            "cache miss ratio increased suddenly",
            "backend fetch volume spiked",
            "latency rose in feature retrieval path",
            "timeouts followed sustained cache misses",
        ],
        suspected_causes=[
            "cache miss spike",
            "cache invalidation issue",
        ],
        resolution="Stabilize cache invalidation and repopulate hot keys.",
        runbook_title="Investigate cache miss spike in feature-store",
        runbook_preconditions=[
            "cache miss ratio elevated",
            "backend fetches increased after invalidation",
            "feature retrieval latency increased",
        ],
        runbook_steps=[
            "Inspect cache miss ratio and invalidation events.",
            "Check whether hot keys were evicted or invalidated unexpectedly.",
            "Repopulate hot keys for affected services.",
            "Confirm feature retrieval latency returns to baseline.",
        ],
        log_templates=[
            "Cache miss ratio increased to {ratio}",
            "Unexpected cache invalidation observed for hot key set",
            "Backend fetch volume rose after cache miss surge",
            "Feature-store latency elevated after cache churn",
        ],
        event_type="cache_miss_spike",
        issue_type="latency_spike",
    ),
]


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_incident_text(
    title: str,
    summary: str,
    symptoms: list[str],
    suspected_causes: list[str],
    resolution: str,
    service: str,
    component: str,
    severity: str,
    timestamp: str,
) -> str:
    return (
        f"Incident Title: {title}\n"
        f"Timestamp: {timestamp}\n"
        f"Service: {service}\n"
        f"Component: {component}\n"
        f"Severity: {severity}\n"
        f"Summary: {summary}\n"
        f"Symptoms: {'; '.join(symptoms)}\n"
        f"Suspected Causes: {'; '.join(suspected_causes)}\n"
        f"Resolution: {resolution}"
    )


def flatten_runbook_text(
    title: str,
    preconditions: list[str],
    steps: list[str],
    service: str,
    component: str,
    timestamp: str,
) -> str:
    return (
        f"Runbook Title: {title}\n"
        f"Timestamp: {timestamp}\n"
        f"Service: {service}\n"
        f"Component: {component}\n"
        f"Preconditions: {'; '.join(preconditions)}\n"
        f"Steps: {'; '.join(steps)}"
    )


def build_runbooks(now: datetime) -> list[dict[str, Any]]:
    runbooks: list[dict[str, Any]] = []

    for idx, template in enumerate(ROOT_CAUSES, start=1):
        recent_dt = now - timedelta(days=random.randint(5, 35))
        old_dt = now - timedelta(days=random.randint(90, 180))

        for version_idx, ts in enumerate([recent_dt, old_dt], start=1):
            doc_id = f"runbook_{idx:03d}_{version_idx}"
            title = template.runbook_title
            if version_idx == 2:
                title = f"{title} (legacy)"

            steps = list(template.runbook_steps)
            if version_idx == 2:
                steps = steps[:-1] + ["Escalate to service owner if issue persists."]

            timestamp = iso_z(ts)
            runbooks.append(
                {
                    "doc_id": doc_id,
                    "doc_type": "runbook",
                    "source": "runbook",
                    "timestamp": timestamp,
                    "service": template.service,
                    "component": template.component,
                    "severity": "info",
                    "title": title,
                    "preconditions": template.runbook_preconditions,
                    "steps": steps,
                    "text": flatten_runbook_text(
                        title=title,
                        preconditions=template.runbook_preconditions,
                        steps=steps,
                        service=template.service,
                        component=template.component,
                        timestamp=timestamp,
                    ),
                }
            )

    extra_runbooks = [
        {
            "doc_id": "runbook_999_1",
            "doc_type": "runbook",
            "source": "runbook",
            "timestamp": iso_z(now - timedelta(days=20)),
            "service": "gateway",
            "component": "traffic-manager",
            "severity": "info",
            "title": "Routine gateway health validation",
            "preconditions": ["routine inspection window", "no active sev issue"],
            "steps": [
                "Check health endpoint response.",
                "Verify config refresh status.",
                "Confirm route table load succeeded.",
            ],
            "text": flatten_runbook_text(
                title="Routine gateway health validation",
                preconditions=["routine inspection window", "no active sev issue"],
                steps=[
                    "Check health endpoint response.",
                    "Verify config refresh status.",
                    "Confirm route table load succeeded.",
                ],
                service="gateway",
                component="traffic-manager",
                timestamp=iso_z(now - timedelta(days=20)),
            ),
        },
        {
            "doc_id": "runbook_999_2",
            "doc_type": "runbook",
            "source": "runbook",
            "timestamp": iso_z(now - timedelta(days=140)),
            "service": "feature-store",
            "component": "redis-cache",
            "severity": "info",
            "title": "Legacy cache restart checklist",
            "preconditions": ["cache restart planned"],
            "steps": [
                "Restart cache nodes sequentially.",
                "Warm a small set of keys.",
                "Notify platform owner.",
            ],
            "text": flatten_runbook_text(
                title="Legacy cache restart checklist",
                preconditions=["cache restart planned"],
                steps=[
                    "Restart cache nodes sequentially.",
                    "Warm a small set of keys.",
                    "Notify platform owner.",
                ],
                service="feature-store",
                component="redis-cache",
                timestamp=iso_z(now - timedelta(days=140)),
            ),
        },
    ]
    runbooks.extend(extra_runbooks)
    return runbooks


def build_incidents(now: datetime, num_incidents: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    incidents: list[dict[str, Any]] = []
    anchor_events: list[dict[str, Any]] = []

    for idx in range(1, num_incidents + 1):
        template = random.choice(ROOT_CAUSES)
        incident_time = now - timedelta(
            hours=random.randint(4, 72),
            minutes=random.randint(0, 59),
        )
        severity = random.choice(["high", "critical"])

        title = f"{template.issue_type.replace('_', ' ').title()} in {template.service}"
        summary = random.choice(
            [
                f"{template.service} showed elevated latency and timeouts during a concentrated incident window.",
                f"{template.component} exhibited degraded behavior that impacted request performance.",
                f"Customer-facing latency rose sharply in {template.service} during a short but severe event.",
            ]
        )

        symptoms = random.sample(template.symptom_phrases, k=min(3, len(template.symptom_phrases)))
        suspected_causes = template.suspected_causes

        if idx % 6 == 0:
            suspected_causes = ["insufficient direct evidence during incident window"]

        resolution = template.resolution if idx % 6 != 0 else "Issue stabilized before root cause was confirmed."

        timestamp = iso_z(incident_time)
        doc_id = f"incident_{idx:03d}"

        incident = {
            "doc_id": doc_id,
            "doc_type": "incident",
            "source": "incident_report",
            "timestamp": timestamp,
            "service": template.service,
            "component": template.component,
            "severity": severity,
            "title": title,
            "summary": summary,
            "symptoms": symptoms,
            "suspected_causes": suspected_causes,
            "resolution": resolution,
            "text": flatten_incident_text(
                title=title,
                summary=summary,
                symptoms=symptoms,
                suspected_causes=suspected_causes,
                resolution=resolution,
                service=template.service,
                component=template.component,
                severity=severity,
                timestamp=timestamp,
            ),
        }
        incidents.append(incident)

        anchor_events.append(
            {
                "incident_id": doc_id,
                "template": template,
                "incident_time": incident_time,
                "service": template.service,
                "component": template.component,
                "weak_evidence_case": idx % 6 == 0,
            }
        )

    return incidents, anchor_events


def fill_template_message(template_str: str) -> str:
    return template_str.format(
        ms=random.choice([420, 680, 950, 1200, 1800, 2200, 2800]),
        count=random.choice([64, 72, 88, 96, 112, 128]),
        ratio=random.choice(["0.41", "0.57", "0.68", "0.73", "0.81"]),
    )


def build_logs(
    now: datetime,
    anchors: list[dict[str, Any]],
    target_log_count: int = 500,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    log_counter = 1

    for anchor in anchors:
        template: RootCauseTemplate = anchor["template"]
        incident_time: datetime = anchor["incident_time"]
        weak_evidence_case: bool = anchor["weak_evidence_case"]

        useful_log_count = 10 if not weak_evidence_case else 4
        noise_log_count = 8

        for _ in range(useful_log_count):
            ts = incident_time - timedelta(minutes=random.randint(0, 20), seconds=random.randint(0, 59))
            msg = fill_template_message(random.choice(template.log_templates))
            log_level = random.choice(["WARN", "ERROR"])
            severity = "high" if log_level == "WARN" else "critical"

            record = {
                "doc_id": f"log_{log_counter:05d}",
                "doc_type": "log",
                "source": "application_log",
                "timestamp": iso_z(ts),
                "service": template.service,
                "component": template.component,
                "severity": severity,
                "log_level": log_level,
                "event_type": template.event_type,
                "message": msg,
                "text": f"{iso_z(ts)} {log_level} {template.service} {template.component} {msg}",
            }
            logs.append(record)
            log_counter += 1

        for _ in range(noise_log_count):
            service, component = random.choice(SERVICES)
            ts = incident_time - timedelta(minutes=random.randint(0, 30), seconds=random.randint(0, 59))
            msg = random.choice(IRRELEVANT_LOG_MESSAGES)
            event_type = random.choice(NOISE_EVENT_TYPES)
            record = {
                "doc_id": f"log_{log_counter:05d}",
                "doc_type": "log",
                "source": "application_log",
                "timestamp": iso_z(ts),
                "service": service,
                "component": component,
                "severity": random.choice(["low", "medium"]),
                "log_level": "INFO",
                "event_type": event_type,
                "message": msg,
                "text": f"{iso_z(ts)} INFO {service} {component} {msg}",
            }
            logs.append(record)
            log_counter += 1

    while len(logs) < target_log_count:
        service, component = random.choice(SERVICES)
        ts = now - timedelta(hours=random.randint(1, 96), minutes=random.randint(0, 59))
        is_noise = random.random() < 0.75

        if is_noise:
            msg = random.choice(IRRELEVANT_LOG_MESSAGES)
            event_type = random.choice(NOISE_EVENT_TYPES)
            log_level = "INFO"
            severity = random.choice(["low", "medium"])
        else:
            template = random.choice(ROOT_CAUSES)
            msg = fill_template_message(random.choice(template.log_templates))
            event_type = template.event_type
            log_level = random.choice(["WARN", "ERROR"])
            severity = random.choice(["medium", "high"])

        record = {
            "doc_id": f"log_{log_counter:05d}",
            "doc_type": "log",
            "source": "application_log",
            "timestamp": iso_z(ts),
            "service": service,
            "component": component,
            "severity": severity,
            "log_level": log_level,
            "event_type": event_type,
            "message": msg,
            "text": f"{iso_z(ts)} {log_level} {service} {component} {msg}",
        }
        logs.append(record)
        log_counter += 1

    logs.sort(key=lambda x: x["timestamp"])
    return logs


def validate_counts(
    incidents: list[dict[str, Any]],
    logs: list[dict[str, Any]],
    runbooks: list[dict[str, Any]],
) -> None:
    if not incidents:
        raise ValueError("No incidents generated.")
    if not logs:
        raise ValueError("No logs generated.")
    if not runbooks:
        raise ValueError("No runbooks generated.")

    doc_ids = defaultdict(int)
    for row in incidents + logs + runbooks:
        doc_ids[row["doc_id"]] += 1

    duplicates = [doc_id for doc_id, count in doc_ids.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate doc_ids found: {duplicates[:5]}")


def main() -> None:
    random.seed(42)
    ensure_dirs()

    now = datetime.now(timezone.utc)

    incidents, anchors = build_incidents(now=now, num_incidents=20)
    runbooks = build_runbooks(now=now)
    logs = build_logs(now=now, anchors=anchors, target_log_count=500)

    validate_counts(incidents=incidents, logs=logs, runbooks=runbooks)

    write_jsonl(INCIDENTS_PATH, incidents)
    write_jsonl(LOGS_PATH, logs)
    write_jsonl(RUNBOOKS_PATH, runbooks)

    print(f"Wrote {len(incidents)} incidents to {INCIDENTS_PATH}")
    print(f"Wrote {len(logs)} logs to {LOGS_PATH}")
    print(f"Wrote {len(runbooks)} runbooks to {RUNBOOKS_PATH}")


if __name__ == "__main__":
    main()