import os
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieve import Retriever
from app.services.guardrails import (
    build_refusal_response,
    evaluate_retrieval_quality,
    validate_reasoning_output,
)
from app.services.reason import call_reasoning_model
from dotenv import load_dotenv
load_dotenv()
import time

app = FastAPI(title="Incident RCA Copilot", version="0.2.0")


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)
    service: Optional[str] = None
    component: Optional[str] = None


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk)

    return deduped


def diversified_retrieval(
    retriever: Retriever,
    question: str,
    final_top_k: int,
    service: Optional[str] = None,
    component: Optional[str] = None,
) -> list[dict]:
    candidate_pool = []

    broad_results = retriever.search(
        query=question,
        top_k=max(final_top_k * 3, 12),
        service=service,
        component=component,
    )
    candidate_pool.extend(broad_results)

    for doc_type in ["incident", "log", "runbook"]:
        typed_results = retriever.search(
            query=question,
            top_k=max(final_top_k, 4),
            service=service,
            component=component,
            doc_type=doc_type,
        )
        candidate_pool.extend(typed_results)

    candidate_pool = dedupe_chunks(candidate_pool)

    doc_type_best = {"incident": [], "log": [], "runbook": []}
    for item in candidate_pool:
        doc_type = item.get("doc_type")
        if doc_type in doc_type_best:
            doc_type_best[doc_type].append(item)

    for doc_type in doc_type_best:
        doc_type_best[doc_type].sort(key=lambda x: x["score"], reverse=True)

    selected = []

    for doc_type in ["incident", "log", "runbook"]:
        if doc_type_best[doc_type]:
            selected.append(doc_type_best[doc_type][0])

    selected_ids = {item["chunk_id"] for item in selected}

    remaining = sorted(candidate_pool, key=lambda x: x["score"], reverse=True)
    for item in remaining:
        if item["chunk_id"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["chunk_id"])
        if len(selected) >= final_top_k:
            break

    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected[:final_top_k]


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    start_time = time.time()

    try:
        print(f"\n[REQUEST] {request.question}")

        retriever = get_retriever()

        retrieved_chunks = diversified_retrieval(
            retriever=retriever,
            question=request.question,
            final_top_k=request.top_k,
            service=request.service,
            component=request.component,
        )

        print(f"[RETRIEVAL] Retrieved {len(retrieved_chunks)} chunks")

        decision = evaluate_retrieval_quality(retrieved_chunks)

        print(f"[GUARDRAIL] {decision.reason}")
        print(f"[GUARDRAIL] Allow reasoning: {decision.allow_reasoning}")

        if not decision.allow_reasoning:
            return build_refusal_response(decision)

        response = call_reasoning_model(
            query=request.question,
            retrieved_chunks=retrieved_chunks,
            stale_evidence_warning=decision.stale_evidence_warning,
        )

        is_valid, validation_message = validate_reasoning_output(response)

        print(f"[VALIDATION] {validation_message}")

        if not is_valid:
            raise HTTPException(
                status_code=500,
                detail=f"Reasoning output validation failed: {validation_message}",
            )

        elapsed = round(time.time() - start_time, 2)
        print(f"[RESPONSE] Completed in {elapsed}s")

        return response

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[ERROR] {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))