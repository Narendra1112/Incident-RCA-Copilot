# RAG-based Incident & Root-Cause Copilot

A RAG-based system for root-cause analysis over unstructured operational data, designed with guardrails to ensure grounded, reliable, and non-hallucinated outputs.

---

##  🔍 Overview

This project supports root-cause analysis queries such as:

- Why did inference latency spike?
- Why were requests rejected due to inflight limits?
- Why are requests timing out?

It retrieves and reasons over:
- application logs
- incident reports
- operational runbooks

---

##  ⚙️ Architecture

Query → Retrieval (FAISS + embeddings) → Guardrails → LLM → Validated JSON Output


---

## 🚀 Features (Phase 1)

- Semantic retrieval using FAISS + sentence-transformers  
- Multi-source evidence selection (logs, incidents, runbooks)  
- Guardrails:
  - insufficient evidence detection  
  - stale evidence warning  
  - no hallucinated reasoning  
- LLM-based reasoning with strict JSON output  
- FastAPI `/analyze` endpoint  
- Evaluation pipeline → **7/7 test scenarios passed**

---

##  Example Output

```json
{
  "issue_type": "worker_saturation",
  "likely_root_cause": "...",
  "supporting_evidence": [...],
  "confidence": "high",
  "recommended_next_actions": [...],
  "insufficient_data_flag": false,
  "stale_evidence_warning": true
}
```
## 🛠️ Tech Stack
 - Python
 - FastAPI
 - FAISS
 - sentence-transformers
 - OpenAI

## ▶️ Setup

```
git clone https://github.com/Narendra1112/Incident-RCA-Copilot.git
cd incident-rca-copilot

pip install -r requirements.txt
```

Create .env:

```
OPENAI_API_KEY=your_api_key
```
Run server:
```
uvicorn app.main:app --reload
```

### Evaluation

Run:

```
python app/evaluation/evaluate.py
```

Result:
```
7/7 tests passed
```

## 🔜 Next Steps

- Validate on real-world incident datasets
- Improve retrieval ranking under noisy logs
- Add ingestion pipeline for new data

## 📌 Notes

This is a Phase 1 implementation focused on:

 - system design
 - grounding and guardrails
 - structured reasoning

## 🔗 Author
 Narendra Kumar Poluka