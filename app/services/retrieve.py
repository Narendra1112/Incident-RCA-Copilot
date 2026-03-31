import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path("app/data/processed")
INDEX_PATH = BASE_DIR / "faiss.index"
ID_MAP_PATH = BASE_DIR / "faiss_id_map.json"
METADATA_PATH = BASE_DIR / "chunk_metadata.json"


class Retriever:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        id_map_path: Path = ID_MAP_PATH,
        metadata_path: Path = METADATA_PATH,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.index = self._load_index(index_path)
        self.id_map = self._load_json(id_map_path)
        self.metadata = self._load_json(metadata_path)
        self.model = SentenceTransformer(model_name)
        self.metadata_by_chunk_id = {
            item["chunk_id"]: item for item in self.metadata
        }

        if self.index.ntotal != len(self.id_map):
            raise ValueError(
                f"FAISS/index map mismatch: ntotal={self.index.ntotal}, id_map={len(self.id_map)}"
            )

    def _load_index(self, path: Path) -> faiss.Index:
        if not path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {path}")
        return faiss.read_index(str(path))

    def _load_json(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}")
        return data

    def _embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")

        embedding = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)

    def _passes_filters(
        self,
        item: dict,
        service: Optional[str] = None,
        component: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> bool:
        if service and item.get("service") != service:
            return False
        if component and item.get("component") != component:
            return False
        if doc_type and item.get("doc_type") != doc_type:
            return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        service: Optional[str] = None,
        component: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_vector = self._embed_query(query)

        search_k = min(max(top_k * 5, 20), self.index.ntotal)
        scores, indices = self.index.search(query_vector, search_k)

        results = []
        seen_chunk_ids = set()

        for score, faiss_id in zip(scores[0], indices[0]):
            if faiss_id == -1:
                continue

            id_entry = self.id_map[faiss_id]
            chunk_id = id_entry["chunk_id"]

            if chunk_id in seen_chunk_ids:
                continue

            metadata = self.metadata_by_chunk_id.get(chunk_id)
            if metadata is None:
                continue

            if not self._passes_filters(
                metadata,
                service=service,
                component=component,
                doc_type=doc_type,
            ):
                continue

            results.append(
                {
                    "score": float(score),
                    "chunk_id": metadata["chunk_id"],
                    "doc_id": metadata["doc_id"],
                    "doc_type": metadata["doc_type"],
                    "source": metadata["source"],
                    "service": metadata.get("service"),
                    "component": metadata.get("component"),
                    "timestamp_start": metadata.get("timestamp_start"),
                    "timestamp_end": metadata.get("timestamp_end"),
                    "chunk_text": metadata["chunk_text"],
                }
            )
            seen_chunk_ids.add(chunk_id)

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "Why did inference latency spike?"
    results = retriever.search(query=query, top_k=5)

    print(f"\nQuery: {query}\n")
    for i, item in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Score: {item['score']:.4f}")
        print(f"Doc Type: {item['doc_type']}")
        print(f"Service: {item['service']}")
        print(f"Component: {item['component']}")
        print(f"Time: {item['timestamp_start']} -> {item['timestamp_end']}")
        print(f"Text: {item['chunk_text'][:400]}")
        print("-" * 80)