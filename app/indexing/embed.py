import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("app/data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("app/data/processed/embeddings.npy")
METADATA_PATH = Path("app/data/processed/chunk_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_text_for_embedding(chunk: dict) -> str:
    parts = [
        f"doc_type: {chunk.get('doc_type', '')}",
        f"source: {chunk.get('source', '')}",
        f"service: {chunk.get('service', '')}",
        f"component: {chunk.get('component', '')}",
        f"time_start: {chunk.get('timestamp_start', '')}",
        f"time_end: {chunk.get('timestamp_end', '')}",
        f"text: {chunk.get('chunk_text', '')}",
    ]
    return "\n".join(parts).strip()


def build_metadata(chunks: list[dict]) -> list[dict]:
    metadata = []
    for chunk in chunks:
        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_type": chunk["doc_type"],
                "source": chunk["source"],
                "service": chunk.get("service"),
                "component": chunk.get("component"),
                "timestamp_start": chunk.get("timestamp_start"),
                "timestamp_end": chunk.get("timestamp_end"),
                "chunk_text": chunk["chunk_text"],
            }
        )
    return metadata


def save_metadata(path: Path, metadata: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        raise ValueError("No chunks found in chunks.jsonl")

    texts = [build_text_for_embedding(chunk) for chunk in chunks]
    metadata = build_metadata(chunks)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    save_metadata(METADATA_PATH, metadata)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()