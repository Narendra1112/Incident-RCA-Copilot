import json
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_PATH = Path("app/data/processed/embeddings.npy")
METADATA_PATH = Path("app/data/processed/chunk_metadata.json")
INDEX_PATH = Path("app/data/processed/faiss.index")
ID_MAP_PATH = Path("app/data/processed/faiss_id_map.json")


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")

    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

    return embeddings.astype(np.float32)


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Metadata file is empty or invalid")

    return metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_id_map(metadata: list[dict]) -> list[dict]:
    id_map = []
    for idx, item in enumerate(metadata):
        id_map.append(
            {
                "faiss_id": idx,
                "chunk_id": item["chunk_id"],
                "doc_id": item["doc_id"],
                "doc_type": item["doc_type"],
                "source": item["source"],
                "service": item.get("service"),
                "component": item.get("component"),
                "timestamp_start": item.get("timestamp_start"),
                "timestamp_end": item.get("timestamp_end"),
            }
        )
    return id_map


def save_id_map(path: Path, id_map: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def main() -> None:
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    metadata = load_metadata(METADATA_PATH)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Mismatch: embeddings count={embeddings.shape[0]} metadata count={len(metadata)}"
        )

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    id_map = build_id_map(metadata)
    save_id_map(ID_MAP_PATH, id_map)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Metadata count: {len(metadata)}")
    print(f"FAISS index ntotal: {index.ntotal}")
    print(f"Saved index to: {INDEX_PATH}")
    print(f"Saved id map to: {ID_MAP_PATH}")


if __name__ == "__main__":
    main()