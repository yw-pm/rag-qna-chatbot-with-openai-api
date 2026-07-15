"""numpy 기반 벡터 저장소.

Chroma/FAISS 같은 외부 벡터 DB를 쓰지 않는다. 문서 수십~수백 개(조각 수천~수만) 규모에서는
전체 내적 한 번이면 밀리초 단위로 끝나고, 설치가 pip 만으로 끝난다는 게 더 큰 장점이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunker import Chunk

_VECTORS_FILE = "vectors.npz"
_META_FILE = "meta.json"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(
        self,
        vectors: np.ndarray,
        chunks: list[Chunk],
        embedding_model: str,
    ) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("벡터 개수와 조각 개수가 다릅니다.")
        self.vectors = vectors
        self.chunks = chunks
        self.embedding_model = embedding_model

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        """인덱싱된 원본 파일 이름 목록 (중복 제거, 정렬)."""
        return sorted({c.source for c in self.chunks})

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if len(self) == 0:
            return []
        # 저장된 벡터도 질의 벡터도 정규화되어 있으므로 내적 = 코사인 유사도.
        scores = self.vectors @ query_vector
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in top_idx]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / _VECTORS_FILE, vectors=self.vectors)
        meta = {
            "embedding_model": self.embedding_model,
            "dimension": int(self.vectors.shape[1]) if len(self) else 0,
            "chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "page": c.page,
                    "index": c.index,
                }
                for c in self.chunks
            ],
        }
        (directory / _META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        vectors_path = directory / _VECTORS_FILE
        meta_path = directory / _META_FILE
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"'{directory}' 에 인덱스가 없습니다. 문서를 먼저 인덱싱하세요."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path)["vectors"]
        chunks = [
            Chunk(text=c["text"], source=c["source"], page=c["page"], index=c["index"])
            for c in meta["chunks"]
        ]
        return cls(vectors, chunks, meta.get("embedding_model", ""))

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return (directory / _VECTORS_FILE).exists() and (directory / _META_FILE).exists()


def delete_index(directory: Path) -> None:
    for name in (_VECTORS_FILE, _META_FILE):
        (directory / name).unlink(missing_ok=True)
