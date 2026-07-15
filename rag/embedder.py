"""OpenAI 호환 API로 텍스트를 임베딩 벡터로 바꾼다."""

from __future__ import annotations

from typing import Callable

import numpy as np
from openai import OpenAI

from rag.config import Settings

# 한 번의 API 호출에 넣을 텍스트 개수. 너무 크면 요청이 거부될 수 있다.
BATCH_SIZE = 64

ProgressFn = Callable[[int, int], None]


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=60.0)


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    on_progress: ProgressFn | None = None,
) -> np.ndarray:
    """텍스트 목록 → (n, d) 정규화된 float32 행렬.

    L2 정규화를 해두면 나중에 코사인 유사도를 내적 한 번으로 계산할 수 있다.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        # 일부 제공자는 순서를 보장하지 않으므로 index로 되돌린다.
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors.extend(item.embedding for item in ordered)
        if on_progress:
            on_progress(min(start + len(batch), len(texts)), len(texts))

    return _normalize(np.asarray(vectors, dtype=np.float32))


def embed_query(client: OpenAI, model: str, query: str) -> np.ndarray:
    """질문 하나 → (d,) 정규화된 벡터."""
    return embed_texts(client, model, [query])[0]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 0 벡터 나눗셈 방지
    return matrix / norms
