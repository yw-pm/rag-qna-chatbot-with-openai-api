"""인덱싱 → 검색 → 답변 생성을 묶은 RAG 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from openai import OpenAI

from rag.chunker import chunk_sections
from rag.config import Settings
from rag.embedder import ProgressFn, embed_query, embed_texts, make_client
from rag.loader import LoadedSection, load_directory, load_path
from rag.prompts import ANSWER_TEMPLATE, CONDENSE_TEMPLATE, SYSTEM_PROMPT
from rag.store import SearchHit, VectorStore

# 이 점수보다 낮은 조각은 질문과 무관하다고 보고 버린다. (코사인 유사도)
MIN_SCORE = 0.2


@dataclass
class Turn:
    question: str
    answer: str


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None
        self._store: VectorStore | None = None

    @property
    def client(self) -> OpenAI:
        """API 클라이언트는 실제로 호출할 때 만든다.

        생성자에서 만들면 키가 없을 때 즉시 예외가 나서, 인덱스 상태 조회처럼
        API가 필요 없는 작업까지 막힌다.
        """
        if self._client is None:
            self._client = make_client(self.settings)
        return self._client

    # ---------- 인덱싱 ----------

    def build_index(
        self,
        sections: list[LoadedSection],
        on_progress: ProgressFn | None = None,
    ) -> VectorStore:
        """섹션 목록을 임베딩해 인덱스를 만들고 디스크에 저장한다."""
        chunks = chunk_sections(
            sections, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("문서에서 텍스트를 한 글자도 추출하지 못했습니다.")

        vectors = embed_texts(
            self.client,
            self.settings.embedding_model,
            [c.text for c in chunks],
            on_progress=on_progress,
        )
        store = VectorStore(vectors, chunks, self.settings.embedding_model)
        store.save(self.settings.index_dir)
        self._store = store
        return store

    def index_directory(
        self, directory: Path | None = None, on_progress: ProgressFn | None = None
    ) -> VectorStore:
        target = directory or self.settings.docs_dir
        sections = load_directory(target)
        if not sections:
            raise ValueError(f"'{target}' 안에서 읽을 수 있는 문서를 찾지 못했습니다.")
        return self.build_index(sections, on_progress=on_progress)

    def index_files(
        self, paths: Sequence[Path], on_progress: ProgressFn | None = None
    ) -> VectorStore:
        sections: list[LoadedSection] = []
        for path in paths:
            sections.extend(load_path(path))
        if not sections:
            raise ValueError("선택한 파일에서 텍스트를 추출하지 못했습니다.")
        return self.build_index(sections, on_progress=on_progress)

    # ---------- 인덱스 접근 ----------

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore.load(self.settings.index_dir)
            self._warn_on_model_mismatch(self._store)
        return self._store

    def has_index(self) -> bool:
        return self._store is not None or VectorStore.exists(self.settings.index_dir)

    def _warn_on_model_mismatch(self, store: VectorStore) -> None:
        if store.embedding_model and store.embedding_model != self.settings.embedding_model:
            raise ValueError(
                f"인덱스는 '{store.embedding_model}' 모델로 만들어졌는데 지금 설정은 "
                f"'{self.settings.embedding_model}' 입니다. 두 모델의 벡터는 호환되지 않으니 "
                f"모델을 되돌리거나 문서를 다시 인덱싱하세요."
            )

    # ---------- 검색 ----------

    def retrieve(self, question: str) -> list[SearchHit]:
        query_vector = embed_query(self.client, self.settings.embedding_model, question)
        hits = self.store.search(query_vector, self.settings.top_k)
        return [h for h in hits if h.score >= MIN_SCORE]

    def condense_question(self, question: str, history: Sequence[Turn]) -> str:
        """후속 질문("그럼 며칠이야?")을 검색 가능한 독립 질문으로 바꾼다."""
        if not history:
            return question

        recent = history[-3:]
        history_text = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in recent)
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": CONDENSE_TEMPLATE.format(
                        history=history_text, question=question
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question

    # ---------- 답변 생성 ----------

    def stream_answer(self, question: str, hits: Sequence[SearchHit]) -> Iterator[str]:
        if not hits:
            yield (
                "제공된 문서에서 관련 내용을 찾지 못했습니다. "
                "질문을 조금 더 구체적으로 바꿔보시거나, 해당 문서가 "
                "인덱싱되어 있는지 확인해 주세요."
            )
            return

        stream = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ANSWER_TEMPLATE.format(
                        context=format_context(hits), question=question
                    ),
                },
            ],
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def ask(self, question: str, history: Sequence[Turn] = ()) -> tuple[str, list[SearchHit]]:
        """스트리밍 없이 한 번에 답을 받는다. (CLI/테스트용)"""
        search_query = self.condense_question(question, history)
        hits = self.retrieve(search_query)
        answer = "".join(self.stream_answer(question, hits))
        return answer, hits


def format_context(hits: Sequence[SearchHit]) -> str:
    """검색된 조각을 [1] 출처: ... 형태로 묶어 프롬프트에 넣을 문자열을 만든다."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] 출처: {hit.chunk.label}\n{hit.chunk.text}")
    return "\n\n---\n\n".join(blocks)
