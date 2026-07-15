"""API 키 없이 돌릴 수 있는 테스트.

실행: pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import Chunk, chunk_sections
from rag.loader import (
    LoadedSection,
    UnsupportedFileError,
    _normalize,
    iter_supported_files,
    load_path,
)
from rag.pipeline import format_context
from rag.store import SearchHit, VectorStore


def make_sections(text: str) -> list[LoadedSection]:
    return [LoadedSection(source="규정.md", text=text, page=None)]


class TestChunker:
    def test_short_text_stays_one_chunk(self):
        chunks = chunk_sections(make_sections("연차는 15일이다."), 800, 120)
        assert len(chunks) == 1
        assert chunks[0].text == "연차는 15일이다."

    def test_long_text_is_split(self):
        text = "\n\n".join(f"제{i}조 내용입니다. " * 20 for i in range(10))
        chunks = chunk_sections(make_sections(text), 200, 40)
        assert len(chunks) > 1

    def test_no_chunk_greatly_exceeds_size(self):
        text = "가나다라마바사아자차. " * 500
        chunks = chunk_sections(make_sections(text), 200, 40)
        assert all(len(c.text) <= 200 for c in chunks)

    def test_metadata_is_preserved(self):
        sections = [LoadedSection(source="취업규칙.pdf", text="본문 " * 200, page=7)]
        chunks = chunk_sections(sections, 100, 20)
        assert all(c.source == "취업규칙.pdf" and c.page == 7 for c in chunks)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            chunk_sections(make_sections("본문"), 100, 100)


class TestChunkLabel:
    def test_label_includes_page_when_present(self):
        assert Chunk("t", "a.pdf", 3, 0).label == "a.pdf (p.3)"

    def test_label_omits_page_when_absent(self):
        assert Chunk("t", "a.md", None, 0).label == "a.md"


class TestNormalize:
    def test_collapses_whitespace_and_newlines(self):
        assert _normalize("가   나\r\n\n\n\n다  ") == "가 나\n\n다"


class TestFileDiscovery:
    @staticmethod
    def populate(root: Path) -> None:
        (root / "취업규칙.md").write_text("연차 15일", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "보안규정.txt").write_text("기밀유지", encoding="utf-8")
        (root / "README.md").write_text("이 폴더에 문서를 넣으세요", encoding="utf-8")
        (root / "~$취업규칙.docx").write_bytes(b"word lock file")
        (root / ".hidden.md").write_text("숨김", encoding="utf-8")
        (root / "표.xlsx").write_bytes(b"unsupported")

    def test_finds_documents_recursively(self, tmp_path):
        self.populate(tmp_path)
        names = {p.name for p in iter_supported_files(tmp_path)}
        assert names == {"취업규칙.md", "보안규정.txt"}

    def test_missing_directory_is_empty(self, tmp_path):
        assert list(iter_supported_files(tmp_path / "없음")) == []


class TestLoadPath:
    def test_reads_utf8_markdown(self, tmp_path):
        path = tmp_path / "규정.md"
        path.write_text("제1조 연차는 15일이다.", encoding="utf-8")
        sections = load_path(path)
        assert len(sections) == 1
        assert sections[0].source == "규정.md"
        assert "연차는 15일" in sections[0].text

    def test_reads_cp949_text(self, tmp_path):
        path = tmp_path / "규정.txt"
        path.write_bytes("연차는 15일이다.".encode("cp949"))
        assert "연차는 15일이다." in load_path(path)[0].text

    def test_blank_sections_are_dropped(self, tmp_path):
        path = tmp_path / "빈파일.txt"
        path.write_text("   \n\n  ", encoding="utf-8")
        assert load_path(path) == []

    def test_unsupported_suffix_raises(self, tmp_path):
        path = tmp_path / "표.xlsx"
        path.write_bytes(b"x")
        with pytest.raises(UnsupportedFileError):
            load_path(path)


class TestVectorStore:
    @staticmethod
    def build() -> VectorStore:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
        chunks = [Chunk(f"내용{i}", "규정.md", None, i) for i in range(3)]
        return VectorStore(vectors, chunks, "test-embed")

    def test_search_ranks_by_similarity(self):
        hits = self.build().search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
        assert [h.chunk.index for h in hits] == [0, 2, 1]
        assert hits[0].score == pytest.approx(1.0)

    def test_search_respects_top_k(self):
        assert len(self.build().search(np.array([1.0, 0.0], dtype=np.float32), 2)) == 2

    def test_top_k_larger_than_store_is_safe(self):
        assert len(self.build().search(np.array([1.0, 0.0], dtype=np.float32), 99)) == 3

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            VectorStore(np.zeros((2, 2), dtype=np.float32), [], "m")

    def test_save_and_load_roundtrip(self, tmp_path):
        original = self.build()
        original.save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert len(loaded) == len(original)
        assert loaded.embedding_model == "test-embed"
        assert loaded.chunks[1].text == "내용1"
        np.testing.assert_allclose(loaded.vectors, original.vectors)

    def test_load_without_index_raises(self, tmp_path):
        assert not VectorStore.exists(tmp_path)
        with pytest.raises(FileNotFoundError):
            VectorStore.load(tmp_path)

    def test_sources_are_deduplicated(self):
        store = VectorStore(
            np.zeros((2, 2), dtype=np.float32),
            [Chunk("a", "z.md", None, 0), Chunk("b", "z.md", None, 1)],
            "m",
        )
        assert store.sources == ["z.md"]


class TestFormatContext:
    def test_numbers_and_labels_each_hit(self):
        hits = [
            SearchHit(Chunk("연차 15일", "규칙.pdf", 4, 0), 0.9),
            SearchHit(Chunk("병가 60일", "규칙.pdf", 5, 1), 0.8),
        ]
        context = format_context(hits)
        assert "[1] 출처: 규칙.pdf (p.4)" in context
        assert "[2] 출처: 규칙.pdf (p.5)" in context
        assert "연차 15일" in context
