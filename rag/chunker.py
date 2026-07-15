"""긴 텍스트를 검색 단위(chunk)로 쪼갠다.

문단 → 줄 → 문장 → 글자 순서로 내려가며 자르기 때문에
"제3조(연차휴가) ..." 같은 조문이 문장 중간에서 끊기는 일을 줄여준다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.loader import LoadedSection

_SEPARATORS = ["\n\n", "\n", ". ", "。", "! ", "? ", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    index: int

    @property
    def label(self) -> str:
        """인용 표시에 쓰는 사람이 읽을 수 있는 출처 이름."""
        return f"{self.source} (p.{self.page})" if self.page else self.source


def chunk_sections(
    sections: list[LoadedSection], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")

    chunks: list[Chunk] = []
    for section in sections:
        for piece in _split_text(section.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    source=section.source,
                    page=section.page,
                    index=len(chunks),
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces = [p for p in _recursive_split(text.strip(), chunk_size) if p.strip()]
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, depth: int = 0) -> list[str]:
    """chunk_size 이하가 될 때까지 구분자를 바꿔가며 재귀적으로 자른다."""
    if len(text) <= chunk_size:
        return [text]
    if depth >= len(_SEPARATORS) - 1:
        # 마지막 수단: 글자 수로 강제 절단.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = _SEPARATORS[depth]
    parts = _split_keeping_separator(text, separator)
    if len(parts) <= 1:
        return _recursive_split(text, chunk_size, depth + 1)

    result: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            result.append(part)
        else:
            result.extend(_recursive_split(part, chunk_size, depth + 1))
    return result


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    if not separator:
        return [text]
    parts = re.split(f"({re.escape(separator)})", text)
    merged: list[str] = []
    for i in range(0, len(parts), 2):
        piece = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if piece:
            merged.append(piece)
    return merged


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """작은 조각들을 chunk_size에 가깝게 다시 합치고, 경계에 overlap을 준다."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            chunks.append(current.strip())
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            # overlap을 붙여서 chunk_size를 넘어버리면 overlap을 포기한다.
            current = tail + piece if len(tail) + len(piece) <= chunk_size else piece

    if current.strip():
        chunks.append(current.strip())
    return chunks
