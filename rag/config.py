"""환경변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    index_dir: Path = PROJECT_ROOT / "storage"
    docs_dir: Path = PROJECT_ROOT / "docs"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini").strip(),
            chunk_size=_env_int("CHUNK_SIZE", 800),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            top_k=_env_int("TOP_K", 5),
            index_dir=_resolve(os.getenv("INDEX_DIR", "storage")),
            docs_dir=_resolve(os.getenv("DOCS_DIR", "docs")),
        )

    def validate(self) -> list[str]:
        """설정 문제를 사람이 읽을 수 있는 메시지 목록으로 돌려준다."""
        problems: list[str] = []
        if not self.api_key:
            problems.append("API 키가 비어 있습니다. (OPENAI_API_KEY)")
        if not self.base_url:
            problems.append("Base URL이 비어 있습니다. (OPENAI_BASE_URL)")
        elif not self.base_url.startswith(("http://", "https://")):
            problems.append("Base URL은 http:// 또는 https:// 로 시작해야 합니다.")
        if not self.embedding_model:
            problems.append("임베딩 모델 이름이 비어 있습니다.")
        if not self.chat_model:
            problems.append("채팅 모델 이름이 비어 있습니다.")
        if self.chunk_overlap >= self.chunk_size:
            problems.append("CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")
        return problems


def _resolve(value: str) -> Path:
    path = Path(value.strip() or ".")
    return path if path.is_absolute() else PROJECT_ROOT / path
