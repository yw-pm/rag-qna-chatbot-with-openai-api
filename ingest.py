"""문서 인덱싱 CLI.

사용 예:
    python ingest.py                  # docs/ 폴더 전체를 인덱싱
    python ingest.py --docs ./내문서   # 다른 폴더를 인덱싱
    python ingest.py a.pdf b.docx     # 특정 파일만 인덱싱
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import Settings
from rag.pipeline import RagPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="문서를 벡터 인덱스로 만듭니다.")
    parser.add_argument("files", nargs="*", type=Path, help="인덱싱할 파일 (생략 시 폴더 전체)")
    parser.add_argument("--docs", type=Path, default=None, help="문서 폴더 경로")
    args = parser.parse_args()

    settings = Settings.from_env()
    problems = settings.validate()
    if problems:
        print("설정 오류:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\n.env.example을 복사해 .env를 만들고 값을 채워주세요.", file=sys.stderr)
        return 1

    pipeline = RagPipeline(settings)

    def on_progress(done: int, total: int) -> None:
        bar = "█" * int(24 * done / total)
        print(f"\r  임베딩 {done}/{total} |{bar:<24}|", end="", flush=True)

    try:
        if args.files:
            store = pipeline.index_files(args.files, on_progress=on_progress)
        else:
            store = pipeline.index_directory(args.docs, on_progress=on_progress)
    except Exception as exc:
        print(f"\n실패: {exc}", file=sys.stderr)
        return 1

    print(f"\n\n완료 — 문서 {len(store.sources)}개, 조각 {len(store)}개")
    print(f"저장 위치: {settings.index_dir}")
    for source in store.sources:
        print(f"  · {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
