"""Q&A 챗봇 — Streamlit 웹 UI.

실행: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag.config import Settings
from rag.loader import SUPPORTED_SUFFIXES, iter_supported_files
from rag.pipeline import RagPipeline, Turn
from rag.store import SearchHit, delete_index

st.set_page_config(page_title="Q&A 챗봇", page_icon="📘", layout="centered")

UPLOAD_TYPES = [s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)]


# ---------------------------------------------------------------- 상태 관리


def init_state() -> None:
    if "initialized" in st.session_state:
        return
    env = Settings.from_env()
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.cfg = {
        "api_key": env.api_key,
        "base_url": env.base_url,
        "embedding_model": env.embedding_model,
        "chat_model": env.chat_model,
        "top_k": env.top_k,
    }
    st.session_state.env = env


def current_settings() -> Settings:
    env: Settings = st.session_state.env
    cfg = st.session_state.cfg
    return Settings(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        embedding_model=cfg["embedding_model"],
        chat_model=cfg["chat_model"],
        chunk_size=env.chunk_size,
        chunk_overlap=env.chunk_overlap,
        top_k=cfg["top_k"],
        index_dir=env.index_dir,
        docs_dir=env.docs_dir,
    )


def get_pipeline(settings: Settings) -> RagPipeline:
    """설정이 바뀌면 파이프라인을 새로 만든다."""
    signature = (settings.api_key, settings.base_url, settings.embedding_model)
    if st.session_state.get("pipeline_signature") != signature:
        st.session_state.pipeline = RagPipeline(settings)
        st.session_state.pipeline_signature = signature
    pipeline: RagPipeline = st.session_state.pipeline
    pipeline.settings = settings  # top_k, chat_model 같은 값은 즉시 반영
    return pipeline


def history() -> list[Turn]:
    turns: list[Turn] = []
    messages = st.session_state.messages
    for i in range(0, len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append(Turn(messages[i]["content"], messages[i + 1]["content"]))
    return turns


# ---------------------------------------------------------------- 사이드바


def render_sidebar(settings: Settings) -> None:
    cfg = st.session_state.cfg

    with st.sidebar:
        st.header("⚙️ 설정")

        with st.expander("API 연결", expanded=not settings.api_key):
            cfg["base_url"] = st.text_input(
                "Base URL",
                value=cfg["base_url"],
                help="OpenAI 호환 엔드포인트. 보통 /v1 로 끝납니다.",
            ).strip()
            cfg["api_key"] = st.text_input(
                "API Key",
                value=cfg["api_key"],
                type="password",
                help=".env 파일에 OPENAI_API_KEY로 넣어두면 매번 입력하지 않아도 됩니다.",
            ).strip()
            cfg["embedding_model"] = st.text_input(
                "임베딩 모델", value=cfg["embedding_model"]
            ).strip()
            cfg["chat_model"] = st.text_input("채팅 모델", value=cfg["chat_model"]).strip()

        cfg["top_k"] = st.slider(
            "검색할 문서 조각 수",
            min_value=1,
            max_value=15,
            value=cfg["top_k"],
            help="많을수록 근거는 풍부해지지만 비용과 응답 시간이 늘어납니다.",
        )

        st.divider()
        render_document_section(settings)
        st.divider()
        render_index_status(settings)

        if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_document_section(settings: Settings) -> None:
    st.subheader("📂 문서")

    uploaded = st.file_uploader(
        "문서 업로드",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"지원 형식: {', '.join(UPLOAD_TYPES)}",
    )
    if uploaded:
        # 업로더는 리런마다 같은 파일을 다시 넘겨주므로, 이미 저장한 것은 건너뛴다.
        already_saved: set[str] = st.session_state.setdefault("saved_uploads", set())
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        newly_saved = []
        for file in uploaded:
            key = f"{file.name}:{file.size}"
            if key in already_saved:
                continue
            (settings.docs_dir / file.name).write_bytes(file.getbuffer())
            already_saved.add(key)
            newly_saved.append(file.name)
        if newly_saved:
            st.success(
                f"{len(newly_saved)}개 파일 저장 완료. 아래 '문서 인덱싱'을 눌러주세요."
            )

    files = list(iter_supported_files(settings.docs_dir))
    if files:
        with st.expander(f"'{settings.docs_dir.name}' 폴더 파일 {len(files)}개"):
            for path in files:
                st.caption(f"• {path.relative_to(settings.docs_dir)}")
    else:
        st.info(f"'{settings.docs_dir.name}' 폴더가 비어 있습니다.")

    if st.button(
        "🔄 문서 인덱싱",
        type="primary",
        use_container_width=True,
        disabled=not files,
    ):
        run_indexing(settings)


def run_indexing(settings: Settings) -> None:
    problems = settings.validate()
    if problems:
        st.error("설정을 먼저 확인해주세요:\n\n" + "\n".join(f"- {p}" for p in problems))
        return

    progress = st.progress(0.0, text="문서를 읽는 중…")

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"임베딩 생성 중… {done}/{total} 조각")

    try:
        pipeline = get_pipeline(settings)
        store = pipeline.index_directory(on_progress=on_progress)
    except Exception as exc:  # 네트워크·인증·파싱 오류를 UI에 그대로 보여준다.
        progress.empty()
        st.error(f"인덱싱 실패: {exc}")
        return

    progress.empty()
    st.success(f"완료! 문서 {len(store.sources)}개 → 조각 {len(store)}개")
    st.session_state.messages = []


def render_index_status(settings: Settings) -> None:
    st.subheader("📊 인덱스")
    try:
        pipeline = get_pipeline(settings)
        if not pipeline.has_index():
            st.warning("아직 인덱스가 없습니다. 문서를 인덱싱해주세요.")
            return
        store = pipeline.store
    except Exception as exc:
        st.error(str(exc))
        return

    col1, col2 = st.columns(2)
    col1.metric("문서", len(store.sources))
    col2.metric("조각", len(store))

    if st.button("인덱스 삭제", use_container_width=True):
        delete_index(settings.index_dir)
        st.session_state.pop("pipeline", None)
        st.session_state.pop("pipeline_signature", None)
        st.rerun()


# ---------------------------------------------------------------- 본문


def render_sources(hits: list[SearchHit]) -> None:
    if not hits:
        return
    with st.expander(f"📎 근거 문서 {len(hits)}개"):
        for i, hit in enumerate(hits, start=1):
            st.markdown(f"**[{i}] {hit.chunk.label}** · 유사도 {hit.score:.3f}")
            st.caption(hit.chunk.text)
            if i < len(hits):
                st.divider()


def handle_question(question: str, settings: Settings) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            pipeline = get_pipeline(settings)
            with st.spinner("문서를 찾는 중…"):
                search_query = pipeline.condense_question(question, history())
                hits = pipeline.retrieve(search_query)
            answer = st.write_stream(pipeline.stream_answer(question, hits))
        except Exception as exc:
            st.error(f"답변 생성 실패: {exc}")
            st.session_state.messages.pop()
            return

        render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )


def main() -> None:
    init_state()
    settings = current_settings()
    render_sidebar(settings)

    st.title("📘 Q&A 챗봇")
    st.caption("업로드한 문서를 근거로 답변합니다. 근거 문서는 답변 아래에서 확인하세요.")

    problems = settings.validate()
    if problems:
        st.warning(
            "사이드바에서 API 설정을 완료해주세요:\n\n"
            + "\n".join(f"- {p}" for p in problems)
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("hits", []))

    pipeline_ready = not problems
    try:
        pipeline_ready = pipeline_ready and get_pipeline(settings).has_index()
    except Exception:
        pipeline_ready = False

    placeholder = (
        "예) 연차휴가는 며칠까지 쓸 수 있나요?"
        if pipeline_ready
        else "먼저 API 설정과 문서 인덱싱을 완료해주세요."
    )
    question = st.chat_input(placeholder, disabled=not pipeline_ready)
    if question:
        handle_question(question, settings)


if __name__ == "__main__":
    main()
