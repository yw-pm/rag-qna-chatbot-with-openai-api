# 📘 문서 Q&A 챗봇 (RAG)

내가 넣은 문서를 근거로 질문에 답하는 RAG 챗봇입니다. 사내 규정, 제품 매뉴얼, 연구 자료,
회의록 등 문서 종류는 가리지 않습니다.

답변마다 **어느 문서 몇 페이지를 근거로 했는지** 함께 보여줍니다. LLM이 그럴듯하게 지어낸
답인지 문서에 실제로 있는 내용인지, 사용자가 원문으로 바로 확인할 수 있습니다.

임베딩·채팅 모델은 **OpenAI 호환 API면 무엇이든** 쓸 수 있습니다. API 키와 Base URL만
바꾸면 OpenAI, Together, Groq, OpenRouter, 자체 호스팅 vLLM/Ollama 등으로 전환됩니다.

## 왜 이렇게 만들었나

- **웹앱(Streamlit)** — 여러 사람이 쓰려면 브라우저 주소만 알면 되어야 합니다. 설치가 필요한 데스크톱 앱은 배포 비용이 큽니다.
- **벡터 DB 없이 numpy** — 문서 수십~수백 개(조각 수천~수만) 규모에서는 전체 내적 계산이 수 밀리초에 끝납니다. Chroma/FAISS를 얹을 이유가 없고, 대신 `pip install`만으로 설치가 끝납니다. 수십만 조각을 넘어가면 그때 교체하세요.
- **문서는 커밋되지 않음** — 민감한 문서가 실수로 공개 저장소에 올라가지 않도록 `.gitignore`가 `docs/`의 실제 파일과 `.env`를 막고 있습니다.

## 화면

아래는 저장소에 포함된 샘플 문서([docs/sample_employee_handbook.md](docs/sample_employee_handbook.md),
가상의 사내 규정)로 질문했을 때의 모습입니다.

```
┌─ 사이드바 ────────┬─ 본문 ─────────────────────────┐
│ API 연결 설정      │  📘 Q&A 챗봇                   │
│  · Base URL       │                                │
│  · API Key        │  🙋 연차는 며칠까지 쓸 수 있어?   │
│  · 임베딩/채팅 모델 │                                │
│                   │  🤖 1년간 80% 이상 출근 시 15일  │
│ 검색 조각 수 [5]   │     이 부여됩니다 [1]. 3년 이상  │
│                   │     이면 2년마다 1일씩 가산되며   │
│ 📂 문서 업로드     │     최대 25일입니다 [1].        │
│ 🔄 문서 인덱싱     │                                │
│                   │     📎 근거 문서 2개 ▾          │
│ 📊 문서 3 / 조각 87│        [1] 취업규칙.pdf (p.4)   │
└───────────────────┴────────────────────────────────┘
```

## 시작하기

### 1. 요구사항

Python 3.10 이상. ([python.org](https://www.python.org/downloads/)에서 설치 —
Windows에서는 설치 화면의 **Add python.exe to PATH** 체크를 꼭 켜세요.)

### 2. 설치

```bash
git clone https://github.com/<your-account>/rag-chatbot.git
cd rag-chatbot

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. API 키 설정

`.env.example`을 `.env`로 복사한 뒤 값을 채웁니다.

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

`.env` 없이 웹 UI 사이드바에서 직접 입력해도 됩니다.

### 4. 문서 넣기

`docs/` 폴더에 문서를 넣습니다. 하위 폴더도 재귀적으로 읽습니다.

지원 형식: `.pdf`, `.docx`, `.txt`, `.md`

샘플 문서가 하나 들어있으니, 바로 시험해본 뒤 지우고 실제 문서로 바꾸셔도 됩니다.

### 5. 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 이 열립니다. 사이드바의 **🔄 문서 인덱싱**을 한 번
누른 뒤 질문하세요. 인덱싱은 문서가 바뀔 때만 다시 하면 됩니다.

터미널에서 인덱싱하려면:

```bash
python ingest.py                  # docs/ 폴더 전체
python ingest.py --docs ./내문서   # 다른 폴더
python ingest.py 매뉴얼.pdf        # 특정 파일만
```

## API 제공자별 설정 예시

| 제공자 | `OPENAI_BASE_URL` | 임베딩 모델 예시 | 채팅 모델 예시 |
| --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | `gpt-4o-mini` |
| Together AI | `https://api.together.xyz/v1` | `BAAI/bge-large-en-v1.5` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| OpenRouter | `https://openrouter.ai/api/v1` | (임베딩 미지원) | `openai/gpt-4o-mini` |
| Ollama (로컬) | `http://localhost:11434/v1` | `nomic-embed-text` | `qwen2.5:14b` |
| vLLM (자체 호스팅) | `http://<서버>:8000/v1` | 서버에 올린 모델명 | 서버에 올린 모델명 |

> 임베딩 API를 제공하지 않는 곳이 꽤 있습니다(OpenRouter 등). 임베딩과 채팅을 서로 다른
> 제공자로 나누고 싶다면 `rag/embedder.py`의 `make_client()`가 별도 설정을 쓰도록
> 고치면 됩니다.

## 동작 방식

```
문서 로드 ─→ 청킹 ─→ 임베딩 ─→ storage/ 에 저장
 (loader)   (chunker)  (embedder)      (store)
                                          │
질문 ─→ 대화 맥락 반영해 질문 재작성 ─→ 임베딩 ─→ 코사인 유사도 상위 K개 검색
                                                        │
                                   근거 조각 + 질문 ─→ LLM ─→ 답변 + [출처 번호]
```

몇 가지 설계 포인트:

- **청킹** — 문단 → 줄 → 문장 순서로 내려가며 자르므로, 한 문단이 문장 중간에서 끊기는 경우가 적습니다. 조각끼리 `CHUNK_OVERLAP` 글자만큼 겹쳐 경계에서 문맥이 사라지는 것을 막습니다.
- **질문 재작성** — "그럼 3년차는요?" 같은 후속 질문은 그대로 검색하면 아무것도 못 찾습니다. 직전 대화 3턴을 참고해 독립적인 질문으로 바꾼 뒤 검색합니다.
- **유사도 하한** — 유사도 0.2 미만 조각은 버립니다. 관련 없는 조각이 딸려 들어가 LLM이 엉뚱한 답을 지어내는 것을 줄입니다. 조정은 `rag/pipeline.py`의 `MIN_SCORE`.
- **출처 강제** — 시스템 프롬프트가 문서에 없는 내용은 "찾을 수 없습니다"라고 답하고, 근거 조각 번호를 `[1]` 형식으로 달도록 지시합니다.
- **안내문 제외** — `README.md`, 숨김 파일, Word 잠금 파일(`~$*.docx`)은 인덱싱하지 않습니다. 폴더 안내문이 검색 근거로 딸려 나오는 것을 막기 위해서입니다.

## 프로젝트 구조

```
rag-chatbot/
├── app.py              # Streamlit 웹 UI
├── ingest.py           # 인덱싱 CLI
├── rag/
│   ├── config.py       # .env 설정 로드 및 검증
│   ├── loader.py       # PDF/DOCX/TXT/MD → 텍스트
│   ├── chunker.py      # 텍스트 → 검색 단위 조각
│   ├── embedder.py     # OpenAI 호환 임베딩 API 호출
│   ├── store.py        # numpy 벡터 저장소 + 코사인 검색
│   ├── prompts.py      # 시스템/답변/질문재작성 프롬프트
│   └── pipeline.py     # 인덱싱·검색·답변 생성 묶음
├── docs/               # 여기에 문서를 넣으세요
├── storage/            # 생성된 인덱스 (자동 생성, 커밋 안 됨)
└── tests/              # API 키 없이 도는 테스트
```

테스트는 `pytest`로 실행합니다. API를 호출하지 않으므로 키 없이 바로 돌아갑니다.

## 튜닝

`.env`에서 조정할 수 있는 값들입니다.

| 값 | 기본 | 언제 올리고 내리나 |
| --- | --- | --- |
| `CHUNK_SIZE` | 800 | 문단이 길고 내용이 서로 얽혀 있으면 ↑(1200~1500). 짧은 FAQ 위주면 ↓(400~600). |
| `CHUNK_OVERLAP` | 120 | 조각 경계에서 답이 잘리는 느낌이면 ↑. `CHUNK_SIZE`의 10~20%가 무난합니다. |
| `TOP_K` | 5 | 답변에 근거가 빠지면 ↑(8~10). 엉뚱한 근거가 섞이면 ↓(3). 올릴수록 비용 증가. |

### 답변 품질 높이기

기본 프롬프트는 문서 종류를 가리지 않도록 중립적으로 쓰여 있습니다. 다루는 문서가
정해져 있다면 `rag/prompts.py`의 `SYSTEM_PROMPT` 첫 문장에 그 맥락을 알려주는 것만으로
답변이 눈에 띄게 좋아집니다.

```python
# 중립 (기본값)
당신은 주어진 문서를 근거로 질문에 답하는 문서 기반 Q&A 어시스턴트입니다.

# 용도를 특정하면 어조와 인용이 더 정확해집니다
당신은 사내 규정을 근거로 직원들의 질문에 답하는 안내 어시스턴트입니다.
```

## 자주 겪는 문제

| 증상 | 원인과 해결 |
| --- | --- |
| `인덱스는 'A' 모델로 만들어졌는데 지금 설정은 'B'` | 임베딩 모델을 바꾸면 기존 벡터와 호환되지 않습니다. 문서를 다시 인덱싱하세요. |
| PDF에서 텍스트가 하나도 안 나옴 | 스캔본(이미지) PDF입니다. OCR을 먼저 돌리거나 텍스트 PDF로 다시 받으세요. |
| `401 Unauthorized` | API 키 또는 Base URL 오류. Base URL이 `/v1`로 끝나는지 확인하세요. |
| 답변이 계속 "찾을 수 없습니다" | 인덱싱을 했는지, `TOP_K`가 너무 작지 않은지, 문서에 실제로 그 내용이 있는지 확인하세요. |
| 한글이 깨져서 인덱싱됨 | `.txt`가 EUC-KR 등일 수 있습니다. UTF-8로 저장하거나 `.docx`로 변환하세요. |

## 여러 사람이 함께 쓸 때

이 프로젝트는 **개인/팀 단위 사용을 전제로** 만들어졌습니다. 조직 단위로 배포한다면
다음을 먼저 검토하세요.

- **인증** — Streamlit 자체에는 로그인이 없습니다. 내부망 전용으로 두거나, 리버스 프록시(nginx + SSO)를 앞에 세우세요.
- **문서 접근 권한** — 지금은 인덱싱된 모든 문서가 모든 사용자에게 검색됩니다. 사람마다 볼 수 있는 문서가 다르다면 인덱스를 분리하거나 조각에 권한 메타데이터를 붙여야 합니다.
- **외부 API 반출** — 질문과 검색된 문서 원문이 API 제공자에게 전송됩니다. 대외비 문서라면 자체 호스팅 vLLM/Ollama처럼 내부 엔드포인트를 쓰거나, 법무/보안팀 검토를 거치세요.
- **동시 사용** — Streamlit 단일 프로세스는 동시 접속이 많아지면 느려집니다. 규모가 커지면 인덱싱을 배치 작업으로 빼고 검색 API를 분리하세요.

## 라이선스

MIT — [LICENSE](LICENSE) 참고.
