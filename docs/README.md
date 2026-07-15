# docs 폴더

이 폴더에 RAG로 검색할 문서를 넣으세요. 하위 폴더도 재귀적으로 읽습니다.

지원 형식: `.pdf`, `.docx`, `.txt`, `.md`

인덱싱에서 제외되는 파일: `README.md`(지금 이 안내문), 숨김 파일(`.`으로 시작),
Word 잠금 파일(`~$취업규칙.docx` 처럼 문서를 열어둔 동안 생기는 파일).

문서를 넣은 뒤 인덱싱합니다.

```bash
python ingest.py
```

또는 웹 UI 사이드바의 **문서 인덱싱** 버튼을 누르세요.

## 주의

`.gitignore`에 의해 이 폴더의 실제 문서는 커밋되지 않습니다
(`docs/README.md`와 `docs/sample_employee_handbook.md`만 예외).
대외비 문서가 실수로 GitHub에 올라가는 것을 막기 위한 설정입니다.
공개 저장소에 문서까지 올려야 한다면 `.gitignore`에서 `docs/*` 줄을 지우세요.
