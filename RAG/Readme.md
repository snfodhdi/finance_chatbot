# PDF RAG 챗봇

PDF 문서를 기반으로 질문에 답하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 환경 설정

1. .env 파일에 OpenAPI키 키 설정

## 사용법

### 1. 명령줄 사용
```bash
python main.py
```

### 2. 웹 인터페이스 사용
```bash
streamlit run streamlit_app.py
```

## 주요 기능

1. **PDF 텍스트 추출**: PyPDF2를 사용하여 PDF에서 텍스트 추출
2. **텍스트 청킹**: 긴 텍스트를 적절한 크기로 분할
3. **벡터 저장**: Chroma DB에 임베딩과 함께 저장
4. **유사도 검색**: 질문과 관련된 문서 찾기
5. **답변 생성**: OpenAI API를 사용하여 컨텍스트 기반 답변 생성

## 파일 구조

- `rag_chatbot.py`: 메인 챗봇 클래스
- `main.py`: 명령줄 실행 스크립트
- `streamlit_app.py`: 웹 인터페이스
- `requirements.txt`: 필요한 라이브러리
- `.env.example`: 환경 변수 예시