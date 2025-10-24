import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class Config:
    # OpenAI 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 모델 설정
    VISION_MODEL = "gpt-4o"
    CHAT_MODEL = "gpt-4o-mini"
    
    # ChromaDB 설정
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "samsung_financial_reports"
    
    # 파일 경로 설정
    UPLOAD_DIR = "./uploads"
    EXCEL_DIR = "./excel_data"
    CHAT_HISTORY_DIR = "./chat_history"
    
    # 청크 설정
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # 리랭크 설정
    RERANK_MODEL = "Dongjin-kr/ko-reranker"
    TOP_K = 10
    RERANK_TOP_K = 5
    
    # Streamlit 설정
    PAGE_TITLE = "삼성전자 재무제표 분석 챗봇"
    PAGE_ICON = "📊"

    # RAGAS 평가 설정
    EVALUATION_THRESHOLD = 0.7  # 70점 기준 (0-1 스케일)
    MAX_RETRIES = 3  # 최대 재시도 횟수
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리들을 생성"""
        directories = [
            cls.UPLOAD_DIR,
            cls.EXCEL_DIR, 
            cls.CHAT_HISTORY_DIR,
            cls.CHROMA_DB_PATH
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)