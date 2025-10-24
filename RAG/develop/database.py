import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
from config import Config

class DatabaseManager:
    def __init__(self):
        print("DatabaseManager 초기화 시작...")
        Config.ensure_directories()

        # ChromaDB 클라이언트 초기화
        print("  - ChromaDB 클라이언트 초기화 중...")
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # 한국어 임베딩 모델 초기화
        print("  - 한국어 임베딩 모델 로딩 중... (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        try:
            self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            print("  ✓ 임베딩 모델 로딩 완료")
        except Exception as e:
            print(f"  ❌ 임베딩 모델 로딩 실패: {e}")
            raise

        # 컬렉션 초기화
        print("  - ChromaDB 컬렉션 초기화 중...")
        self.collection = None
        self._initialize_collection()
        print("DatabaseManager 초기화 완료")
    
    def _initialize_collection(self):
        """컬렉션 초기화 또는 기존 컬렉션 로드"""
        try:
            # 기존 컬렉션이 있는지 확인
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if Config.COLLECTION_NAME in collection_names:
                self.collection = self.client.get_collection(Config.COLLECTION_NAME)
                print(f"기존 컬렉션 '{Config.COLLECTION_NAME}' 로드됨")
            else:
                self.collection = self.client.create_collection(
                    name=Config.COLLECTION_NAME,
                    metadata={"description": "삼성전자 재무제표 문서"}
                )
                print(f"새 컬렉션 '{Config.COLLECTION_NAME}' 생성됨")
                
        except Exception as e:
            print(f"컬렉션 초기화 오류: {e}")
            # 컬렉션을 새로 생성
            try:
                self.collection = self.client.create_collection(
                    name=Config.COLLECTION_NAME,
                    metadata={"description": "삼성전자 재무제표 문서"}
                )
            except Exception as e2:
                print(f"컬렉션 생성 오류: {e2}")
    
    def add_documents(self, chunks: List[str], filename: str) -> bool:
        """문서 청크들을 데이터베이스에 추가"""
        if not self.collection:
            print("컬렉션이 초기화되지 않았습니다.")
            return False
        
        try:
            # 임베딩 생성
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # 고유 ID 생성
            ids = [f"{filename}_{i}_{uuid.uuid4()}" for i in range(len(chunks))]
            
            # 메타데이터 생성
            metadatas = []
            for i, chunk in enumerate(chunks):
                # 청크에서 페이지 번호 추출
                page_num = self._extract_page_number(chunk)
                metadata = {
                    "filename": filename,
                    "chunk_index": i,
                    "page_number": page_num,
                    "source": f"{filename}_page_{page_num}"
                }
                metadatas.append(metadata)
            
            # ChromaDB에 추가
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"{len(chunks)}개 청크가 데이터베이스에 추가되었습니다.")
            return True
            
        except Exception as e:
            print(f"문서 추가 오류: {e}")
            return False
    
    def _extract_page_number(self, chunk: str) -> int:
        """청크에서 페이지 번호 추출"""
        lines = chunk.split('\n')
        for line in lines:
            if line.startswith('페이지:'):
                try:
                    return int(line.split(':')[1].strip())
                except:
                    pass
        return 1
    
    def search_similar_documents(self, query: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """유사한 문서 검색"""
        if not self.collection:
            print("컬렉션이 초기화되지 않았습니다.")
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # 유사도 검색
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 정리
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'relevance_score': 1 - results['distances'][0][i]  # 거리를 관련성 점수로 변환
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"검색 오류: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보 반환"""
        if not self.collection:
            return {"error": "컬렉션이 초기화되지 않았습니다."}
        
        try:
            count = self.collection.count()
            return {
                "name": Config.COLLECTION_NAME,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            return {"error": f"정보 조회 오류: {e}"}
    
    def clear_collection(self) -> bool:
        """컬렉션 초기화 (모든 문서 삭제)"""
        if not self.collection:
            return False

        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(Config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"description": "삼성전자 재무제표 문서"}
            )
            print("컬렉션이 초기화되었습니다.")
            return True

        except Exception as e:
            print(f"컬렉션 초기화 오류: {e}")
            return False

    def delete_documents_by_filename(self, filename: str) -> bool:
        """특정 파일명의 모든 문서 삭제"""
        if not self.collection:
            return False

        try:
            # 해당 파일명을 가진 모든 문서 검색
            results = self.collection.get(
                where={"filename": filename}
            )

            if not results['ids']:
                print(f"'{filename}' 파일의 문서가 없습니다.")
                return False

            # 해당 문서들 삭제
            self.collection.delete(ids=results['ids'])

            deleted_count = len(results['ids'])
            print(f"'{filename}' 파일의 문서 {deleted_count}개가 삭제되었습니다.")
            return True

        except Exception as e:
            print(f"문서 삭제 오류: {e}")
            return False

    def get_uploaded_files(self) -> List[str]:
        """업로드된 파일 목록 반환"""
        if not self.collection:
            return []

        try:
            # 모든 문서의 메타데이터 가져오기
            results = self.collection.get()

            if not results['metadatas']:
                return []

            # 고유한 파일명 추출
            filenames = set()
            for metadata in results['metadatas']:
                if 'filename' in metadata:
                    filenames.add(metadata['filename'])

            return sorted(list(filenames))

        except Exception as e:
            print(f"파일 목록 조회 오류: {e}")
            return []
    
    def get_documents_by_filename(self, filename: str) -> List[Dict]:
        """특정 파일명의 문서들 반환"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=["documents", "metadatas"]
            )
            
            documents = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    doc = {
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"문서 조회 오류: {e}")
            return []