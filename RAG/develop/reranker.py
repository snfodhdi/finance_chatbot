from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import CrossEncoder
from config import Config

class KoreanReranker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cross_encoder = None
        self._initialize_model()
    
    def _initialize_model(self):
        """리랭크 모델 초기화"""
        print("    - 한국어 리랭크 모델 로딩 중... (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        try:
            # 한국어 리랭크 모델 로드 시도
            self.cross_encoder = CrossEncoder('Dongjin-kr/ko-reranker')
            print("    ✓ 한국어 리랭크 모델이 성공적으로 로드되었습니다.")

        except Exception as e:
            print(f"    ⚠️  한국어 리랭크 모델 로드 실패: {e}")
            try:
                print("    - 대안으로 다국어 리랭크 모델 로딩 중...")
                # 대안: 다국어 리랭크 모델 사용
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("    ✓ 다국어 리랭크 모델을 사용합니다.")

            except Exception as e2:
                print(f"    ❌ 리랭크 모델 로드 실패: {e2}")
                print("    ⚠️  리랭커 없이 계속 진행합니다 (검색 품질이 낮아질 수 있습니다)")
                self.cross_encoder = None
    
    def rerank_documents(self, query: str, documents: List[Dict], 
                        top_k: int = Config.RERANK_TOP_K) -> List[Dict]:
        """검색된 문서들을 리랭크"""
        if not self.cross_encoder or not documents:
            # 리랭크 모델이 없거나 문서가 없으면 원본 순서 유지
            return documents[:top_k]
        
        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = []
            for doc in documents:
                doc_text = doc.get('document', '')
                # 문서가 너무 길면 앞부분만 사용
                if len(doc_text) > 512:
                    doc_text = doc_text[:512]
                query_doc_pairs.append([query, doc_text])
            
            # 리랭크 점수 계산
            scores = self.cross_encoder.predict(query_doc_pairs)
            
            # 점수와 문서를 쌍으로 묶어서 정렬
            scored_documents = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(scores[i])
                scored_documents.append(doc_copy)
            
            # 리랭크 점수로 내림차순 정렬
            scored_documents.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return scored_documents[:top_k]
            
        except Exception as e:
            print(f"리랭크 오류: {e}")
            # 오류 발생 시 원본 순서로 반환
            return documents[:top_k]
    
    def calculate_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """문서들의 관련성 점수만 계산"""
        if not self.cross_encoder:
            # 리랭크 모델이 없으면 균등한 점수 반환
            return [0.5] * len(documents)
        
        try:
            query_doc_pairs = []
            for doc in documents:
                # 문서가 너무 길면 앞부분만 사용
                doc_text = doc[:512] if len(doc) > 512 else doc
                query_doc_pairs.append([query, doc_text])
            
            scores = self.cross_encoder.predict(query_doc_pairs)
            return scores.tolist()
            
        except Exception as e:
            print(f"관련성 점수 계산 오류: {e}")
            return [0.5] * len(documents)
    
    def filter_relevant_documents(self, query: str, documents: List[Dict], 
                                threshold: float = 0.3) -> List[Dict]:
        """관련성 임계값 이상인 문서들만 필터링"""
        if not documents:
            return []
        
        # 리랭크 수행
        reranked_docs = self.rerank_documents(query, documents, len(documents))
        
        # 임계값 이상인 문서들만 필터링
        relevant_docs = []
        for doc in reranked_docs:
            rerank_score = doc.get('rerank_score', 0)
            # 리랭크 점수를 0-1 범위로 정규화 (sigmoid 함수 사용)
            normalized_score = 1 / (1 + torch.exp(-torch.tensor(rerank_score)))
            
            if normalized_score >= threshold:
                doc['normalized_rerank_score'] = float(normalized_score)
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def get_best_passages(self, query: str, documents: List[Dict], 
                         max_passages: int = 3) -> List[Dict]:
        """가장 관련성 높은 passage들을 선별"""
        if not documents:
            return []
        
        # 리랭크 수행
        reranked_docs = self.rerank_documents(query, documents, len(documents))
        
        # 상위 passage들 선별
        best_passages = []
        for doc in reranked_docs[:max_passages]:
            # 문서에서 가장 관련성 높은 부분 추출
            document_text = doc.get('document', '')
            sentences = document_text.split('.')
            
            if len(sentences) > 3:
                # 문서가 긴 경우 중간 부분 추출
                mid_start = len(sentences) // 3
                mid_end = mid_start * 2
                best_passage = '.'.join(sentences[mid_start:mid_end]).strip()
            else:
                best_passage = document_text
            
            passage_doc = doc.copy()
            passage_doc['best_passage'] = best_passage
            best_passages.append(passage_doc)
        
        return best_passages
    
    def explain_relevance(self, query: str, document: str, score: float) -> str:
        """관련성 점수에 대한 설명 생성"""
        if score > 0.8:
            relevance_level = "매우 높음"
        elif score > 0.6:
            relevance_level = "높음"
        elif score > 0.4:
            relevance_level = "보통"
        elif score > 0.2:
            relevance_level = "낮음"
        else:
            relevance_level = "매우 낮음"
        
        return f"관련성: {relevance_level} (점수: {score:.3f})"