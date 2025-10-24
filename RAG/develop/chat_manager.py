import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
import pandas as pd
from config import Config
from database import DatabaseManager
from reranker import KoreanReranker

# RAGAS 평가 메트릭
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

class ChatManager:
    def __init__(self, db_manager=None):
        print("ChatManager 초기화 시작...")
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

        # db_manager가 전달되지 않으면 새로 생성 (하위 호환성 유지)
        if db_manager is None:
            print("  - DatabaseManager 생성 중...")
            self.db_manager = DatabaseManager()
        else:
            print("  - 기존 DatabaseManager 사용")
            self.db_manager = db_manager

        print("  - KoreanReranker 초기화 중...")
        self.reranker = KoreanReranker()

        Config.ensure_directories()
        print("ChatManager 초기화 완료")
    
    def create_new_chat(self) -> str:
        """새 채팅 세션 생성"""
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        chat_data = {
            "chat_id": chat_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "title": "새 채팅",
            "messages": []
        }
        
        self._save_chat_data(chat_id, chat_data)
        return chat_id
    
    def load_chat_history(self, chat_id: str) -> Optional[Dict]:
        """채팅 히스토리 로드"""
        chat_file = os.path.join(Config.CHAT_HISTORY_DIR, f"{chat_id}.json")
        
        if os.path.exists(chat_file):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"채팅 로드 오류: {e}")
                return None
        return None
    
    def save_message(self, chat_id: str, role: str, content: str, 
                    sources: List[str] = None) -> bool:
        """메시지 저장"""
        chat_data = self.load_chat_history(chat_id)
        if not chat_data:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "sources": sources or []
        }
        
        chat_data["messages"].append(message)
        chat_data["updated_at"] = datetime.now().isoformat()
        
        # 첫 번째 사용자 메시지를 제목으로 설정
        if role == "user" and len(chat_data["messages"]) == 1:
            title = content[:50] + "..." if len(content) > 50 else content
            chat_data["title"] = title
        
        return self._save_chat_data(chat_id, chat_data)
    
    def _save_chat_data(self, chat_id: str, chat_data: Dict) -> bool:
        """채팅 데이터 파일에 저장"""
        try:
            chat_file = os.path.join(Config.CHAT_HISTORY_DIR, f"{chat_id}.json")
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"채팅 저장 오류: {e}")
            return False
    
    def get_chat_list(self) -> List[Dict]:
        """채팅 목록 반환 (최신 순)"""
        chat_list = []
        
        if not os.path.exists(Config.CHAT_HISTORY_DIR):
            return chat_list
        
        for filename in os.listdir(Config.CHAT_HISTORY_DIR):
            if filename.endswith('.json'):
                chat_id = filename[:-5]  # .json 제거
                chat_data = self.load_chat_history(chat_id)
                
                if chat_data:
                    chat_list.append({
                        "chat_id": chat_id,
                        "title": chat_data.get("title", "새 채팅"),
                        "updated_at": chat_data.get("updated_at", ""),
                        "message_count": len(chat_data.get("messages", []))
                    })
        
        # 최신 업데이트 순으로 정렬
        chat_list.sort(key=lambda x: x["updated_at"], reverse=True)
        return chat_list
    
    def get_excel_data_summary(self, filename: str) -> str:
        """엑셀 파일의 데이터 요약 반환"""
        excel_path = os.path.join(Config.EXCEL_DIR, f"{filename}.xlsx")
        
        if not os.path.exists(excel_path):
            return ""
        
        try:
            # 엑셀 파일의 모든 시트 읽기
            xl_file = pd.ExcelFile(excel_path)
            summary_parts = []
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                if not df.empty:
                    summary = f"[{sheet_name}] "
                    summary += f"행: {len(df)}, 열: {len(df.columns)}"
                    
                    # 컬럼명 추가
                    columns = ", ".join(df.columns.astype(str)[:5])  # 처음 5개 컬럼만
                    if len(df.columns) > 5:
                        columns += "..."
                    summary += f", 컬럼: {columns}"
                    
                    summary_parts.append(summary)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            print(f"엑셀 요약 오류: {e}")
            return ""
    
    def generate_response(self, chat_id: str, user_query: str,
                         uploaded_filename: str = None, max_retries: int = 3) -> str:
        """사용자 쿼리에 대한 응답 생성 (검증 모델 포함)"""
        # 1. 사용자 메시지 저장
        self.save_message(chat_id, "user", user_query)

        # 2. 컨텍스트 검색
        context_sources = []
        context_text = ""

        # ChromaDB에서 관련 문서 검색 (항상 수행)
        search_results = self.db_manager.search_similar_documents(
            user_query, Config.TOP_K
        )

        if search_results:
            # 리랭크 수행
            reranked_results = self.reranker.rerank_documents(
                user_query, search_results, Config.RERANK_TOP_K
            )

            # 컨텍스트 구성
            context_parts = []
            for result in reranked_results:
                doc = result['document']
                metadata = result['metadata']
                source = f"{metadata.get('filename', '')} (페이지 {metadata.get('page_number', '')})"

                context_parts.append(f"[출처: {source}]\n{doc}")
                context_sources.append(source)

            context_text = "\n\n".join(context_parts)

        # 엑셀 데이터 요약 추가 (uploaded_filename이 있을 때만)
        if uploaded_filename:
            excel_summary = self.get_excel_data_summary(uploaded_filename)
            if excel_summary:
                context_text += f"\n\n[엑셀 데이터 요약]\n{excel_summary}"
                context_sources.append(f"{uploaded_filename}.xlsx")

        # 3. 채팅 히스토리 로드
        chat_data = self.load_chat_history(chat_id)
        recent_messages = []

        if chat_data and chat_data.get("messages"):
            # 최근 5개 메시지만 컨텍스트로 사용
            recent_messages = chat_data["messages"][-10:]

        # 4. GPT-4o mini로 응답 생성 (검증 기반 재생성 로직)
        response, score, attempt = self._generate_validated_response(
            user_query, context_text, recent_messages, max_retries
        )

        # 5. 응답 저장 (평가 정보 추가)
        evaluation_info = f"\n\n[품질 평가: {score}점, 시도 횟수: {attempt}회]"
        self.save_message(chat_id, "assistant", response + evaluation_info, context_sources)

        return response

    def _generate_validated_response(self, user_query: str, context: str,
                                    chat_history: List[Dict], max_retries: int = 3) -> tuple:
        """검증 기반 응답 생성 - 70점 이상 나올 때까지 재시도"""
        best_response = ""
        best_score = 0

        for attempt in range(1, max_retries + 1):
            print(f"\n=== 답변 생성 시도 {attempt}/{max_retries} ===")

            # 답변 생성
            response = self._generate_gpt_response(user_query, context, chat_history)

            # 답변 평가
            score = self._evaluate_response(user_query, response, context)
            print(f"평가 점수: {score}점")

            # 최고 점수 갱신
            if score > best_score:
                best_score = score
                best_response = response

            # 70점 이상이면 통과
            if score >= 70:
                print(f"✓ 품질 기준 통과 ({score}점)")
                return response, score, attempt
            else:
                print(f"✗ 품질 기준 미달 ({score}점 < 70점)")
                if attempt < max_retries:
                    print("답변 재생성 중...")

        # 최대 시도 횟수 도달 - 가장 좋은 답변 반환
        print(f"\n최대 시도 횟수 도달. 최고 점수 답변 반환 ({best_score}점)")
        return best_response, best_score, max_retries

    def _evaluate_response(self, user_query: str, response: str, context: str) -> int:
        """RAGAS를 사용하여 답변 평가 (0-100점)"""
        try:
            # RAGAS 평가를 위한 데이터 준비
            # context가 여러 문서로 구성되어 있으므로 리스트로 분리
            if context:
                # 컨텍스트를 개별 문서로 분리 (줄바꿈 기준)
                contexts = [ctx.strip() for ctx in context.split('\n\n') if ctx.strip()]
            else:
                contexts = ["정보 없음"]

            # RAGAS 평가용 데이터셋 구성
            eval_data = {
                'question': [user_query],
                'answer': [response],
                'contexts': [contexts]
            }

            dataset = Dataset.from_dict(eval_data)

            # RAGAS 메트릭으로 평가
            # faithfulness: 답변이 컨텍스트에 충실한가 (0-1)
            # answer_relevancy: 답변이 질문과 관련있는가 (0-1)
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy]
            )

            # 디버깅: result 구조 출력
            print(f"\n[DEBUG] Result type: {type(result)}")
            print(f"[DEBUG] Result keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")
            print(f"[DEBUG] Result content: {result}")

            # 점수 추출 및 계산
            # RAGAS result는 dict이며, 각 메트릭은 리스트 또는 단일 값일 수 있음
            faithfulness_score = result['faithfulness']
            relevancy_score = result['answer_relevancy']

            # 리스트인 경우 첫 번째 값 또는 평균 사용
            if isinstance(faithfulness_score, (list, tuple)):
                faithfulness_score = faithfulness_score[0] if len(faithfulness_score) > 0 else 0.5
            if isinstance(relevancy_score, (list, tuple)):
                relevancy_score = relevancy_score[0] if len(relevancy_score) > 0 else 0.5

            # float으로 변환
            faithfulness_score = float(faithfulness_score)
            relevancy_score = float(relevancy_score)

            # 가중 평균 계산 (충실도 60%, 관련성 40%)
            final_score = (faithfulness_score * 0.6 + relevancy_score * 0.4)

            # 0-1 스케일을 0-100으로 변환
            score_100 = int(final_score * 100)

            print(f"\n=== RAGAS 평가 결과 ===")
            print(f"Faithfulness (충실도): {faithfulness_score:.2f}")
            print(f"Answer Relevancy (관련성): {relevancy_score:.2f}")
            print(f"최종 점수: {score_100}점")

            return score_100

        except Exception as e:
            print(f"RAGAS 평가 오류: {e}")
            import traceback
            print(f"상세 오류:\n{traceback.format_exc()}")
            print("기본값 50점 반환")
            return 50  # 오류 시 중간 점수 반환

    def _generate_gpt_response(self, user_query: str, context: str,
                              chat_history: List[Dict]) -> str:
        """GPT-4o mini를 사용하여 응답 생성"""
        system_prompt = """
당신은 삼성전자의 재무제표와 영업 실적을 분석하는 회계재무담당자입니다.

역할:
- 제공된 문서와 데이터를 바탕으로 정확하고 유용한 정보를 제공
- 재무 데이터를 이해하기 쉽게 설명
- 질문에 대한 근거를 명확히 제시
- 한국어로 자연스럽고 전문적인 답변 제공

지침:
1. 제공된 컨텍스트 정보를 우선적으로 활용하세요
2. 답변의 근거가 되는 문서나 페이지를 명시하세요
3. 데이터가 없거나 불분명한 경우 솔직히 말하세요
4. 재무 용어는 이해하기 쉽게 설명하세요
5. 구체적인 수치나 데이터가 있으면 정확히 인용하세요
6. 질문자가 언급하는 연도나 분기를 정확하게 지켜서 질문하세요.
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 채팅 히스토리 추가
        for msg in chat_history[:-1]:  # 마지막 메시지(현재 질문) 제외
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # 현재 질문과 컨텍스트 추가
        user_content = f"질문: {user_query}"
        if context:
            user_content += f"\n\n참고 자료:\n{context}"
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = self.client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"GPT 응답 생성 오류: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
    
    def delete_chat(self, chat_id: str) -> bool:
        """채팅 삭제"""
        try:
            chat_file = os.path.join(Config.CHAT_HISTORY_DIR, f"{chat_id}.json")
            if os.path.exists(chat_file):
                os.remove(chat_file)
                return True
            return False
        except Exception as e:
            print(f"채팅 삭제 오류: {e}")
            return False

    def delete_all_chats(self) -> bool:
        """모든 채팅 기록 삭제"""
        try:
            if not os.path.exists(Config.CHAT_HISTORY_DIR):
                return True

            deleted_count = 0
            for filename in os.listdir(Config.CHAT_HISTORY_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(Config.CHAT_HISTORY_DIR, filename)
                    os.remove(file_path)
                    deleted_count += 1

            print(f"총 {deleted_count}개의 채팅이 삭제되었습니다.")
            return True
        except Exception as e:
            print(f"전체 채팅 삭제 오류: {e}")
            return False