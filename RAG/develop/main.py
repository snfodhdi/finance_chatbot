import streamlit as st
import os
import time
import warnings
from datetime import datetime
from typing import Optional

# PyTorch 경고 억제
import logging
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*torch.classes.*')

# Streamlit logger 레벨 조정
logging.getLogger('streamlit').setLevel(logging.ERROR)

from config import Config
from pdf_processor import PDFProcessor
from database import DatabaseManager
from chat_manager import ChatManager

# 페이지 설정
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    
    .user-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    
    .assistant-message {
        background-color: #f0f2f6;
        border-left-color: #ff7f0e;
    }
    
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .chat-list-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0.3rem;
        cursor: pointer;
        border: 1px solid #ddd;
    }
    
    .chat-list-item:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        # 초기화 중 메시지 표시
        print("=" * 60)
        print("Streamlit 앱 초기화 중...")
        print("=" * 60)

        try:
            print("1/4: PDF 프로세서 초기화 중...")
            self.pdf_processor = PDFProcessor()
            print("✓ PDF 프로세서 초기화 완료")

            print("2/4: 데이터베이스 초기화 중... (임베딩 모델 로딩, 시간이 걸릴 수 있습니다)")
            self.db_manager = DatabaseManager()
            print("✓ 데이터베이스 초기화 완료")

            print("3/4: 채팅 매니저 초기화 중... (리랭커 모델 로딩, 시간이 걸릴 수 있습니다)")
            # ChatManager에 db_manager 전달하여 중복 생성 방지
            self.chat_manager = ChatManager(db_manager=self.db_manager)
            print("✓ 채팅 매니저 초기화 완료")

            print("4/4: 세션 상태 초기화 중...")
            # 세션 상태 초기화
            self._initialize_session_state()
            print("✓ 세션 상태 초기화 완료")

            print("=" * 60)
            print("✓ Streamlit 앱 초기화 완료!")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 초기화 실패: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if "current_chat_id" not in st.session_state:
            # 자동으로 새 채팅 생성
            st.session_state.current_chat_id = self.chat_manager.create_new_chat()

        if "uploaded_filenames" not in st.session_state:
            st.session_state.uploaded_filenames = []

        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.title("📊 문서 업로드")

            # PDF 업로드 (다중 파일 지원)
            uploaded_files = st.file_uploader(
                "삼성전자 재무제표 PDF를 업로드하세요",
                type=['pdf'],
                accept_multiple_files=True,
                help="여러 PDF 파일을 동시에 선택할 수 있습니다. 선택하면 자동으로 분석이 시작됩니다."
            )

            # 업로드된 파일 처리
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    if file_id not in st.session_state.processed_files:
                        self._process_uploaded_pdf(uploaded_file)
                        st.session_state.processed_files.add(file_id)

            # 업로드된 파일 목록 표시
            if st.session_state.uploaded_filenames:
                st.subheader("📁 업로드된 파일")
                for idx, filename in enumerate(st.session_state.uploaded_filenames, 1):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"{idx}. {filename}")
                    with col2:
                        # 파일 삭제 버튼
                        if st.button("🗑️", key=f"delete_file_{idx}", help=f"{filename} 삭제"):
                            self._delete_file(filename)

            st.divider()

            # 재학습 섹션
            st.subheader("🔄 문서 재학습")
            st.caption("정확도가 낮은 특정 분기의 PDF를 다시 업로드하여 재학습할 수 있습니다.")

            # 데이터베이스에 저장된 파일 목록
            db_files = self.db_manager.get_uploaded_files()
            if db_files:
                selected_file = st.selectbox(
                    "삭제할 파일 선택",
                    db_files,
                    help="데이터베이스에서 해당 파일의 모든 문서를 삭제합니다."
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("선택한 파일 삭제", use_container_width=True):
                        self._delete_file_from_db(selected_file)
                with col2:
                    if st.button("전체 파일 삭제", use_container_width=True):
                        self._delete_all_files_from_db()
            else:
                st.info("데이터베이스에 저장된 파일이 없습니다.")
            
            st.divider()
            
            # 새 채팅 버튼
            if st.button("새 채팅 시작", use_container_width=True):
                self._create_new_chat()
            
            st.divider()

            # 채팅 목록
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("💬 채팅 기록")
            with col2:
                # 전체 삭제 버튼
                if st.button("🗑️", key="delete_all_chats", help="모든 채팅 삭제", use_container_width=True):
                    self._delete_all_chats()

            self._render_chat_list()

            st.divider()
            
            # 데이터베이스 정보
            self._render_database_info()
    
    def _process_uploaded_pdf(self, uploaded_file):
        """업로드된 PDF 처리"""
        try:
            # 파일명에서 확장자 제거
            filename = os.path.splitext(uploaded_file.name)[0]

            # 중복 체크: 같은 파일명이 이미 데이터베이스에 있으면 삭제
            db_files = self.db_manager.get_uploaded_files()
            if filename in db_files:
                st.warning(f"⚠️ '{filename}' 파일이 이미 존재합니다. 기존 데이터를 삭제하고 재학습합니다.")
                self.db_manager.delete_documents_by_filename(filename)

            # 파일 정보 출력
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 파일: {uploaded_file.name} (크기: {file_size_mb:.2f} MB)")

            # 1단계: PDF 바이트 읽기
            try:
                with st.spinner(f"1/3 단계: '{uploaded_file.name}' 파일을 읽는 중..."):
                    pdf_bytes = uploaded_file.read()
                    st.success(f"✓ 파일 읽기 완료 ({len(pdf_bytes)} bytes)")
            except Exception as e:
                st.error(f"❌ 파일 읽기 실패: {str(e)}")
                return

            # 2단계: PDF 처리
            try:
                with st.spinner(f"2/3 단계: PDF 분석 중... (이 작업은 페이지 수에 따라 시간이 걸릴 수 있습니다)"):
                    extracted_data, chunks = self.pdf_processor.process_pdf(pdf_bytes, filename)

                    if not chunks:
                        st.error("❌ PDF에서 텍스트를 추출할 수 없습니다.")
                        st.warning("가능한 원인:")
                        st.write("• PDF가 이미지로만 구성되어 OCR이 필요할 수 있습니다")
                        st.write("• PDF 파일이 손상되었을 수 있습니다")
                        st.write("• API 호출 중 에러가 발생했을 수 있습니다")
                        return

                    st.success(f"✓ PDF 분석 완료 ({len(extracted_data)}페이지, {len(chunks)}개 청크)")

            except Exception as e:
                st.error(f"❌ PDF 분석 실패: {str(e)}")
                st.write(f"에러 타입: {type(e).__name__}")
                import traceback
                st.code(traceback.format_exc())
                return

            # 3단계: 데이터베이스에 추가
            try:
                with st.spinner(f"3/3 단계: 데이터베이스에 저장 중..."):
                    success = self.db_manager.add_documents(chunks, filename)

                    if success:
                        # 파일명 리스트에 추가 (중복 방지)
                        if filename not in st.session_state.uploaded_filenames:
                            st.session_state.uploaded_filenames.append(filename)
                        st.session_state.processing_complete = True
                        st.success(f"✅ '{uploaded_file.name}' 분석이 완료되었습니다!")
                        st.info(f"📄 {len(chunks)}개의 텍스트 청크가 데이터베이스에 저장되었습니다.")

                        # 엑셀 파일 생성 확인
                        excel_path = os.path.join(Config.EXCEL_DIR, f"{filename}.xlsx")
                        if os.path.exists(excel_path):
                            st.info("📊 도표 데이터가 엑셀 파일로 저장되었습니다.")
                    else:
                        st.error("❌ 데이터베이스 저장 중 오류가 발생했습니다.")

            except Exception as e:
                st.error(f"❌ 데이터베이스 저장 실패: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

        except Exception as e:
            st.error(f"❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def _create_new_chat(self):
        """새 채팅 생성"""
        new_chat_id = self.chat_manager.create_new_chat()
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chat_messages = []
        # uploaded_filenames는 유지하여 새 채팅에서도 데이터베이스 활용 가능
        st.rerun()
    
    def _render_chat_list(self):
        """채팅 목록 렌더링"""
        chat_list = self.chat_manager.get_chat_list()

        if not chat_list:
            st.info("이전 채팅이 없습니다.")
            return

        for chat in chat_list[:10]:  # 최근 10개만 표시
            chat_id = chat["chat_id"]
            title = chat["title"]
            updated_at = chat["updated_at"]

            # 시간 포맷팅
            try:
                dt = datetime.fromisoformat(updated_at)
                time_str = dt.strftime("%m/%d %H:%M")
            except:
                time_str = "시간 정보 없음"

            # 채팅 선택 버튼과 삭제 버튼을 나란히 배치
            col1, col2 = st.columns([4, 1])

            with col1:
                # 채팅 선택 버튼
                if st.button(
                    f"💬 {title[:25]}{'...' if len(title) > 25 else ''}",
                    key=f"chat_{chat_id}",
                    help=f"업데이트: {time_str}",
                    use_container_width=True
                ):
                    self._load_chat(chat_id)

            with col2:
                # 삭제 버튼
                if st.button(
                    "🗑️",
                    key=f"delete_{chat_id}",
                    help="채팅 삭제",
                    use_container_width=True
                ):
                    self._delete_chat(chat_id)
    
    def _load_chat(self, chat_id: str):
        """특정 채팅 로드"""
        chat_data = self.chat_manager.load_chat_history(chat_id)

        if chat_data:
            st.session_state.current_chat_id = chat_id
            st.session_state.chat_messages = chat_data.get("messages", [])
            st.rerun()

    def _delete_chat(self, chat_id: str):
        """특정 채팅 삭제"""
        # 삭제하려는 채팅이 현재 열려있는 채팅인 경우
        if st.session_state.current_chat_id == chat_id:
            # 새 채팅 생성
            new_chat_id = self.chat_manager.create_new_chat()
            st.session_state.current_chat_id = new_chat_id
            st.session_state.chat_messages = []

        # 채팅 삭제
        success = self.chat_manager.delete_chat(chat_id)

        if success:
            st.success(f"채팅이 삭제되었습니다.")
            st.rerun()
        else:
            st.error("채팅 삭제에 실패했습니다.")

    def _delete_all_chats(self):
        """모든 채팅 삭제"""
        # 확인 상태 확인
        if st.session_state.get('confirm_delete_all'):
            # 모든 채팅 삭제
            success = self.chat_manager.delete_all_chats()

            if success:
                # 새 채팅 생성
                new_chat_id = self.chat_manager.create_new_chat()
                st.session_state.current_chat_id = new_chat_id
                st.session_state.chat_messages = []
                st.session_state.confirm_delete_all = False

                st.success("모든 채팅이 삭제되었습니다.")
                st.rerun()
            else:
                st.error("채팅 삭제에 실패했습니다.")
                st.session_state.confirm_delete_all = False
        else:
            # 첫 클릭: 확인 요청
            st.session_state.confirm_delete_all = True
            st.warning("⚠️ 다시 클릭하면 모든 채팅이 삭제됩니다!")
            st.rerun()

    def _delete_file(self, filename: str):
        """세션에서 파일 삭제 (UI 목록에서만 제거)"""
        if filename in st.session_state.uploaded_filenames:
            st.session_state.uploaded_filenames.remove(filename)
            st.success(f"'{filename}' 파일이 목록에서 제거되었습니다.")
            st.rerun()

    def _delete_file_from_db(self, filename: str):
        """데이터베이스에서 특정 파일 삭제"""
        success = self.db_manager.delete_documents_by_filename(filename)

        if success:
            # 세션 상태에서도 제거
            if filename in st.session_state.uploaded_filenames:
                st.session_state.uploaded_filenames.remove(filename)

            st.success(f"'{filename}' 파일이 데이터베이스에서 삭제되었습니다.")
            st.info("💡 같은 파일을 다시 업로드하여 재학습할 수 있습니다.")
            st.rerun()
        else:
            st.error(f"'{filename}' 파일 삭제에 실패했습니다.")

    def _delete_all_files_from_db(self):
        """데이터베이스에서 모든 파일 삭제"""
        if st.session_state.get('confirm_clear_db'):
            success = self.db_manager.clear_collection()

            if success:
                st.session_state.uploaded_filenames = []
                st.session_state.processed_files = set()
                st.session_state.processing_complete = False
                st.session_state.confirm_clear_db = False

                st.success("모든 파일이 데이터베이스에서 삭제되었습니다.")
                st.rerun()
            else:
                st.error("데이터베이스 초기화에 실패했습니다.")
                st.session_state.confirm_clear_db = False
        else:
            st.session_state.confirm_clear_db = True
            st.warning("⚠️ 다시 클릭하면 모든 파일이 삭제됩니다!")
            st.rerun()

    def _render_database_info(self):
        """데이터베이스 정보 렌더링"""
        st.subheader("🗄️ 데이터베이스")
        
        db_info = self.db_manager.get_collection_info()
        
        if "error" not in db_info:
            st.info(f"📚 저장된 문서: {db_info.get('document_count', 0)}개")
        else:
            st.warning(db_info["error"])
        
        # 데이터베이스 초기화 버튼
        if st.button("🗑️ 데이터베이스 초기화", use_container_width=True):
            if st.session_state.get('confirm_clear'):
                self.db_manager.clear_collection()
                st.session_state.uploaded_filenames = []
                st.session_state.processed_files = set()
                st.session_state.processing_complete = False
                st.session_state.confirm_clear = False
                st.success("데이터베이스가 초기화되었습니다.")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ 다시 클릭하면 모든 데이터가 삭제됩니다!")
    
    def render_main_content(self):
        """메인 콘텐츠 렌더링"""
        # 헤더
        st.markdown('<div class="main-header">삼성전자 재무제표 분석 챗봇</div>', unsafe_allow_html=True)

        # 현재 상태 표시
        if st.session_state.uploaded_filenames:
            file_count = len(st.session_state.uploaded_filenames)
            if file_count == 1:
                st.success(f"📄 현재 로드된 문서: {st.session_state.uploaded_filenames[0]}")
            else:
                st.success(f"📄 현재 로드된 문서: {file_count}개")
                with st.expander("업로드된 파일 목록 보기", expanded=False):
                    for idx, filename in enumerate(st.session_state.uploaded_filenames, 1):
                        st.write(f"{idx}. {filename}")
        else:
            st.info("📋 PDF 파일을 업로드하여 분석을 시작하세요.")

        # 채팅 인터페이스
        if st.session_state.current_chat_id:
            self._render_chat_interface()
        else:
            st.info("💬 새 채팅을 시작하거나 기존 채팅을 선택하세요.")
    
    def _render_chat_interface(self):
        """채팅 인터페이스 렌더링"""
        # 채팅 메시지 표시
        for message in st.session_state.chat_messages:
            self._render_message(message)

        # 사용자 입력
        user_input = st.chat_input(
            "삼성전자 영업실적에 대해 질문하세요. ex) 25년 2분기 매출 알려줘."
        )

        if user_input and st.session_state.current_chat_id:
            self._handle_user_input(user_input)
    
    def _render_message(self, message: dict):
        """메시지 렌더링"""
        role = message["role"]
        content = message["content"]
        sources = message.get("sources", [])
        timestamp = message.get("timestamp", "")
        
        # 시간 포맷팅
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M")
        except:
            time_str = ""
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
                if time_str:
                    st.caption(f"🕐 {time_str}")
        
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)
                
                # 출처 정보 표시
                if sources:
                    with st.expander("📚 참고 자료", expanded=False):
                        for source in sources:
                            st.write(f"• {source}")
                
                if time_str:
                    st.caption(f"🕐 {time_str}")
    
    def _handle_user_input(self, user_input: str):
        """사용자 입력 처리"""
        # 사용자 메시지 즉시 표시
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # 응답 생성 중 메시지
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                # 업로드된 파일이 있으면 첫 번째 파일명 전달, 없으면 None
                uploaded_filename = st.session_state.uploaded_filenames[0] if st.session_state.uploaded_filenames else None
                response = self.chat_manager.generate_response(
                    st.session_state.current_chat_id,
                    user_input,
                    uploaded_filename
                )

        # 전체 채팅 다시 로드
        self._load_chat(st.session_state.current_chat_id)
        st.rerun()
    
    def run(self):
        """애플리케이션 실행"""
        self.render_sidebar()
        self.render_main_content()

def main():
    """메인 함수"""
    # 초기화 상태를 세션에 저장
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False
        st.session_state.app_instance = None

    # 앱이 아직 초기화되지 않았으면 초기화 시도
    if not st.session_state.app_initialized:
        # 로딩 화면 표시
        with st.container():
            st.title("🚀 앱 초기화 중...")
            st.info("AI 모델을 로딩하고 있습니다. 최초 실행 시 모델 다운로드로 1-2분 정도 소요될 수 있습니다.")
            st.write("터미널/콘솔에서 진행 상황을 확인할 수 있습니다.")

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("PDF 프로세서 초기화 중...")
                progress_bar.progress(25)

                status_text.text("데이터베이스 및 임베딩 모델 로딩 중...")
                progress_bar.progress(50)

                status_text.text("채팅 매니저 및 리랭커 모델 로딩 중...")
                progress_bar.progress(75)

                # 실제 앱 초기화
                app = StreamlitApp()

                status_text.text("완료!")
                progress_bar.progress(100)

                # 초기화 성공
                st.session_state.app_initialized = True
                st.session_state.app_instance = app

                # 페이지 새로고침
                st.rerun()

            except Exception as e:
                st.error(f"❌ 앱 초기화 실패: {e}")
                st.error("터미널/콘솔에서 더 자세한 에러 로그를 확인하세요.")

                with st.expander("에러 상세 정보"):
                    import traceback
                    st.code(traceback.format_exc())

                # 재시도 버튼
                if st.button("🔄 다시 시도"):
                    st.session_state.app_initialized = False
                    st.rerun()

                st.stop()

    # 앱이 초기화되었으면 실행
    if st.session_state.app_initialized and st.session_state.app_instance:
        st.session_state.app_instance.run()

if __name__ == "__main__":
    main()