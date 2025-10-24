import os
import base64
import json
from io import BytesIO
from typing import List, Dict, Any, Tuple
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import openai
from tqdm import tqdm
from config import Config
import time
import gc  # 가비지 컬렉션

class PDFProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """PDF를 이미지 리스트로 변환 (PyMuPDF 사용)"""
        pdf_document = None
        try:
            # PDF 유효성 검사
            if not pdf_bytes or len(pdf_bytes) == 0:
                print("❌ 빈 PDF 파일입니다.")
                return []

            # PDF 문서 열기
            try:
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                print(f"❌ PDF 파일을 열 수 없습니다: {e}")
                print("파일이 손상되었거나 올바른 PDF 형식이 아닐 수 있습니다.")
                return []

            # 페이지 수 확인
            page_count = len(pdf_document)
            if page_count == 0:
                print("❌ PDF에 페이지가 없습니다.")
                return []

            print(f"PDF 페이지 수: {page_count}")

            # 페이지 수가 너무 많으면 경고
            if page_count > 50:
                print(f"⚠️  페이지 수가 많습니다 ({page_count}페이지). 처리에 시간이 오래 걸릴 수 있습니다.")

            images = []

            # 각 페이지를 이미지로 변환
            for page_num in range(page_count):
                try:
                    page = pdf_document[page_num]
                    zoom = 2.78
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)

                    # PIL Image로 변환
                    img_data = pix.tobytes("png")
                    img = Image.open(BytesIO(img_data))

                    # 이미지가 너무 크면 리사이즈 (메모리 및 API 제한 고려)
                    max_dimension = 2048  # OpenAI API 권장 크기
                    if img.width > max_dimension or img.height > max_dimension:
                        ratio = min(max_dimension / img.width, max_dimension / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        print(f"  페이지 {page_num + 1} 이미지 리사이즈: {new_size}")

                    images.append(img)

                except Exception as e:
                    print(f"⚠️  페이지 {page_num + 1} 이미지 변환 실패: {e}")
                    # 실패한 페이지는 건너뛰고 계속 진행
                    continue

            return images

        except Exception as e:
            print(f"❌ PDF 변환 중 예상치 못한 오류: {type(e).__name__}: {e}")
            return []

        finally:
            # PDF 문서 닫기
            if pdf_document:
                try:
                    pdf_document.close()
                except:
                    pass
    
    def image_to_base64(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def extract_text_from_image(self, image: Image.Image, page_num: int, max_retries: int = 1) -> Dict[str, Any]:
        """GPT-4 Vision을 사용하여 이미지에서 텍스트와 도표 추출 (2분 제한)"""
        base64_image = self.image_to_base64(image)

        prompt = """
        이 이미지는 삼성전자의 재무제표나 영업 실적 발표자료의 한 페이지입니다.
        다음과 같이 분석해주세요:

        1. 텍스트 내용: 페이지의 모든 텍스트를 정확히 추출해주세요.
        2. 도표/표 데이터: 차트, 그래프, 표가 있다면 데이터를 구조화하여 추출해주세요.
        3. 페이지 제목: 페이지의 주요 제목이나 섹션명을 식별해주세요.

        응답은 다음 JSON 형식으로 해주세요:
        {
            "page_title": "페이지 제목",
            "text_content": "추출된 텍스트 내용",
            "tables": [
                {
                    "table_title": "표 제목",
                    "headers": ["컬럼1", "컬럼2", ...],
                    "data": [["값1", "값2", ...], ...]
                }
            ],
            "charts": [
                {
                    "chart_title": "차트 제목",
                    "chart_type": "차트 유형",
                    "data": {"x축": [], "y축": [], "범례": []}
                }
            ]
        }
        """

        # 재시도 로직 (최대 1회 재시도로 제한)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=Config.VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                        "detail": "high"  # 고품질 분석 유지
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    timeout=110  # 110초로 타임아웃 설정 (재시도 1회 시 총 2분 이내)
                )

                result_text = response.choices[0].message.content

                # JSON 파싱 시도
                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 기본 구조로 반환
                    result = {
                        "page_title": f"페이지 {page_num}",
                        "text_content": result_text,
                        "tables": [],
                        "charts": []
                    }

                result["page_number"] = page_num
                return result

            except openai.APITimeoutError as e:
                print(f"⏰ 페이지 {page_num} - API 타임아웃 (110초 초과), 건너뜁니다...")
                # 타임아웃 시 즉시 빈 결과 반환 (재시도 안함)
                return {
                    "page_number": page_num,
                    "page_title": f"페이지 {page_num} (타임아웃)",
                    "text_content": "",
                    "tables": [],
                    "charts": []
                }

            except openai.RateLimitError as e:
                print(f"페이지 {page_num} - Rate limit 도달, {attempt + 1}/{max_retries} 재시도 중...")
                if attempt < max_retries - 1:
                    wait_time = 5  # 대기 시간을 5초로 단축
                    print(f"{wait_time}초 대기 중...")
                    time.sleep(wait_time)
                else:
                    print(f"페이지 {page_num} - Rate limit 초과, 빈 결과 반환")
                    return {
                        "page_number": page_num,
                        "page_title": f"페이지 {page_num} (Rate limit)",
                        "text_content": "",
                        "tables": [],
                        "charts": []
                    }

            except openai.APIError as e:
                print(f"페이지 {page_num} - OpenAI API 오류: {e}, {attempt + 1}/{max_retries} 재시도 중...")
                if attempt < max_retries - 1:
                    time.sleep(3)  # 대기 시간을 3초로 단축
                else:
                    print(f"페이지 {page_num} - API 오류 지속, 빈 결과 반환")
                    return {
                        "page_number": page_num,
                        "page_title": f"페이지 {page_num} (API 오류)",
                        "text_content": "",
                        "tables": [],
                        "charts": []
                    }

            except Exception as e:
                print(f"페이지 {page_num} - 예상치 못한 오류: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # 대기 시간을 3초로 단축
                else:
                    print(f"페이지 {page_num} - 처리 실패, 빈 결과 반환")
                    return {
                        "page_number": page_num,
                        "page_title": f"페이지 {page_num} (처리 실패)",
                        "text_content": "",
                        "tables": [],
                        "charts": []
                    }

        # 모든 재시도 실패 시 빈 결과 반환
        print(f"페이지 {page_num} - 처리 실패, 빈 결과 반환")
        return {
            "page_number": page_num,
            "page_title": f"페이지 {page_num}",
            "text_content": "",
            "tables": [],
            "charts": []
        }
    
    def save_tables_to_excel(self, extracted_data: List[Dict], filename: str):
        """추출된 표 데이터를 엑셀 파일로 저장"""
        # 테이블이나 차트가 있는지 먼저 확인
        has_data = any(
            page_data.get("tables") or page_data.get("charts")
            for page_data in extracted_data
        )

        if not has_data:
            print("추출된 테이블이나 차트가 없어 엑셀 파일을 생성하지 않습니다.")
            return

        Config.ensure_directories()
        excel_path = os.path.join(Config.EXCEL_DIR, f"{filename}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            sheet_num = 1

            for page_data in extracted_data:
                page_num = page_data.get("page_number", 0)
                tables = page_data.get("tables", [])
                charts = page_data.get("charts", [])
                
                # 표 데이터 저장
                for i, table in enumerate(tables):
                    if table.get("data") and table.get("headers"):
                        df = pd.DataFrame(table["data"], columns=table["headers"])
                        sheet_name = f"Page{page_num}_Table{i+1}"[:31]  # Excel 시트명 길이 제한
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_num += 1
                
                # 차트 데이터 저장
                for i, chart in enumerate(charts):
                    chart_data = chart.get("data", {})
                    if chart_data:
                        # 차트 데이터를 DataFrame으로 변환
                        max_len = max(len(v) if isinstance(v, list) else 1 for v in chart_data.values())
                        
                        df_data = {}
                        for key, value in chart_data.items():
                            if isinstance(value, list):
                                df_data[key] = value + [None] * (max_len - len(value))
                            else:
                                df_data[key] = [value] + [None] * (max_len - 1)
                        
                        df = pd.DataFrame(df_data)
                        sheet_name = f"Page{page_num}_Chart{i+1}"[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_num += 1
    
    def chunk_text(self, text: str, chunk_size: int = Config.CHUNK_SIZE, 
                   chunk_overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
        """텍스트를 청크로 분할"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기 시도
            if end < len(text):
                # 마지막 마침표나 줄바꿈 찾기
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                boundary = max(last_period, last_newline)
                if boundary > start:
                    end = boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Tuple[List[Dict], List[str]]:
        """PDF 전체 처리 파이프라인"""
        print(f"\n{'='*50}")
        print(f"PDF 처리 시작: {filename}")
        print(f"파일 크기: {len(pdf_bytes) / (1024*1024):.2f} MB")
        print(f"{'='*50}\n")

        # PDF를 이미지로 변환
        try:
            images = self.pdf_to_images(pdf_bytes)
            if not images:
                print("❌ PDF를 이미지로 변환할 수 없습니다.")
                return [], []
            print(f"✓ PDF를 {len(images)}개 페이지로 변환 완료")
        except Exception as e:
            print(f"❌ PDF 변환 오류: {type(e).__name__}: {e}")
            return [], []

        # 각 페이지에서 데이터 추출
        extracted_data = []
        all_chunks = []
        failed_pages = []

        for i, image in enumerate(tqdm(images, desc="PDF 페이지 처리 중", unit="페이지")):
            try:
                print(f"\n▶ 페이지 {i+1}/{len(images)} 처리 중...")

                # API 호출 전 약간의 딜레이 추가 (Rate Limit 방지)
                if i > 0:
                    time.sleep(1)  # 1초 대기

                # 페이지 처리 (110초 API 타임아웃 내장)
                page_data = self.extract_text_from_image(image, i+1)
                extracted_data.append(page_data)

                # 텍스트를 청크로 분할
                text_content = page_data.get("text_content", "")
                page_title = page_data.get("page_title", "")

                # 타임아웃으로 건너뛴 페이지인지 확인
                if "(타임아웃)" in page_title:
                    print(f"⏰ 페이지 {i+1}: 2분 타임아웃으로 건너뜀")
                    failed_pages.append(i+1)
                elif text_content:
                    chunks = self.chunk_text(text_content)
                    # 메타데이터 추가
                    for chunk in chunks:
                        chunk_with_metadata = f"""
페이지: {i+1}
제목: {page_data.get('page_title', '')}
내용: {chunk}
                        """.strip()
                        all_chunks.append(chunk_with_metadata)
                    print(f"✓ 페이지 {i+1}: {len(chunks)}개 청크 생성")
                else:
                    print(f"⚠️  페이지 {i+1}: 텍스트 추출 실패 또는 빈 페이지 - 건너뛰고 계속 진행")
                    failed_pages.append(i+1)

            except KeyboardInterrupt:
                print(f"\n\n사용자가 처리를 중단했습니다.")
                print(f"현재까지 {i}개 페이지 처리 완료")
                break

            except Exception as e:
                print(f"❌ 페이지 {i+1} 처리 중 오류: {type(e).__name__}: {e}")
                print(f"   이 페이지를 건너뛰고 다음 페이지로 계속 진행합니다...")
                failed_pages.append(i+1)
                # 오류가 발생해도 계속 진행
                continue

        # 처리 결과 요약
        print(f"\n{'='*50}")
        print(f"PDF 처리 완료: {filename}")
        print(f"총 페이지: {len(images)}")
        print(f"성공한 페이지: {len(images) - len(failed_pages)}")
        print(f"실패한 페이지: {len(failed_pages)}")
        if failed_pages:
            print(f"실패한 페이지 번호: {failed_pages}")
        print(f"추출된 청크 수: {len(all_chunks)}")
        print(f"{'='*50}\n")

       # 표 데이터를 엑셀로 저장
        try:
            self.save_tables_to_excel(extracted_data, filename)
        except Exception as e:
            print(f"⚠️  엑셀 저장 중 오류 (무시하고 계속): {e}")

        return extracted_data, all_chunks