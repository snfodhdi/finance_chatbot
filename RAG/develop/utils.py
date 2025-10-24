import os
import re
import hashlib
from typing import List, Dict, Any
from datetime import datetime, timedelta

def clean_text(text: str) -> str:
    """텍스트 정리 함수"""
    if not text:
        return ""
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 연속된 줄바꿈 제거
    text = re.sub(r'\n+', '\n', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def truncate_text(text: str, max_length: int = 100) -> str:
    """텍스트 길이 제한"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def generate_file_hash(file_content: bytes) -> str:
    """파일 해시 생성"""
    return hashlib.md5(file_content).hexdigest()

def format_timestamp(timestamp: str) -> str:
    """타임스탬프 포맷팅"""
    try:
        dt = datetime.fromisoformat(timestamp)
        now = datetime.now()
        
        # 오늘인지 확인
        if dt.date() == now.date():
            return dt.strftime("오늘 %H:%M")
        
        # 어제인지 확인
        yesterday = now - timedelta(days=1)
        if dt.date() == yesterday.date():
            return dt.strftime("어제 %H:%M")
        
        # 일주일 이내인지 확인
        if (now - dt).days <= 7:
            weekdays = ["월", "화", "수", "목", "금", "토", "일"]
            weekday = weekdays[dt.weekday()]
            return f"{weekday}요일 {dt.strftime('%H:%M')}"
        
        # 그 외
        return dt.strftime("%m/%d %H:%M")
        
    except:
        return "시간 정보 없음"

def extract_numbers_from_text(text: str) -> List[float]:
    """텍스트에서 숫자 추출"""
    # 쉼표가 포함된 숫자도 처리
    pattern = r'[\d,]+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # 쉼표 제거 후 숫자로 변환
            number = float(match.replace(',', ''))
            numbers.append(number)
        except ValueError:
            continue
    
    return numbers

def format_korean_currency(amount: float) -> str:
    """한국어 통화 포맷팅"""
    if amount >= 1_000_000_000_000:  # 조
        return f"{amount/1_000_000_000_000:.1f}조원"
    elif amount >= 100_000_000:  # 억
        return f"{amount/100_000_000:.1f}억원"
    elif amount >= 10_000:  # 만
        return f"{amount/10_000:.1f}만원"
    else:
        return f"{amount:,.0f}원"

def validate_api_key(api_key: str) -> bool:
    """OpenAI API 키 유효성 검사"""
    if not api_key:
        return False
    
    # sk-로 시작하고 적절한 길이인지 확인
    if not api_key.startswith('sk-'):
        return False
    
    if len(api_key) < 40:
        return False
    
    return True

def safe_filename(filename: str) -> str:
    """안전한 파일명으로 변환"""
    # 특수문자 제거
    safe_name = re.sub(r'[^\w\s-]', '', filename)
    # 공백을 언더스코어로 변경
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    # 길이 제한
    safe_name = safe_name[:50]
    
    return safe_name

def parse_financial_terms(text: str) -> Dict[str, List[str]]:
    """재무 용어 추출"""
    financial_terms = {
        '매출': ['매출', '수익', '매출액', '총매출'],
        '이익': ['영업이익', '순이익', '당기순이익', '세전이익'],
        '자산': ['총자산', '유동자산', '고정자산', '무형자산'],
        '부채': ['총부채', '유동부채', '장기부채'],
        '자본': ['자본총계', '자기자본', '주주지분'],
        '비율': ['ROE', 'ROA', '부채비율', '유동비율']
    }
    
    found_terms = {}
    text_lower = text.lower()
    
    for category, terms in financial_terms.items():
        found = []
        for term in terms:
            if term.lower() in text_lower:
                found.append(term)
        if found:
            found_terms[category] = found
    
    return found_terms

def chunk_overlap_detector(chunks: List[str], overlap_threshold: float = 0.7) -> List[int]:
    """청크 간 중복 감지"""
    overlapping_indices = []
    
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk1_words = set(chunks[i].split())
            chunk2_words = set(chunks[j].split())
            
            if not chunk1_words or not chunk2_words:
                continue
            
            intersection = chunk1_words.intersection(chunk2_words)
            union = chunk1_words.union(chunk2_words)
            
            jaccard_similarity = len(intersection) / len(union)
            
            if jaccard_similarity > overlap_threshold:
                overlapping_indices.extend([i, j])
    
    return list(set(overlapping_indices))

def estimate_token_count(text: str) -> int:
    """토큰 수 추정 (한국어 고려)"""
    # 한국어는 일반적으로 1.5-2 토큰/단어
    korean_chars = len(re.findall(r'[가-힣]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    numbers = len(re.findall(r'\d+', text))
    
    estimated_tokens = korean_chars * 1.5 + english_words + numbers * 0.5
    return int(estimated_tokens)

def create_search_query_variants(query: str) -> List[str]:
    """검색 쿼리 변형 생성"""
    variants = [query]
    
    # 동의어 사전
    synonyms = {
        '매출': ['수익', '매출액', '총매출'],
        '이익': ['수익', '순이익', '영업이익'],
        '성장': ['증가', '상승', '개선'],
        '감소': ['하락', '축소', '저하'],
        '분기': ['Q1', 'Q2', 'Q3', 'Q4', '1분기', '2분기', '3분기', '4분기']
    }
    
    # 동의어로 치환된 변형 생성
    for original, synonym_list in synonyms.items():
        if original in query:
            for synonym in synonym_list:
                variant = query.replace(original, synonym)
                if variant not in variants:
                    variants.append(variant)
    
    return variants

def log_performance(func_name: str, execution_time: float, **kwargs):
    """성능 로깅"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'function': func_name,
        'execution_time': execution_time,
        'metadata': kwargs
    }
    
    # 로그 파일에 기록 (옵션)
    log_file = 'performance.log'
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{log_entry}\n")
    except:
        pass  # 로깅 실패는 무시

def memory_usage_mb() -> float:
    """메모리 사용량 반환 (MB)"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return round(memory_mb, 2)
    except ImportError:
        return 0.0

def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """임시 파일 정리"""
    if not os.path.exists(directory):
        return
    
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=max_age_hours)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_modified_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"임시 파일 삭제: {filename}")
                except Exception as e:
                    print(f"파일 삭제 실패 {filename}: {e}")