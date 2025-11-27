"""
회의록 검색 챗봇 FastAPI 앱
- 엔드포인트만 포함
- 모든 로직은 별도 모듈로 분리
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging

from .search import parse_meetings_list
from .llm import call_hyperclova_rag

# 모델
from .models import ChatRequest, ChatResponse

# 설정
from .config import ENABLE_PERSONA

# 데이터베이스 & 컨텍스트
from .database import init_db_connection, test_db_connection
from .context import init_redis_client, get_context, save_context, delete_context

# 검색
from .search import (
    is_off_topic_query, 
    get_off_topic_response,
    parse_date_from_query,
    parse_status_from_query,
    search_meetings_direct,
    search_with_persona,
    has_search_intent,
    extract_keywords_from_query
)

# 포맷팅
from .formatting import (
    format_single_meeting,
    format_single_meeting_with_persona,
    format_multiple_meetings_short,
    calculate_shown_counts
)

# 선택 처리
from .selection import handle_selection

from datetime import datetime
import re

# ============================================================
# 로깅 설정
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 앱 초기화
# ============================================================

app = FastAPI(title="회의록 검색 챗봇")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 클라이언트 초기화
# ============================================================

# # DB 연결 테스트 (서버 시작 시)
# from .database import test_db_connection
# if not test_db_connection():
#     print("[⚠️] MySQL 연결 실패 - 서버는 시작되지만 DB 기능은 작동하지 않을 수 있습니다.")

# Redis 초기화
redis_client = init_redis_client()

# ============================================================
# Phase 2-A: Template 페르소나 함수들
# ============================================================

def get_user_id_by_name(user_name: str) -> int:
    """사용자 이름으로 user_id 조회"""
    from .database import get_db_connection
    
    with get_db_connection() as conn:
        if not conn:
            return 1
        
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM user WHERE name = %s", (user_name,))
            result = cursor.fetchone()
            return result['id'] if result else 1
        except Exception as e:
            print(f"[ERROR] user_id 조회 실패: {e}")
            return 1
        finally:
            cursor.close()

# ============================================================
# COUNT 질문 감지 함수
# ============================================================
def is_count_question(query: str) -> bool:
    """COUNT 질문 감지"""
    query_clean = query.replace(" ", "").lower()
    
    # 1. 기본 패턴
    safe_patterns = ["하나", "하나야", "하나임", "하나니", "하나냐", "몇", "몇개", "몇번", "개수", "총", "횟수"]
    if any(p in query_clean for p in safe_patterns):
        return True
    
    # 2. 숫자 + 단위
    if re.search(r'[0-9일이삼사오육칠팔구십]+개|[0-9]+번', query_clean):
        return True
    
    # 3. 제한 표현
    if re.search(r'(그거|저거|이거|그것|저것|이것)(밖에|뿐|만)', query_clean):
        return True
    
    # 4. 종료 확인 (Task 키워드 제외)
    if re.search(r'(끝|다|전부|모두)(이야|임|야|니|냐|인가)', query_clean):
        # 컨텍스트 기반 확인 질문은 제외 ("이게 끝이야?", "그게 다야?" 등)
        if any(prefix in query_clean for prefix in ['이게', '그게', '저게', '그거', '저거', '이거']):
            return False
    
        # Task 키워드 있으면 제외
        if not any(kw in query for kw in ['사람', '누가', '담당', '할일', '할 일', '멤버', '참석']):
            return True
    
    # 5. 추가 확인
    if re.search(r'^(더|또)\s*(있|없)', query):
        if any(ref in query for ref in ['그거', '저거', '이거', '그것', '저것', '이것', '그', '저', '이']) or len(query) < 15:
            return True
    
    return False

# ============================================================
# 컨텍스트 의존 질문 감지
# ============================================================

def is_context_dependent_query(query: str) -> bool:
    """컨텍스트 의존적인 짧은 질문인지 판단"""
    # 대명사/지시어
    pronouns = ['그', '그거', '그것', '그게', '저', '저거', '저것', '저게', '이', '이거', '이것', '이게']
    
    # 짧은 질문 패턴
    short_patterns = [
        '다른', '그 외', '누가', '누구', '언제', '어디서', 
        '뭐', '무엇', '어떻게', '왜', '할 일', '할일',
        '그럼', '그러면', '그래서', '또', '다시', '아니',
        '사람', '담당'
    ]
    
    query_lower = query.lower().strip()
    
    # 1. 대명사로 시작하거나 포함
    for pronoun in pronouns:
        if query_lower.startswith(pronoun) or f" {pronoun} " in f" {query_lower} ":
            return True
    
    # 2. 10글자 이하이고 패턴 포함
    if len(query) <= 15 and any(p in query for p in short_patterns):
        return True
    
    return False

# ============================================================
# 다중 회의 처리
# ============================================================
def handle_multiple_meetings(lambda_response: str, user_query: str, 
                            request: ChatRequest, session_id: str) -> ChatResponse:
    """여러 회의 발견 시 명확화 질문"""
    
    # Lambda 응답에서 회의 목록 파싱
    meetings = parse_meetings_list(lambda_response)
    
    if not meetings:
        # 파싱 실패 → Lambda 원본 반환
        return ChatResponse(
            answer=lambda_response,
            history=request.history + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": lambda_response}
            ],
            source="lambda_raw",
            session_id=session_id
        )
    
    total_count = len(meetings)
    
    # ========== 10개 이상이면 재검색 유도 ==========
    if total_count >= 10:
        too_many_msg = f"""회의록 {total_count}개를 찾았어요! 너무 많네요. 😅

더 구체적인 키워드로 다시 검색해주시겠어요?

💡 검색 팁:
- 날짜 추가: "이번주 기획 회의", "1월 15일 회의"
- 주제 명확히: "마케팅", "디자인", "개발"
- 참석자 추가: "김철수가 참석한 회의"

예시: "이번주 마케팅 회의", "1월 디자인 회의" """
        
        print(f"[DEBUG] 너무 많은 결과: {total_count}개 → 재검색 유도")
        
        return ChatResponse(
            answer=too_many_msg,
            source="too_many_meetings",
            session_id=session_id
        )
    
    # ========== 10개 미만: HyperCLOVA X에게 판단 맡기기 ==========
    print(f"[DEBUG] {total_count}개 회의 발견 → HyperCLOVA X RAG 호출")
    
    # HyperCLOVA X RAG 호출 (여러 회의 처리 규칙 적용)
    rag_answer = call_hyperclova_rag(user_query, lambda_response)
    
    if rag_answer:
        print(f"✅ RAG 답변 생성 성공!")

        shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])

        # ========== 컨텍스트 저장 (선택 가능하도록) ==========
        context = {
            'state': 'awaiting_selection',
            'shown_completed': shown_completed,
            'shown_scheduled': shown_scheduled,
            'meetings': meetings[:5],  # 상위 5개만
            'total_count': total_count,
            'original_query': user_query,
            'lambda_response': lambda_response
        }
        save_context(session_id, context)
        print(f"[DEBUG] 컨텍스트 저장 완료: {len(meetings[:5])}개 회의")
        
        return ChatResponse(
            answer=rag_answer,
            source="multiple_meetings",
            session_id=session_id
        )
    
    # RAG 실패 시 Lambda 원본 반환
    print(f"⚠️ RAG 실패 → Lambda 원본 답변 반환")
    
    return ChatResponse(
        answer=lambda_response,
        source="lambda_raw",
        session_id=session_id
    )

def is_obvious_pattern(user_query: str) -> bool:
    """
    명확한 패턴인지 확인 (LLM 호출 불필요)
    """
    query_strip = user_query.strip()
    
    obvious_patterns = [
        # 번호 선택
        query_strip.isdigit(),
        # 날짜 선택 (정규식)
        bool(re.match(r'^\d{1,2}월\s?\d{1,2}일$', query_strip)),
        # 명확한 회의명 (길고 물음표 없음)
        (len(user_query) > 8 and '회의' in user_query and not any(w in user_query for w in ['?', '뭐', '어떤', '있어'])),
    ]
    return any(obvious_patterns)

def is_detail_question(query: str, context: dict) -> bool:
    """
    회의 상세 질문인지 판단
    
    Args:
        query: 사용자 질문
        context: 현재 컨텍스트
    
    Returns:
        상세 질문 여부
    """
    # 컨텍스트에 선택된 회의가 없으면 False
    if not context or context.get('state') != 'meeting_selected':
        return False
    
    # 상세 질문 패턴
    detail_patterns = [
        '예산', '얼마', '금액', '비용',
        '누가', '누구', '발표자', '담당자',
        '어떻게', '방법', '과정',
        '왜', '이유', '목적',
        '언제', '시간', '일정', '몇 분', '얼마나', '기간',
        '결론', '결과', '결정',
        '내용', '주요', '핵심', '요약',
        '발표', '논의', '합의', '의견',
        '도구', '기술', '방식', '참석자', '발언'
    ]
    
    # 제외 패턴 (다른 intent와 구분)
    exclude_patterns = [
        '할일', 'task',  # task_search
        '참석', '멤버',  # participant_search
        '검색', '키워드'  # meeting_search, keyword_search
    ]
    
    # 제외 패턴이 있으면 False
    if any(pattern in query for pattern in exclude_patterns):
        return False
    
    # 상세 질문 패턴이 있으면 True
    return any(pattern in query for pattern in detail_patterns)

def is_participant_query(user_query: str, context: dict = None) -> dict:
    """
    참석자 관련 질문 감지 (패턴 매칭)
    """
    
    # ========== 1. Task 패턴 먼저 체크 (제외) ==========
    task_patterns = [
        r'누가.*?(해|하|담당|맡)',
        r'다른\s*사람.*?(일|해|담당)',
        r'누구.*?(일|담당|맡)',
    ]
    
    for pattern in task_patterns:
        if re.search(pattern, user_query):
            return {'is_participant': False, 'query_type': None, 'person_name': None}
    
    # ========== 2. Participant 패턴 체크 ==========
    participant_patterns = [
        r'참석',
        r'참여',
        r'누구.*?(회의|미팅)',
        r'(회의|미팅).*?누구',
        r'누가.*?(있었|나왔|왔)',
        r'멤버',
        r'참석자',
        r'함께',
        r'같이',
        r'([가-힣]{2,4})랑.*?(회의|미팅)',
        r'([가-힣]{2,4}).*?회의.*?(했|함)',
    ]
    
    has_participant_pattern = False
    for pattern in participant_patterns:
        if re.search(pattern, user_query):
            has_participant_pattern = True
            break
    
    if not has_participant_pattern:
        return {'is_participant': False, 'query_type': None, 'person_name': None}
    
    # ========== 3. 이름 추출 시도 ==========
    name_patterns = [
        r'([가-힣]{2,4})[가이]?\s*참석',
        r'([가-힣]{2,4})[가이]?\s*회의',
        r'([가-힣]{2,4})랑',
        r'참석.*?([가-힣]{2,4})[가이]?',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, user_query)
        if match:
            person_name = match.group(1)
            person_name = re.sub(r'[가이은는을를]$', '', person_name)
            
            if person_name not in ['사람', '누가', '누구', '회의', '미팅', '멤버', '거기', '여기']:
                return {
                    'is_participant': True,
                    'query_type': 'person_meetings',
                    'person_name': person_name
                }
    
    # ========== 4. 이름 없으면 회의 참석자 조회 ==========
    if context and context.get('selected_meeting_id'):
        return {
            'is_participant': True,
            'query_type': 'meeting_participants',
            'person_name': None
        }
    
    return {'is_participant': False, 'query_type': None, 'person_name': None}

def detect_pronoun_meeting_reference(user_query: str) -> bool:
    """
    대명사 + 회의 참조 감지 (오타 허용)
    """
    # 1. 정규식 패턴 (띄어쓰기 허용)
    pronoun_patterns = [
        r'저\s*회의',
        r'그\s*회의', 
        r'이\s*회의',
        r'해당\s*회의',
    ]
    
    for pattern in pronoun_patterns:
        if re.search(pattern, user_query):
            return True
    
    # 2. 단독 지시어
    standalone_refs = ['거기', '여기']
    if any(ref in user_query for ref in standalone_refs):
        return True
    
    # 3. 특수문자 제거 후 토큰 매칭 (오타 허용)
    cleaned = re.sub(r'[^\w\s]', '', user_query)
    tokens = cleaned.split()
    
    pronoun_tokens = {'저', '그', '이', '해당'}
    
    # 유사도 기반 "회의" 감지 (오타 허용)
    import difflib
    for i in range(len(tokens)):
        # 대명사 체크
        if tokens[i] in pronoun_tokens:
            # 다음 토큰이 "회의"와 비슷한지 체크
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                # 1. 조사 제거
                next_token_no_josa = re.sub(r'에서|에게|한테|부터|까지', '', next_token)
                # 2. 한글만 추출
                next_token_clean = re.sub(r'[^가-힣]', '', next_token_no_josa)
                
                # "회의", "미팅"과의 유사도 계산
                similarity_meeting = difflib.SequenceMatcher(None, next_token_clean, '회의').ratio()
                similarity_miting = difflib.SequenceMatcher(None, next_token_clean, '미팅').ratio()
                
                if similarity_meeting >= 0.5 or similarity_miting >= 0.5:
                    print(f"[DEBUG] 대명사 + 회의 유사 단어 감지: '{tokens[i]} {next_token}' (정제: '{next_token_clean}', 유사도: {max(similarity_meeting, similarity_miting):.1%})")
                    return True
    
    return False

def needs_llm_analysis(user_query: str, context: dict) -> bool:
    """
    LLM 분석이 필요한지 확인 (최소화)
    """
    # 1. 명확한 오타가 있으면 LLM 필요
    if any(char in user_query for char in ['ㅅ', 'ㅈ', 'ㄱ', 'ㄴ', 'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ']):
        return True
    
    # 2. 컨텍스트 있고 대명사만 쓴 짧은 질문 (5자 이하)
    if context and context.get('state') == 'meeting_selected' and len(user_query) <= 5:
        pronouns = ['그거', '저거', '이거', '거기', '여기']
        if any(p in user_query for p in pronouns):
            return True
    
    # 3. 그 외는 LLM 안 씀
    return False

# ============================================================
# 엔드포인트
# ============================================================

@app.get("/")
def root():
    """헬스 체크"""
    return {"status": "ok", "message": "회의록 검색 챗봇 서버가 실행 중입니다."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        user_query = request.message.strip()
        user_name = request.user_name
        
        # "나", "내", "저" → 로그인 사용자 이름으로 자동 치환
        if any(keyword in user_query for keyword in ['할일', '할 일', 'task', 'todo', '업무']):
            original_user_query = user_query
            user_query = user_query.replace('나의', user_name).replace('내', user_name).replace('나', user_name).replace('저', user_name)
            if original_user_query != user_query:
                print(f"[자동 치환] '{original_user_query}' → '{user_query}'")

        # user_job/user_position 처리: NONE이 아니면 해당 값 사용, NONE이면 DB에서 조회
        user_job = request.user_job if request.user_job and request.user_job != 'NONE' else None
        user_position = request.user_position if request.user_position and request.user_position != 'NONE' else None
        
        # user_id 먼저 조회 (항상 필요함!)
        user_id = get_user_id_by_name(user_name)

        # DB에서 사용자 정보 조회 (직무/직급이 NONE일 때만)
        if not user_job or not user_position:
            from .database import get_db_connection
            
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT job, position FROM user WHERE id = %s", (user_id,))
                    user_data = cursor.fetchone()
                    cursor.close()
                    
                    if user_data:
                        if not user_job:
                            user_job = user_data.get('job', 'NONE')
                        if not user_position:
                            user_position = user_data.get('position', 'NONE')
        
        # 기본값 설정
        if not user_job:
            user_job = 'NONE'
        if not user_position:
            user_position = 'NONE'

        # ========== Job 값 정규화 (대문자 변환) ==========
        user_job_normalized = user_job.upper()

        # 유효한 직무만 허용
        valid_jobs = ['NONE', 'PROJECT_MANAGER', 'FRONTEND_DEVELOPER', 
                    'BACKEND_DEVELOPER', 'DATABASE_ADMINISTRATOR', 'SECURITY_DEVELOPER']
        if user_job_normalized not in valid_jobs:
            user_job_normalized = 'NONE'
            
        print(f"\n{'='*70}")
        print(f"💬 사용자 질문: {user_query}")
        print(f"👤 User Name: {user_name}")
        print(f"👤 User Job (원본): {user_job}")
        print(f"👤 User Job (정규화): {user_job_normalized}")
        print(f"👤 User Position: {user_position}")
        print(f"🔑 Session ID: {session_id}")
        print(f"{'='*70}\n")

        # ========== 변수 초기화 ==========
        original_query = user_query
        context = get_context(session_id)
        intent = None
        llm_analysis = None

        # ========== 0차: Task 질문 최우선 체크 ==========
        # 이름 재사용 조건 먼저 체크
        name_reuse_condition = False
        if context and context.get('state') == 'meeting_selected' and context.get('last_person_name'):
            pronoun_detected = detect_pronoun_meeting_reference(user_query)
            print(f"[DEBUG] 이름 재사용 조건 체크: state={context.get('state')}, name={context.get('last_person_name')}, pronoun={pronoun_detected}")
            if pronoun_detected:
                name_reuse_condition = True
        
        # ========== 0차: Task 질문 최우선 체크 ==========
        is_task_query_preliminary = (
            ('일' in user_query and any(kw in user_query for kw in ['맡은', '담당', '완료', '끝난', '남은', '해야'])) or
            any(pattern in user_query.lower() for pattern in ['task', '액션', '할일', '할 일']) or
            any(pattern in user_query for pattern in ['해야될', '해야할', '해야되는', '해야하는', '해야 될', '해야 되는', '남은', '미완료', 'todo']) or
            (context and context.get('selected_meeting_id') and
            any(ref in user_query for ref in ['저 회의', '그 회의', '회의안에서', '회의에서', '거기']) and
            any(task_word in user_query for task_word in ['일', '할일', '담당', '맡은', 'task'])) or
            (context and context.get('state') == 'meeting_selected' and 
            user_query.strip() in ['나는?', '나는', '내꺼는?', '내꺼는', '내가?', '내가']) or
            (context and context.get('state') == 'meeting_selected' and
            any(word in user_query for word in ['전원', '모두', '전부']) and
            '회의' not in user_query and
            len(user_query) < 15) or
            name_reuse_condition
        )
                
        if is_task_query_preliminary:
            print(f"[DEBUG] Task 질문 우선 감지 → LLM 건너뛰기")
            # Task 질문은 LLM 없이 바로 처리

        # ========== 확인 질문 처리 ==========
        elif context and context.get('state') == 'meeting_list_shown':
            # 1. 확인 질문
            confirmation_patterns = ['끝', '전부', '다야', '이게 다', '그게 다', '이게끝', '그게끝']
            if any(pattern in user_query for pattern in confirmation_patterns):
                meeting_count = len(context.get('meeting_list', []))
                
                return ChatResponse(
                    answer=f"네, 맞아요! 총 {meeting_count}개의 회의예요. 😊\n\n더 자세히 알고 싶은 회의가 있으면 번호나 제목을 알려주세요!",
                    source="confirmation",
                    session_id=session_id,
                    history=request.history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": f"네, 맞아요! 총 {meeting_count}개의 회의예요. 😊"}
                    ]
                )
            
            # 2. 페이지네이션 요청 (나머지/더)
            more_keywords = ['나머지', '나머지도', '남은', '남은거', '더', '더보기', '더보여', 
                            '더있어', '더줘', '더알려', '추가', '추가로', '계속', '이어서', 
                            '다음', '또', '그외', '줘봐', '줘', '보여줘']
            
            if any(keyword in user_query for keyword in more_keywords):
                print(f"[DEBUG] meeting_list_shown 상태에서 페이지네이션 요청")
                # 상태를 awaiting_selection으로 변경하고 아래 로직으로 넘김
                context['state'] = 'awaiting_selection'
                save_context(session_id, context)
                # 아래 awaiting_selection 처리로 넘어감
        
        # ========== 번호 선택 우선 체크 ==========
        elif (context and context.get('state') == 'awaiting_selection' and 
            user_query.strip().isdigit()):
            print(f"[DEBUG] 번호 선택 감지 → 선택 처리로 이동")
            # 선택 처리로 넘어감

        # ========== RAG 상세 질문 우선 체크 ==========
        elif (context and context.get('state') == 'meeting_selected' and 
            len(user_query) < 30 and
            not user_query.strip().isdigit()):
            
            # COUNT 질문이 아니고, 검색 키워드도 없으면 RAG 가능성
            if (not is_count_question(user_query) and  # ← 함수 호출로 변경
                not any(word in user_query for word in ['회의', '검색', '할일', '참석', '키워드'])):
                print(f"[DEBUG] RAG 상세 질문 가능성 → LLM으로 확인")
        
        # ========== 1차: 명확한 패턴 빠른 처리 ==========
        elif is_obvious_pattern(user_query):
            print(f"[DEBUG] 명확한 패턴 감지 → LLM 호출 스킵")
            intent = 'meeting_search'

        # ========== 2차: LLM 전처리 (오타 보정 + 의도 파악) ==========
        elif needs_llm_analysis(user_query, context):
            print(f"[DEBUG] LLM 전처리 필요 → HyperCLOVA X 호출")
            from .llm import preprocess_query_with_llm
            
            llm_analysis = preprocess_query_with_llm(user_query, context)
            
            corrected_query = llm_analysis.get('corrected_query', user_query)
            intent = llm_analysis.get('intent', 'meeting_search')
            is_contextual = llm_analysis.get('is_contextual', False)
            
            print(f"[LLM 분석] 원본: {user_query}")
            print(f"[LLM 분석] 보정: {corrected_query}")
            print(f"[LLM 분석] 의도: {intent}")
            print(f"[LLM 분석] 컨텍스트 사용: {is_contextual}")
            
            # 보정된 쿼리로 교체
            user_query = corrected_query
            
            # Phase 1: 새로운 검색 의도가 명확하면, 컨텍스트 삭제 후 검색으로 유도 (추가된 로직)
            is_selection_state = context and context.get('state') == 'awaiting_selection'
            # Note: llm_analysis 변수는 LLM 분석이 성공했을 때만 존재합니다.
            is_new_search_intent = llm_analysis.get('intent') == 'meeting_search'
            
            # 명확한 선택 패턴인지 확인 (숫자/날짜만 허용)
            is_obvious_selection = user_query.strip().isdigit() or bool(re.match(r'^\d{1,2}월\s?\d{1,2}일$', user_query.strip()))
            
            if is_selection_state and is_new_search_intent and not is_obvious_selection:
                print(f"[DEBUG] 새로운 검색 의도 감지 (LLM Intent: {intent}) → 컨텍스트 무시")
                delete_context(session_id)
                # 컨텍스트를 삭제했으므로, 아래 Task/Participant 처리는 건너뛰고 
                # 최종 MySQL 검색으로 바로 진입하도록 pass 처리합니다.
                pass 
            
            # ========== Intent별 자동 처리 (Task/Participant) ==========
            scope_expansion = llm_analysis.get('scope_expansion', False) if llm_analysis else False

            # 1. Task 검색 intent
            if intent == 'task_search':
                # 컨텍스트 활용 여부 결정
                if is_contextual and context and context.get('state') == 'meeting_selected' and not scope_expansion:
                    # 특정 회의의 할일 검색
                    selected_meeting_id = context.get('selected_meeting_id')
                    meeting_title = context.get('meeting_title', '')
                    
                    print(f"[DEBUG] Task 검색 - 컨텍스트 활용: meeting_id={selected_meeting_id}")
                    
                    from .search import search_tasks
                    task_response, tasks = search_tasks(
                        user_query=user_query,
                        user_id=user_id,
                        meeting_id=selected_meeting_id
                    )
                    
                    return ChatResponse(
                        answer=task_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": task_response}
                        ],
                        source="task_search_contextual",
                        session_id=session_id
                    )
                else:
                    # 전체 회의의 할일 검색 (scope_expansion=True 또는 컨텍스트 없음)
                    print(f"[DEBUG] Task 검색 - 전체 검색 (scope_expansion={scope_expansion})")
                    
                    from .search import search_tasks
                    task_response, tasks = search_tasks(
                        user_query=user_query,
                        user_id=user_id,
                        meeting_id=None
                    )
                    
                    return ChatResponse(
                        answer=task_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": task_response}
                        ],
                        source="task_search_global",
                        session_id=session_id
                    )
            
            # 2. Participant 검색 intent
            elif intent == 'participant_search':
                # 특정 회의 참석자 vs 특정 사람이 참석한 회의
                
                # 이름 패턴 확인
                name_match = re.search(r'([가-힣]{2,4})', user_query)
                
                # 1순위: "누가 참석" 패턴 체크
                if '누가' in user_query:
                    # "마케팅 회의에 누가 참석했어?" → 회의 검색 후 참석자 조회
                    meeting_keyword_match = re.search(r'(.+?)\s*회의', user_query)
                    if meeting_keyword_match:
                        meeting_keyword = meeting_keyword_match.group(1).strip()
                        print(f"[DEBUG] Participant 검색 - 회의명으로 검색: {meeting_keyword}")
                        
                        # 회의 검색
                        from .search import search_meetings_direct
                        search_response, meetings = search_meetings_direct(
                            user_query=meeting_keyword,
                            date_info=None,
                            status=None,
                            user_job=user_job_normalized,
                            selected_meeting_id=None,
                            user_id=user_id
                        )
                        
                        if meetings and len(meetings) >= 1:
                            meeting_id = meetings[0]['id']
                            from .search import search_participants
                            participant_response, results = search_participants(
                                query_type="meeting_participants",
                                meeting_id=meeting_id
                            )
                            return ChatResponse(
                                answer=participant_response,
                                history=request.history + [
                                    {"role": "user", "content": original_query},
                                    {"role": "assistant", "content": participant_response}
                                ],
                                source="participant_meeting_members",
                                session_id=session_id
                            )
                        else:
                            # 회의 없음 → 상태 완화 시도
                            search_response_retry, meetings_retry = search_meetings_direct(
                                user_query=meeting_keyword,
                                date_info=None,
                                status=None,
                                user_job=user_job_normalized,
                                selected_meeting_id=None,
                                user_id=user_id
                            )
                            
                            if meetings_retry and len(meetings_retry) >= 1:
                                meeting_id = meetings_retry[0]['id']
                                from .search import search_participants
                                participant_response, results = search_participants(
                                    query_type="meeting_participants",
                                    meeting_id=meeting_id
                                )
                                return ChatResponse(
                                    answer=participant_response,
                                    source="participant_meeting_members",
                                    session_id=session_id
                                )
                            else:
                                return ChatResponse(
                                    answer=f"❌ '{meeting_keyword}' 관련 회의를 찾을 수 없어요.",
                                    source="participant_no_meeting",
                                    session_id=session_id
                                )
                            
                # 2순위: 특정 사람 검색
                if name_match and any(w in user_query for w in ['참석한', '나온', '있었']) and '누가' not in user_query:
                    # 특정 사람이 참석한 회의 검색
                    person_name = name_match.group(1)
                    # 조사 제거 (가, 이, 은, 는, 을, 를)
                    person_name = re.sub(r"[가이은는을를]$", "", person_name)
                    print(f"[DEBUG] Participant 검색 - 특정 사람: {person_name}")
                    
                    from .search import search_participants
                    participant_response, results = search_participants(
                        query_type="person_meetings",
                        person_name=person_name
                    )
                    
                    # 단일 회의면 컨텍스트 저장
                    if results and len(results) == 1:
                        meeting = results[0]
                        context = {
                            'state': 'meeting_selected',
                            'selected_meeting_id': meeting['id'],
                            'meeting_title': meeting['title'],
                            'original_query': user_query
                        }
                        save_context(session_id, context)
                        print(f"[DEBUG] 단일 회의 컨텍스트 저장: meeting_id={meeting['id']}")
                    elif results and len(results) > 1:
                        # 여러 회의 - 선택 대기 상태
                        meetings_serializable = []
                        for meeting in results:
                            meeting_copy = {}
                            for key, value in meeting.items():
                                if isinstance(value, datetime):
                                    meeting_copy[key] = value.isoformat()
                                else:
                                    meeting_copy[key] = value
                            meetings_serializable.append(meeting_copy)
                        
                        shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])

                        context = {
                            'state': 'awaiting_selection',
                            'shown_completed': shown_completed,
                            'shown_scheduled': shown_scheduled,
                            'meetings': meetings_serializable[:10],
                            'offset': min(5, len(meetings_serializable[:10])),
                            'total_count': len(results),
                            'original_query': user_query
                        }
                        save_context(session_id, context)
                        print(f"[DEBUG] 여러 회의 컨텍스트 저장: {len(results)}개")
                    
                    return ChatResponse(
                        answer=participant_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": participant_response}
                        ],
                        source="participant_person_meetings",
                        session_id=session_id
                    )
                
                elif is_contextual and context and context.get('state') == 'meeting_selected':
                    # 특정 회의의 참석자 조회
                    selected_meeting_id = context.get('selected_meeting_id')
                    print(f"[DEBUG] Participant 검색 - 특정 회의: meeting_id={selected_meeting_id}")
                    
                    from .search import search_participants
                    participant_response, results = search_participants(
                        query_type="meeting_participants",
                        meeting_id=selected_meeting_id
                    )
                    
                    return ChatResponse(
                        answer=participant_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": participant_response}
                        ],
                        source="participant_meeting_members",
                        session_id=session_id
                    )
                
                else:
                    # 참석자 정보 부족
                    fallback_msg = "누구의 참석 정보를 알려드릴까요? 😊\n예: '김철수가 참석한 회의', '채용 전략 회의 참석자'"
                    
                    return ChatResponse(
                        answer=fallback_msg,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": fallback_msg}
                        ],
                        source="participant_clarification",
                        session_id=session_id
                    )
        
            # ========== LLM 보정 후 다시 Task 체크 ==========
            if intent == 'task_search' or (context and context.get('selected_meeting_id') and detect_pronoun_meeting_reference(user_query)):
                print(f"[DEBUG] LLM 보정 후 Task 질문 재감지")
                # Task 질문이면 아래 is_task_query로 진행
        
        # ========== Task 질문 체크 ==========
        is_task_query = (
            ('일' in user_query and any(kw in user_query for kw in ['맡은', '담당', '완료', '끝난', '남은', '해야'])) or
            any(pattern in user_query.lower() for pattern in ['task', '액션', '할일', '할 일']) or
            any(pattern in user_query for pattern in ['해야될', '해야할', '해야되는', '해야하는', '해야 될', '해야 되는', '남은', '미완료', 'todo']) or    
            (context and context.get('state') == 'meeting_selected' and 
            ('사람' in user_query or '담당' in user_query or '누가' in user_query or '아무도' in user_query)) or
            ('전체' in user_query or '모두' in user_query or '전부' in user_query) or
            (context and context.get('state') == 'meeting_selected' and 
            detect_pronoun_meeting_reference(user_query) and
            any(task_word in user_query for task_word in ['일', '할일', '담당', '맡은', '완료', 'task'])) or
            (context and context.get('state') == 'meeting_selected' and 
            user_query.strip() in ['나는?', '나는', '내꺼는?', '내꺼는', '내가?', '내가']) or
            (context and context.get('state') == 'meeting_selected' and 
            context.get('last_person_name') and 
            detect_pronoun_meeting_reference(user_query))
        )

        # # 컨텍스트에 회의가 있으면 해당 회의의 Task만 조회
        # context = get_context(session_id)
        # if context and context.get('selected_meeting_id'):
        #     meeting_id = context['selected_meeting_id']
        #     print(f"[컨텍스트] 회의 ID={meeting_id}의 Task 조회")
            
        #     from .search import search_tasks
        #     message, tasks = search_tasks(
        #         user_query=user_query,
        #         user_id=user_id,
        #         meeting_id=meeting_id,
        #         user_name=user_name
        #     )
            
        #     print(f"[DEBUG] Task 검색 완료, meeting_id={meeting_id}")
            
        #     return ChatResponse(
        #         answer=message,
        #         source="task_query",
        #         session_id=session_id
        #     )
        
        if is_task_query:
            print(f"[DEBUG] Task 질문 감지")
            
            from .search import search_tasks
            
            # 컨텍스트에서 meeting_id 가져오기 (전체 키워드 없을 때만)
            meeting_id_from_context = None
            if context and context.get('selected_meeting_id'):
                if not any(keyword in user_query for keyword in ['전체', '모든', '전부', '다른 회의']):
                    meeting_id_from_context = context['selected_meeting_id']
                    print(f"[컨텍스트] 회의 ID={meeting_id_from_context}의 Task 조회")
            
            # "X 회의에서 할일" 패턴 감지
            has_meeting_context_in_query = (
                (('회의에서' in user_query or '미팅에서' in user_query or '회의안에서' in user_query) and ('할일' in user_query or '할 일' in user_query)) or
                (context and context.get('selected_meeting_id') and any(re.search(pattern, user_query) for pattern in [r'그\s*중', r'저\s*중', r'이\s*중']))
            )
            
            meeting_id = None
            
            if has_meeting_context_in_query:
                print(f"[DEBUG] 'X 회의에서 할일' 패턴 감지 → 회의 검색 먼저")
                
                # 대명사 체크
                pronouns = ['저 회의', '그 회의', '이 회의', '해당 회의', '거기', '저회의', '그회의', '이회의']
                has_pronoun = detect_pronoun_meeting_reference(user_query)
                
                if has_pronoun and context and context.get('selected_meeting_id'):
                    # 대명사면 무조건 컨텍스트 사용
                    meeting_id = context['selected_meeting_id']
                    print(f"[DEBUG] 대명사 감지 → 컨텍스트 회의 사용 (ID: {meeting_id})")
                else:
                    # 회의명으로 검색
                    meeting_pattern = r'([가-힣a-zA-Z0-9\s]+)(회의|미팅)에서'
                    match = re.search(meeting_pattern, user_query)
                    
                    if match:
                        meeting_query = match.group(1).strip() + match.group(2)
                        print(f"[DEBUG] 추출된 회의명: {meeting_query}")
                        
                        from .search import search_meetings_direct
                        _, meetings = search_meetings_direct(
                            user_query=meeting_query,
                            date_info=None,
                            status=None,
                            user_job=None,
                            selected_meeting_id=None,
                            user_id=user_id
                        )
                        
                        if meetings and len(meetings) == 1:
                            meeting_id = meetings[0]['id']
                            print(f"[DEBUG] 단일 회의 발견: {meetings[0]['title']} (ID: {meeting_id})")
                        elif meetings and len(meetings) > 1:
                            meeting_id = meetings[0]['id']
                            print(f"[DEBUG] 여러 '{meeting_query}' 발견 ({len(meetings)}개) → 최신 회의 사용: {meetings[0]['title']} (ID: {meeting_id})")
                        else:
                            print(f"[DEBUG] '{meeting_query}' 회의를 찾을 수 없음")

            # 컨텍스트 우선 사용
            if not meeting_id:
                meeting_id = meeting_id_from_context
            
            if not meeting_id and context and context.get('selected_meeting_id'):
                meeting_id = context['selected_meeting_id']
                
                # "전체" 키워드만 체크 (타인 이름은 search_tasks에서 판단)
                if any(keyword in user_query for keyword in ['전체', '모든', '전부']):
                    meeting_id = None
                    print(f"[DEBUG] '{user_query}' - 전체 검색 키워드 감지, meeting_id 초기화")

            # 기존 함수 사용!
            user_id = get_user_id_by_name(user_name)
            print(f"[DEBUG] user_id 조회 성공: {user_id}")

            # 타인 이름 목록 DB에서 조회
            from .config import DB_CONFIG

            try:
                with get_db_connection() as conn:
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM user WHERE id != %s", (user_id,))
                        other_names = [row['name'] for row in cursor.fetchall()]
                        cursor.close()
                        print(f"[DEBUG] DB에서 타인 이름 조회: {other_names}")
                    else:
                        other_names = []
            except Exception as e:
                print(f"[DEBUG] 타인 이름 조회 실패: {e}")
                other_names = []
            
            # 1. 이름이 쿼리에 있으면 저장
            for name in other_names:
                if name in user_query:
                    if context:
                        context['last_person_name'] = name
                        save_context(session_id, context)
                        print(f"[DEBUG] 컨텍스트에 이름 저장: {name}")
                    break
            
            # 2. 이름이 없고 + meeting_id 있고 + 이전 이름 있으면 → 이름 재사용
            if meeting_id and context and context.get('last_person_name'):
                # 현재 쿼리에 이름이 없는지 체크
                has_name_in_query = any(name in user_query for name in other_names)
                if not has_name_in_query:
                    person_name = context.get('last_person_name')
                    print(f"[DEBUG] 이전 질문의 이름 재사용: {person_name}")
                    # user_query에 이름 추가
                    user_query = user_query + f" {person_name}"
                    print(f"[DEBUG] 쿼리 확장: {user_query}")

            # user_id를 어디서 가져올지 결정
            from .database import get_db_connection

            # chat 함수 내부에서
            user_id = get_user_id_by_name(user_name) if user_name else 1
            message, tasks = search_tasks(user_query, user_id=user_id, meeting_id=meeting_id, user_name=user_name)

            # Task 검색 시에도 컨텍스트 저장
            print(f"[DEBUG] Task 검색 완료, meeting_id={meeting_id}")
             
            if meeting_id:
                try:
                    print(f"[DEBUG] 컨텍스트 저장 시도: meeting_id={meeting_id}")
                    
                    from .database import get_db_connection
                    
                    with get_db_connection() as conn:
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT title FROM meeting WHERE id = %s", (meeting_id,))
                            meeting = cursor.fetchone()
                            cursor.close()
                            
                            if meeting:
                                # 기존 컨텍스트 가져와서 업데이트
                                existing_context = get_context(session_id) or {}
                                
                                # ← fetchone() 결과가 튜플이므로 인덱스로 접근
                                title = meeting[0] if isinstance(meeting, tuple) else meeting.get('title')
                                
                                existing_context.update({
                                    'state': 'meeting_selected',
                                    'selected_meeting_id': meeting_id,
                                    'meeting_title': title,
                                })
                                save_context(session_id, existing_context)
                                print(f"[DEBUG] ✅ 컨텍스트 저장 성공: meeting_id={meeting_id}, title={title}")
                            else:
                                print(f"[DEBUG] ❌ 회의 정보 조회 실패: meeting_id={meeting_id}")
                                
                except Exception as e:
                    print(f"[DEBUG] ❌ 컨텍스트 저장 중 예외 발생: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DEBUG] meeting_id 없음, 컨텍스트 저장 스킵")

            return ChatResponse(
                answer=message,
                history=request.history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": message}
                ],
                source="task_query",
                session_id=session_id
            )
        
        # ========== RAG 상세 질문 처리 ==========
        # 회의 선택 후 상세 질문 (예산, 참여자, 시간 등)
        # RAG 질문 판단: 키워드 1차 체크 → LLM 2차 확인
        detail_keywords = ['예산', '얼마', '금액', '비용', '언제', '시간', '몇 시', '어떻게', '어떤', '왜', '방법', '결과', '결론', '내용', '주요', '핵심', '이유', '목적']
        is_detail_question = any(keyword in user_query for keyword in detail_keywords)
        
        # LLM으로 한 번 더 확인 (키워드로 못 잡은 경우)
        if (context and context.get('state') == 'meeting_selected' and 
            not is_detail_question):
            from .llm import classify_query_intent
            meeting_title = context.get('meeting_title', '')
            llm_intent = classify_query_intent(user_query, meeting_title)
            if llm_intent == "RAG":
                is_detail_question = True
                print(f"[LLM 보강] RAG 질문으로 재분류")

        if (context and context.get('state') == 'meeting_selected' and 
            is_detail_question and
            intent not in ['task_search', 'participant_search', 'meeting_search']):
            
            print(f"[DEBUG] RAG 상세 질문 처리: '{user_query}'")

            selected_meeting_id = context.get('selected_meeting_id')
            meeting_title = context.get('meeting_title', '선택된 회의')
            
            from .database import get_db_connection
            
            with get_db_connection() as conn:
                if not conn:
                    return ChatResponse(
                        answer="데이터베이스 연결에 실패했어요. 😢",
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": "데이터베이스 연결에 실패했어요. 😢"}
                        ],
                        source="db_connection_error",
                        session_id=session_id
                    )
                
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT 
                            m.id, 
                            m.title, 
                            m.description, 
                            m.scheduled_at, 
                            mr.summary, 
                            m.status,
                            GROUP_CONCAT(
                                CONCAT(t.speaker_name, ': ', t.text) 
                                ORDER BY t.sequence_order 
                                SEPARATOR '\n'
                            ) as transcript_text
                        FROM meeting m
                        LEFT JOIN meeting_result mr ON m.id = mr.meeting_id
                        LEFT JOIN transcript t ON m.id = t.meeting_id
                        WHERE m.id = %s
                        GROUP BY m.id
                    """, (selected_meeting_id,))
                    meeting = cursor.fetchone()
                    cursor.close()
                    
                    if meeting:
                        from .llm import answer_meeting_question
                        rag_answer = answer_meeting_question(meeting, user_query)
                        
                        return ChatResponse(
                            answer=rag_answer,
                            history=request.history + [
                                {"role": "user", "content": original_query},
                                {"role": "assistant", "content": rag_answer}
                            ],
                            source="meeting_detail_rag",
                            session_id=session_id
                        )
                except Exception as e:
                    logger.error(f"RAG 처리 오류: {e}")
        
        # ========== 컨텍스트 기반 쿼리 확장 ==========
        selected_meeting_id = None
        selected_meeting_title = None

        # 개수 확인 질문 체크 (전역으로 먼저 정의)
        is_count_check = is_count_question(user_query)

        if context and context.get('state') == 'meeting_selected':
            selected_meeting_id = context.get('selected_meeting_id')
            selected_meeting_title = context.get('meeting_title', '')
            
            print(f"[컨텍스트] 이전 선택 회의: {selected_meeting_title} (ID: {selected_meeting_id})")
            
            # 짧은 질문이거나 대명사 사용하면 회의명 추가
            # 단, 새로운 검색 의도가 명확한 경우는 제외

            # 명확한 컨텍스트 참조 표현
            context_refs = ['그 회의', '저 회의', '이 회의', '해당 회의', '거기', '여기서', '그거', '그것', '그게', '저거', '저것', '이거', '이것']

            # 할일/Task 관련 표현 (컨텍스트 활용 대상)
            task_refs = ['내가', '나의', '할일', '할 일', '담당', '맡은', '누가', '다른 사람']
            # RAG 상세 질문 키워드 (확장하면 안 됨)
            detail_keywords = ['예산', '얼마', '금액', '비용', '언제', '시간', '몇 시', '어떻게', '어떤', '왜', '방법', '결과', '결론', '내용', '주요', '핵심', '이유', '목적', '참석자', '참여자', '발표자', '분위기', '반응', '의견', '어땠', '어떻']
            is_detail_question = any(keyword in user_query for keyword in detail_keywords)

            # 키워드로 못 잡으면 LLM으로 한 번 더 체크
            if not is_detail_question:
                from .llm import classify_query_intent
                intent = classify_query_intent(user_query, selected_meeting_title)
                if intent == "RAG":
                    is_detail_question = True
                    print(f"[LLM 보강] RAG 질문으로 재분류")

            if context and context.get('state') == 'meeting_selected':
                selected_meeting_id = context.get('selected_meeting_id')
                selected_meeting_title = context.get('meeting_title', '')
                
                print(f"[컨텍스트] 이전 선택 회의: {selected_meeting_title} (ID: {selected_meeting_id})")
                
                # 🔹 LLM으로 의도 분류
                from .llm import classify_query_intent
                intent = classify_query_intent(user_query, selected_meeting_title)
                
                if intent == "RAG":
                    # RAG 상세 질문 → 컨텍스트 유지, 확장 안 함
                    print(f"[LLM 의도] RAG 질문 → 컨텍스트 유지")
                    # selected_meeting_id 유지하여 1055번 줄 RAG 처리로 진행
                    
                elif intent == "NEW_SEARCH":
                    # 새로운 검색 → 컨텍스트 삭제
                    print(f"[LLM 의도] 새로운 검색 → 컨텍스트 삭제")
                    context = None
                    selected_meeting_id = None
                    selected_meeting_title = None
                    delete_context(session_id)
                    
                elif intent == "CONTEXT_DEPENDENT":
                    # 컨텍스트 확장 (할일, 담당자 등)
                    if not is_count_check:  # COUNT 질문은 확장 안 함
                        user_query = f"{selected_meeting_title} 회의에서 {user_query}"
                        print(f"[LLM 의도] 컨텍스트 확장: {user_query}")
                    else:
                        print(f"[LLM 의도] CONTEXT_DEPENDENT이지만 COUNT 질문 → 확장 안 함")

            # 명확한 컨텍스트 참조가 있으면 무조건 컨텍스트 활용
            has_context_ref = any(ref in user_query for ref in context_refs)

            # Task 질문이고 새로운 검색 키워드 없으면 컨텍스트 활용
            is_task_query = any(ref in user_query for ref in task_refs)
            new_search_words = ['뭐', '어떤', '있어', '있었어', '있나', '찾아', '검색']
            has_new_search = any(word in user_query for word in new_search_words)

            # 컨텍스트 활용 조건
            # 1. is_context_dependent_query가 True면 기본적으로 컨텍스트 활용
            # 2. 단, 명확한 새 검색 패턴만 제외
                        
            # 명확한 새 검색 패턴: "회의" + 검색동사 (단, RAG 질문 제외)
            explicit_new_search_patterns = [
                ('회의' in user_query and '뭐' in user_query and not is_detail_question),
                ('회의' in user_query and '어떤' in user_query and not is_detail_question),
                ('회의' in user_query and any(w in user_query for w in ['있어', '있었어', '있나']) and not is_detail_question),
                ('회의' in user_query and any(w in user_query for w in ['찾아', '검색']) and not is_detail_question),
                (user_query.count('회의') >= 2),
            ]

            # 전체 검색 명시 패턴 (컨텍스트 무시)
            global_search_patterns = [
                any(w in user_query for w in ['전체', '모든', '모두', '전부']),
                any(w in user_query for w in ['다른 회의', '다른회의', '다른 것']),
                ('다른' in user_query and any(w in user_query for w in ['회의', '일', '할일'])),
                ('더' in user_query and any(w in user_query for w in ['있어', '없어', '뭐'])),
            ]

            # 확인/검증 질문 패턴 (엄격하게)
            is_question = user_query.strip().endswith('?') or any(w in user_query[-3:] for w in ['야', '니', '나', '까', '지'])
            
            # 1순위: Task 질문 체크
            is_task = any(w in user_query for w in ['할일', '할 일', '담당', '맡은'])
            
            confirmation_patterns = [
                ('맞' in user_query and any(w in user_query for w in ['아', '지', '니', '나', '요'])),  # "맞아?", "맞지?"
                (any(w in user_query for w in ['그거', '저거', '이거']) and any(w in user_query for w in ['야', '니', '나'])),  # "그거야?"
                ('최근' in user_query and any(w in user_query for w in ['그거', '저거']) and is_question),  # "최근 그거야?"
            ]
            
            # 2순위: 확인 질문 체크 (Task 아닐 때만)
            is_confirmation = (
                not is_task and
                is_question and 
                any(confirmation_patterns) and
                not any(w in user_query for w in ['뭐', '무엇', '누가'])  # Task 질문 제외
            )

            explicit_new_search = any(explicit_new_search_patterns)
            wants_global_search = any(global_search_patterns)

            # 3순위: 일반 컨텍스트 활용
            should_use_context = (
                (is_context_dependent_query(user_query) or is_confirmation) and 
                not explicit_new_search and
                not wants_global_search and
                not is_count_check and  # ← COUNT 질문은 컨텍스트 확장 안 함
                not is_detail_question  # ← RAG 상세 질문도 확장 안 함
            )

            if should_use_context:
                user_query = f"{selected_meeting_title} 회의에서 {user_query}"
                print(f"[컨텍스트 확장] {original_query} → {user_query}")
            elif is_detail_question:
                print(f"[RAG 질문] 컨텍스트 확장 안 함: {user_query}")

            else:
                if wants_global_search:
                    print(f"[컨텍스트] 전체 검색 요청 → 컨텍스트 무시")
                elif is_count_check:
                    print(f"[컨텍스트] 개수 확인 질문 → 컨텍스트 유지 (확장 안 함)")
                    # COUNT 질문은 컨텍스트는 유지하되 확장하지 않음
                else:
                    print(f"[컨텍스트] 새로운 검색 → 컨텍스트 무시")
                    context = None
                    selected_meeting_id = None  # 컨텍스트 필터 해제
                    selected_meeting_title = None
                    delete_context(session_id)  # 컨텍스트 삭제

        # === 0-1단계: 통계 질문 체크 (Phase 3) ===
        count_keywords = ['몇 번', '몇번', '몇개', '몇 개', '총 몇', '총몇', '횟수']
        count_patterns = [
            r'하나\w*\?$',      # 하나야? 하나임? 하나니?
            r'\d+개\w*\?$',     # 2개야? 3개임?
            r'끝이\w*\?$',      # 끝이야? 끝임?
            r'전부\w*\?$',      # 전부야? 전부임?
            r'다\w*\?$',        # 다야? 다임?
            r'뿐이\w*\?$',      # 뿐이야? 뿐임?
        ]
        is_count_query = any(keyword in user_query for keyword in count_keywords) or \
                        any(re.search(pattern, user_query) for pattern in count_patterns)

        if is_count_query:
            print(f"\n📊 통계 질문 감지: '{user_query}'")
            
            # 날짜/상태 파싱
            date_info = parse_date_from_query(user_query)
            status = parse_status_from_query(user_query)
            
            # 컨텍스트에서 이전 검색 상태 가져오기
            if not status and context and context.get('search_status'):
                status = context.get('search_status')
                print(f"[DEBUG] 컨텍스트에서 상태 복원: {status}")
                
            # ========== "했어" 같은 과거형 어미가 있으면 완료된 회의로 처리 ==========
            if not status:
                past_tense_patterns = [
                    r'했어\??$', r'했니\??$', r'했나\??$', r'했냐\??$',
                    r'했습니까\??$', r'했는가\??$'
                ]
                for pattern in past_tense_patterns:
                    if re.search(pattern, user_query):
                        status = 'COMPLETED'
                        print(f"[DEBUG] 통계 질문 + 과거형 어미 → COMPLETED로 처리")
                        break
            
            # 일반적인 회의 목록 요청 패턴 체크
            general_list_patterns = ['회의 뭐', '회의 있', '회의 목록', '회의 보여', '회의 알려', '무슨 회의']
            is_general_request = any(p in user_query for p in general_list_patterns) and not date_info.get('type')
            
            if is_general_request:
                print("[DEBUG] 일반 회의 목록 요청으로 판단 → 키워드 검색 생략")
                keywords = []  # 키워드 없이 전체 검색
            else:
                keywords = extract_keywords_from_query(user_query)
            excluded_for_count = [
                '했어', '했니', '했나', '했냐', '있어', '있었어', '몇', '개', '번', '횟수',
                # 종결어미 추가
                '개야', '번이야', '거야', '거니', '이야', '예요', '이에요',
                '뭐야', '뭔가', '뭐지', '인가', '인지', '인데', '네요', '추가', '알려'
            ]
            
            keywords = [k for k in keywords if k not in excluded_for_count]
            
            # 키워드 없으면 컨텍스트에서 재사용
            if not keywords and context and context.get('original_query'):
                original_query = context.get('original_query')
                print(f"[DEBUG] COUNT - 키워드 없음, 컨텍스트에서 추출: '{original_query}'")
                keywords = extract_keywords_from_query(original_query)
                keywords = [k for k in keywords if k not in excluded_for_count]
                print(f"[DEBUG] COUNT - 컨텍스트 키워드: {keywords}")

            print(f"[DEBUG] 통계 쿼리 키워드: {keywords if keywords else '(없음)'}")
            
            # COUNT 쿼리 실행
            from .search import search_meeting_count
            count_result = search_meeting_count(
                keywords if keywords else None, 
                date_info, 
                status,
                user_job_normalized
            )
            
            if count_result:
                count = count_result['count']
                meetings = count_result['meetings']
                
                # ========== Phase 2-A: 페르소나 정렬 적용 ==========
                if ENABLE_PERSONA and meetings and len(meetings) > 1:
                    meetings = search_with_persona(meetings, user_job_normalized)
                    print(f"[DEBUG] Phase 2-A (통계 초기): {user_job_normalized} 관련도 순으로 정렬")
                
                # ========== 답변 생성 ==========
                # 상태별 표현
                if status == 'COMPLETED':
                    status_text = "완료된 회의"
                elif status == 'SCHEDULED':
                    status_text = "예정된 회의"
                elif status == 'RECORDING':
                    status_text = "진행중인 회의"
                else:
                    status_text = "회의"
                
                # 날짜 표현
                if date_info and date_info.get('original'):
                    date_text = f"{date_info['original']} "
                else:
                    date_text = ""
                
                # 키워드 표현
                if keywords:
                    keyword_text = f"'{', '.join(keywords)}' 관련 "
                else:
                    keyword_text = ""
                
                # 시제에 맞는 동사 선택
                if status == 'COMPLETED':
                    verb = "있었어요"  # 과거
                elif status == 'SCHEDULED':
                    verb = "있어요"    # 미래
                elif status == 'RECORDING':
                    verb = "있어요"    # 현재
                else:
                    verb = "있었어요"  # 전체 (과거형)

                answer = f"{date_text}{keyword_text}{status_text}는 총 {count}번 {verb}! 📊\n\n"

                if meetings and len(meetings) > 0:
                    answer += "📅 날짜별로 보면:\n\n"
                    for i, meeting in enumerate(meetings[:5], 1):
                        scheduled_at = meeting.get('scheduled_at')
                        if isinstance(scheduled_at, str):
                            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                        
                        # 날짜 포맷 변경: (2025년 01월 20일) 형식
                        date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                        title = meeting.get('title', '제목 없음')
                        answer += f"{i}. {title} {date_str}\n"
                    
                    if count > 5:
                        answer += f"\n💡 이 외에도 {count - 5}개가 더 있어요!"
                        
                # ========== 컨텍스트 저장 (후속 질문 대비) ==========
                if meetings and count > 0:
                    # datetime → str 변환 (전체 회의 저장!)
                    meetings_serializable = []
                    for meeting in meetings:
                        meeting_copy = {}
                        for key, value in meeting.items():
                            if isinstance(value, datetime):
                                meeting_copy[key] = value.isoformat()
                            else:
                                meeting_copy[key] = value
                        meetings_serializable.append(meeting_copy)
                    
                    context_data = {
                        'state': 'count_result',
                        'meetings': meetings_serializable,
                        'total_count': count,
                        'original_query': user_query
                    }
                    save_context(session_id, context_data)
                    print(f"[DEBUG] 통계 결과 컨텍스트 저장: {count}개 회의")
                                    
                    answer += "\n\n💬 \"그 회의들 보여줘\" 라고 물어보시면 자세히 알려드릴게요!"
                
                return ChatResponse(
                    answer=answer,
                    history=request.history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": answer}
                    ],
                    source="count_query",
                    session_id=session_id
                )
            else:
                error_msg = "회의 개수를 조회할 수 없어요. 😢"
                return ChatResponse(
                    answer=error_msg,
                    history=request.history,
                    source="count_error",
                    session_id=session_id
                )

        # ========== LLM 전처리 (오타 보정 + 의도 파악) ==========
        # 컨텍스트 의존적이거나 짧은 질문일 때 LLM 분석 실행
        preprocessed = None  # 무조건 초기화

        # 명확한 패턴이 아닐 때만 LLM 분석
        if not is_obvious_pattern(user_query) and needs_llm_analysis(user_query, context):
            from .llm import preprocess_query_with_llm
            
            preprocessed = preprocess_query_with_llm(user_query, context)
            llm_analysis = preprocessed
            print(f"[DEBUG] LLM 전처리 결과: {preprocessed}")
            
            corrected_query = preprocessed.get('corrected_query', user_query)
            intent = preprocessed.get('intent', 'meeting_search')
            is_contextual = preprocessed.get('is_contextual', False)
            scope_expansion = preprocessed.get('scope_expansion', False)
            
            # 보정된 쿼리로 교체
            user_query = corrected_query
            
            # Phase 1: 새로운 검색 의도가 명확하면, 컨텍스트 삭제 후 검색으로 유도 (추가된 로직)
            is_selection_state = context and context.get('state') == 'awaiting_selection'
            is_new_search_intent = preprocessed.get('intent') == 'meeting_search' if preprocessed else False

            # 명확한 선택 패턴인지 확인 (숫자/날짜만 허용)
            is_obvious_selection = user_query.strip().isdigit() or bool(re.match(r'^\d{1,2}월\s?\d{1,2}일$', user_query.strip()))
            
            if is_selection_state and is_new_search_intent and not is_obvious_selection:
                print(f"[DEBUG] 새로운 검색 의도 감지 (LLM Intent: {intent}) → 컨텍스트 무시")
                delete_context(session_id)
                # 컨텍스트를 삭제했으므로, 아래 Task/Participant 처리를 건너뛰고 
                # 최종 MySQL 검색으로 바로 진입하도록 pass 처리합니다.
                pass 
            
            # ========== Intent별 자동 처리 (Task/Participant) ==========
            scope_expansion = llm_analysis.get('scope_expansion', False)

            # 1. Task 검색 intent
            if intent == 'task_search':
                # 컨텍스트 활용 여부 결정
                if is_contextual and context and context.get('state') == 'meeting_selected' and not scope_expansion:
                    # 특정 회의의 할일 검색
                    selected_meeting_id = context.get('selected_meeting_id')
                    meeting_title = context.get('meeting_title', '')
                    
                    print(f"[DEBUG] Task 검색 - 컨텍스트 활용: meeting_id={selected_meeting_id}")
                    
                    from .search import search_tasks
                    task_response, tasks = search_tasks(
                        user_query=user_query,
                        user_id=user_id,
                        meeting_id=selected_meeting_id
                    )
                    
                    return ChatResponse(
                        answer=task_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": task_response}
                        ],
                        source="task_search_contextual",
                        session_id=session_id
                    )
                else:
                    # 전체 회의의 할일 검색 (scope_expansion=True 또는 컨텍스트 없음)
                    print(f"[DEBUG] Task 검색 - 전체 검색 (scope_expansion={scope_expansion})")
                    
                    from .search import search_tasks
                    task_response, tasks = search_tasks(
                        user_query=user_query,
                        user_id=user_id,
                        meeting_id=None
                    )
                    
                    return ChatResponse(
                        answer=task_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": task_response}
                        ],
                        source="task_search_global",
                        session_id=session_id
                    )
            
            # 2. Participant 검색 intent
            elif intent == 'participant_search':
                # 특정 회의 참석자 vs 특정 사람이 참석한 회의
                
                # 1순위: "누가 참석" 패턴 체크 (특정 회의의 참석자 조회)
                if '누가' in user_query:
                    # "마케팅 회의에 누가 참석했어?" → 회의 검색 후 참석자 조회
                    # 회의명 추출 (간단히 "회의" 앞의 단어들)
                    meeting_keyword_match = re.search(r'(.+?)\s*회의', user_query)
                    if meeting_keyword_match:
                        meeting_keyword = meeting_keyword_match.group(1).strip()
                        print(f"[DEBUG] Participant 검색 - 회의명으로 검색: {meeting_keyword}")
                        
                        # 회의 검색
                        from .search import search_meetings_direct
                        search_response, meetings = search_meetings_direct(
                            user_query=meeting_keyword,
                            date_info=None,
                            status=None,
                            user_job=user_job_normalized,
                            selected_meeting_id=None,
                            user_id=user_id
                        )
                        
                        if meetings and len(meetings) == 1:
                            # 단일 회의 발견 → 참석자 조회
                            meeting_id = meetings[0]['id']
                            print(f"[DEBUG] Participant 검색 - 특정 회의: meeting_id={meeting_id}")
                            
                            from .search import search_participants
                            participant_response, results = search_participants(
                                query_type="meeting_participants",
                                meeting_id=meeting_id
                            )
                            
                            return ChatResponse(
                                answer=participant_response,
                                history=request.history + [
                                    {"role": "user", "content": original_query},
                                    {"role": "assistant", "content": participant_response}
                                ],
                                source="participant_meeting_members",
                                session_id=session_id
                            )
                        elif meetings and len(meetings) > 1:
                            # 여러 회의 발견 → 선택 요청
                            return ChatResponse(
                                answer=f"{meeting_keyword} 관련 회의가 여러 개 있어요. 어떤 회의의 참석자를 확인하시겠어요?\n\n{search_response}",
                                history=request.history + [
                                    {"role": "user", "content": original_query},
                                    {"role": "assistant", "content": search_response}
                                ],
                                source="participant_multiple_meetings",
                                session_id=session_id
                            )
                    # 컨텍스트 활용
                    elif is_contextual and context and context.get('state') == 'meeting_selected':
                        selected_meeting_id = context.get('selected_meeting_id')
                        print(f"[DEBUG] Participant 검색 - 특정 회의 (컨텍스트): meeting_id={selected_meeting_id}")
                        
                        from .search import search_participants
                        participant_response, results = search_participants(
                            query_type="meeting_participants",
                            meeting_id=selected_meeting_id
                        )
                        
                        return ChatResponse(
                            answer=participant_response,
                            history=request.history + [
                                {"role": "user", "content": original_query},
                                {"role": "assistant", "content": participant_response}
                            ],
                            source="participant_meeting_members",
                            session_id=session_id
                        )
                
                # 2순위: 특정 사람이 참석한 회의 검색
                name_match = re.search(r'([가-힣]{2,4})', user_query)
                
                if name_match and any(w in user_query for w in ['참석한', '나온', '있었']):
                    # "김철수가 참석한 회의?" → 특정 사람 검색
                    person_name = name_match.group(1)
                    # 조사 제거 (가, 이, 은, 는, 을, 를)
                    person_name = re.sub(r"[가이은는을를]$", "", person_name)
                    print(f"[DEBUG] Participant 검색 - 특정 사람: {person_name}")
                    
                    from .search import search_participants
                    participant_response, results = search_participants(
                        query_type="person_meetings",
                        person_name=person_name
                    )
                    
                    # 단일 회의면 컨텍스트 저장
                    if results and len(results) == 1:
                        meeting = results[0]
                        context = {
                            'state': 'meeting_selected',
                            'selected_meeting_id': meeting['id'],
                            'meeting_title': meeting['title'],
                            'original_query': user_query
                        }
                        save_context(session_id, context)
                        print(f"[DEBUG] 단일 회의 컨텍스트 저장: meeting_id={meeting['id']}")
                    
                    elif results and len(results) > 1:
                        # 여러 회의 - 선택 대기 상태
                        meetings_serializable = []
                        for meeting in results:
                            meeting_copy = {}
                            for key, value in meeting.items():
                                if isinstance(value, datetime):
                                    meeting_copy[key] = value.isoformat()
                                else:
                                    meeting_copy[key] = value
                            meetings_serializable.append(meeting_copy)
                            
                        shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])

                        context = {
                            'state': 'awaiting_selection',
                            'shown_completed': shown_completed,
                            'shown_scheduled': shown_scheduled,
                            'meetings': meetings_serializable[:10],
                            'offset': min(5, len(meetings_serializable[:10])),
                            'total_count': len(results),
                            'original_query': user_query
                        }
                        save_context(session_id, context)
                        print(f"[DEBUG] 여러 회의 컨텍스트 저장: {len(results)}개")
                    
                    return ChatResponse(
                        answer=participant_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": participant_response}
                        ],
                        source="participant_person_meetings",
                        session_id=session_id
                    )
                                
                else:
                    # 참석자 정보 부족
                    fallback_msg = "누구의 참석 정보를 알려드릴까요? 😊\n예: '김철수가 참석한 회의', '채용 전략 회의 참석자'"
                    
                    return ChatResponse(
                        answer=fallback_msg,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": fallback_msg}
                        ],
                        source="participant_clarification",
                        session_id=session_id
                    )
                
            # 3. 회의 상세 질문 (RAG) - keyword_search보다 먼저
            if is_detail_question(user_query, context):
                print(f"[DEBUG] 회의 상세 질문 감지 (RAG)")
                
                selected_meeting_id = context.get('selected_meeting_id')
                meeting_title = context.get('meeting_title', '선택된 회의')
                
                # 회의 정보 가져오기
                from .database import get_db_connection
                
                with get_db_connection() as conn:
                    if not conn:
                        return ChatResponse(
                            answer="데이터베이스 연결에 실패했어요. 😢",
                            history=request.history + [
                                {"role": "user", "content": original_query},
                                {"role": "assistant", "content": "데이터베이스 연결에 실패했어요. 😢"}
                            ],
                            source="db_connection_error",
                            session_id=session_id
                        )
                    
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT 
                                m.id, 
                                m.title, 
                                m.description, 
                                m.scheduled_at, 
                                mr.summary, 
                                m.status,
                                GROUP_CONCAT(
                                    CONCAT(t.speaker_name, ': ', t.text) 
                                    ORDER BY t.sequence_order 
                                    SEPARATOR '\n'
                                ) as transcript_text
                            FROM meeting m
                            LEFT JOIN meeting_result mr ON m.id = mr.meeting_id
                            LEFT JOIN transcript t ON m.id = t.meeting_id
                            WHERE m.id = %s
                            GROUP BY m.id
                        """, (selected_meeting_id,))
                        meeting = cursor.fetchone()
                        cursor.close()
                        
                        if meeting:
                            # RAG 답변 생성
                            from .llm import answer_meeting_question
                            rag_answer = answer_meeting_question(meeting, user_query)
                            
                            return ChatResponse(
                                answer=rag_answer,
                                history=request.history + [
                                    {"role": "user", "content": original_query},
                                    {"role": "assistant", "content": rag_answer}
                                ],
                                source="meeting_detail_rag",
                                session_id=session_id
                            )
                        else:
                            return ChatResponse(
                                answer=f"❌ {meeting_title} 정보를 찾을 수 없어요.",
                                history=request.history + [
                                    {"role": "user", "content": original_query},
                                    {"role": "assistant", "content": f"❌ {meeting_title} 정보를 찾을 수 없어요."}
                                ],
                                source="meeting_not_found",
                                session_id=session_id
                            )
                    except Exception as e:
                        logger.error(f"RAG 처리 중 오류: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        return ChatResponse(
                            answer="회의 정보를 가져오는 중 오류가 발생했어요. 😢",
                            history=request.history + [
                                {"role": "user", "content": original_query},
                                {"role": "assistant", "content": "회의 정보를 가져오는 중 오류가 발생했어요. 😢"}
                            ],
                            source="rag_error",
                            session_id=session_id
                        )

            # 4. Keyword 검색 intent
            elif intent == 'keyword_search':
                # "'예산' 키워드 있는 회의?"
                print(f"[DEBUG] Keyword 검색 intent 감지")
                
                # 키워드 추출 (따옴표 있으면 따옴표 안, 없으면 첫 단어)
                keyword_pattern = re.search(r"['\"\'](.+?)['\"\']", user_query)

                if keyword_pattern:
                    keyword_name = keyword_pattern.group(1).strip()
                elif '키워드' in user_query:
                    # "전략 키워드 있는 회의?" → "전략" 추출
                    keyword_match = re.search(r'([가-힣a-zA-Z0-9]+)\s*키워드', user_query)
                    if keyword_match:
                        keyword_name = keyword_match.group(1).strip()
                    else:
                        keyword_name = None
                else:
                    keyword_name = None
                
                if keyword_name:
                    print(f"[DEBUG] Keyword 검색: '{keyword_name}'")
                    
                    from .search import search_keywords
                    keyword_response, meetings = search_keywords(
                        keyword_name=keyword_name,
                        user_job=user_job_normalized
                    )
                    
                    # 컨텍스트 저장 (단일 회의면)
                    if meetings and len(meetings) == 1:
                        save_context(session_id, {
                            'state': 'meeting_selected',
                            'selected_meeting_id': meetings[0]['id'],
                            'meeting_title': meetings[0]['title']
                        })
                        print(f"[DEBUG] 단일 회의 컨텍스트 저장: meeting_id={meetings[0]['id']}")
                    elif meetings and len(meetings) > 1:
                        meetings_serializable = []
                        for meeting in meetings:
                            meeting_copy = {}
                            for key, value in meeting.items():
                                if isinstance(value, datetime):
                                    meeting_copy[key] = value.isoformat()
                                else:
                                    meeting_copy[key] = value
                            meetings_serializable.append(meeting_copy)
                        
                        shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])

                        save_context(session_id, {
                            'state': 'awaiting_selection',
                            'shown_completed': shown_completed,
                            'shown_scheduled': shown_scheduled,
                            'meetings': meetings_serializable[:10],
                            'offset': min(5, len(meetings_serializable[:10])),
                            'total_count': len(meetings)
                        })
                        print(f"[DEBUG] 여러 회의 컨텍스트 저장: {len(meetings)}개")
                    
                    return ChatResponse(
                        answer=keyword_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": keyword_response}
                        ],
                        source="keyword_search",
                        session_id=session_id
                    )
                else:
                    # 키워드 추출 실패
                    return ChatResponse(
                        answer="어떤 키워드로 검색하시겠어요? 😊\n예: \"'예산' 키워드 있는 회의?\"",
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": "어떤 키워드로 검색하시겠어요? 😊"}
                        ],
                        source="keyword_clarification",
                        session_id=session_id
                    )

        # === Participant 질문 처리 ===
        participant_info = is_participant_query(user_query, context)
        
        # ========== LLM 의도 확인 (패턴 실패 시 보완) ==========
        if not participant_info['is_participant']:
            # LLM 전처리 결과가 있고, participant_search로 판단했으면
            if preprocessed is not None and preprocessed.get('intent') == 'participant_search':
                print(f"[DEBUG] LLM이 participant_search로 판단 → Participant 처리")
                participant_info = {
                    'is_participant': True,
                    'query_type': 'meeting_participants' if context and context.get('selected_meeting_id') else None,
                    'person_name': None
                }
        
        if participant_info['is_participant']:
            print(f"\n👥 참석자 질문 감지")
            print(f"[DEBUG] query_type: {participant_info['query_type']}")
            print(f"[DEBUG] person_name: {participant_info['person_name']}")
            
            from .search import search_participants
            
            if participant_info['query_type'] == 'meeting_participants':
                # 회의 컨텍스트 필요
                if not context or not context.get('selected_meeting_id'):
                    answer = "어떤 회의의 참석자를 알려드릴까요? 회의를 먼저 선택해주세요! 😊"
                    return ChatResponse(
                        answer=answer,
                        source="participant_query",
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": answer}
                        ],
                        session_id=session_id
                    )
                
                meeting_id = context['selected_meeting_id']
                answer, participants = search_participants(
                    query_type="meeting_participants",
                    meeting_id=meeting_id
                )
                
                # 컨텍스트 유지
                save_context(session_id, context)
                
                return ChatResponse(
                    answer=answer,
                    source="participant_query",
                    history=request.history + [
                        {"role": "user", "content": original_query},
                        {"role": "assistant", "content": answer}
                    ],
                    session_id=session_id
                )
            
            elif participant_info['query_type'] == 'person_meetings':
                person_name = participant_info['person_name']
                answer, meetings = search_participants(
                    query_type="person_meetings",
                    person_name=person_name
                )
                
                # 여러 회의면 컨텍스트 저장
                if len(meetings) > 1:
                    meetings_serializable = []
                    for m in meetings[:10]:
                        meeting_copy = {}
                        for key, value in m.items():
                            if isinstance(value, datetime):
                                meeting_copy[key] = value.isoformat()
                            else:
                                meeting_copy[key] = value
                        meetings_serializable.append(meeting_copy)
                    
                    shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])
                    
                    save_context(session_id, {
                        'state': 'awaiting_selection',
                        'shown_completed': shown_completed,
                        'shown_scheduled': shown_scheduled,
                        'last_query': user_query,
                        'meetings': meetings_serializable
                    })
                elif len(meetings) == 1:
                    # 단일 회의면 선택 상태로
                    save_context(session_id, {
                        'state': 'meeting_selected',
                        'selected_meeting_id': meetings[0]['id'],
                        'meeting_title': meetings[0]['title']
                    })
                
                return ChatResponse(
                    answer=answer,
                    source="participant_query",
                    history=request.history + [
                        {"role": "user", "content": original_query},
                        {"role": "assistant", "content": answer}
                    ],
                    session_id=session_id
                )

            from .llm import preprocess_query_with_llm
            
            preprocessed = preprocess_query_with_llm(user_query, context)
            print(f"[DEBUG] LLM 전처리 결과: {preprocessed}")
            
            corrected_query = preprocessed.get('corrected_query', user_query)
            intent = preprocessed.get('intent', 'meeting_search')
            is_contextual = preprocessed.get('is_contextual', False)
            scope_expansion = preprocessed.get('scope_expansion', False)
            
            # 보정된 쿼리 사용
            user_query = corrected_query
            
            # ========== Intent별 자동 처리 ==========
            
            # 1. Task 검색 intent
            if intent == 'task_search':
                # 컨텍스트 활용 여부 결정
                if is_contextual and context and context.get('state') == 'meeting_selected' and not scope_expansion:
                    # 특정 회의의 할일 검색
                    selected_meeting_id = context.get('selected_meeting_id')
                    meeting_title = context.get('meeting_title', '')
                    
                    print(f"[DEBUG] Task 검색 - 컨텍스트 활용: meeting_id={selected_meeting_id}")
                    
                    from .search import search_tasks
                    task_response, tasks = search_tasks(
                        user_query=user_query,
                        user_id=user_id,
                        meeting_id=selected_meeting_id
                    )
                    
                    return ChatResponse(
                        answer=task_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": task_response}
                        ],
                        source="task_search_contextual",
                        session_id=session_id
                    )
                else:
                    # 전체 회의의 할일 검색 (scope_expansion=True 또는 컨텍스트 없음)
                    print(f"[DEBUG] Task 검색 - 전체 검색 (scope_expansion={scope_expansion})")
                    
                    from .search import search_tasks
                    task_response, tasks = search_tasks(
                        user_query=user_query,
                        user_id=user_id,
                        meeting_id=None
                    )
                    
                    return ChatResponse(
                        answer=task_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": task_response}
                        ],
                        source="task_search_global",
                        session_id=session_id
                    )
            
            # 2. Participant 검색 intent
            elif intent == 'participant_search':
                # 특정 회의 참석자 vs 특정 사람이 참석한 회의
                
                # 이름 패턴 확인
                name_match = re.search(r'([가-힣]{2,4})', user_query)
                
                if name_match and any(w in user_query for w in ['참석한', '나온', '있었', '회의']):
                    # 특정 사람이 참석한 회의 검색
                    person_name = name_match.group(1)
                    # 조사 제거 (가, 이, 은, 는, 을, 를)
                    person_name = re.sub(r"[가이은는을를]$", "", person_name)
                    print(f"[DEBUG] Participant 검색 - 특정 사람: {person_name}")
                    
                    from .search import search_participants
                    participant_response, results = search_participants(
                        query_type="person_meetings",
                        person_name=person_name
                    )
                    
                    # 단일 회의면 컨텍스트 저장
                    if results and len(results) == 1:
                        meeting = results[0]
                        context = {
                            'state': 'meeting_selected',
                            'selected_meeting_id': meeting['id'],
                            'meeting_title': meeting['title'],
                            'original_query': user_query
                        }
                        save_context(session_id, context)
                        print(f"[DEBUG] 단일 회의 컨텍스트 저장: meeting_id={meeting['id']}")
                    elif results and len(results) > 1:
                        # 여러 회의 - 선택 대기 상태
                        meetings_serializable = []
                        for meeting in results:
                            meeting_copy = {}
                            for key, value in meeting.items():
                                if isinstance(value, datetime):
                                    meeting_copy[key] = value.isoformat()
                                else:
                                    meeting_copy[key] = value
                            meetings_serializable.append(meeting_copy)
                        
                        shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])

                        context = {
                            'state': 'awaiting_selection',
                            'shown_completed': shown_completed,
                            'shown_scheduled': shown_scheduled,
                            'meetings': meetings_serializable[:10],
                            'offset': min(5, len(meetings_serializable[:10])),
                            'total_count': len(results),
                            'original_query': user_query
                        }
                        save_context(session_id, context)
                        print(f"[DEBUG] 여러 회의 컨텍스트 저장: {len(results)}개")
                    
                    return ChatResponse(
                        answer=participant_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": participant_response}
                        ],
                        source="participant_person_meetings",
                        session_id=session_id
                    )
                
                elif is_contextual and context and context.get('state') == 'meeting_selected':
                    # 특정 회의의 참석자 조회
                    selected_meeting_id = context.get('selected_meeting_id')
                    print(f"[DEBUG] Participant 검색 - 특정 회의: meeting_id={selected_meeting_id}")
                    
                    from .search import search_participants
                    participant_response, results = search_participants(
                        query_type="meeting_participants",
                        meeting_id=selected_meeting_id
                    )
                    
                    return ChatResponse(
                        answer=participant_response,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": participant_response}
                        ],
                        source="participant_meeting_members",
                        session_id=session_id
                    )
                
                else:
                    # 참석자 정보 부족
                    fallback_msg = "누구의 참석 정보를 알려드릴까요? 😊\n예: '김철수가 참석한 회의', '채용 전략 회의 참석자'"
                    
                    return ChatResponse(
                        answer=fallback_msg,
                        history=request.history + [
                            {"role": "user", "content": original_query},
                            {"role": "assistant", "content": fallback_msg}
                        ],
                        source="participant_clarification",
                        session_id=session_id
                    )

        # ========== 기존 레거시 패턴 매칭 (LLM 처리 안 된 경우만) ==========

        # ========== RAG 상세 질문 우선 처리 ==========
        detail_keywords = ['예산', '얼마', '금액', '비용', '언제', '시간', '몇 시', '어떻게', '왜', '방법', '결과', '결론', '내용', '주요', '핵심', '이유', '목적', '참석자', '참여자', '발표자', '분위기', '반응', '의견', '어땠', '어떻']
        is_detail_question_early = any(keyword in user_query for keyword in detail_keywords)
        
        if (context and context.get('state') == 'meeting_selected' and 
            is_detail_question_early and
            not any(keyword in user_query for keyword in ['할일', '할 일', 'task', '담당', '맡은'])):
            
            print(f"[DEBUG] RAG 상세 질문 우선 처리: '{user_query}'")
            
            selected_meeting_id = context.get('selected_meeting_id')
            
            from .database import get_db_connection
            
            with get_db_connection() as conn:
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT 
                                m.id, 
                                m.title, 
                                m.description, 
                                m.scheduled_at, 
                                mr.summary, 
                                m.status,
                                GROUP_CONCAT(
                                    CONCAT(t.speaker_name, ': ', t.text) 
                                    ORDER BY t.sequence_order 
                                    SEPARATOR '\n'
                                ) as transcript_text
                            FROM meeting m
                            LEFT JOIN meeting_result mr ON m.id = mr.meeting_id
                            LEFT JOIN transcript t ON m.id = t.meeting_id
                            WHERE m.id = %s
                            GROUP BY m.id
                        """, (selected_meeting_id,))
                        meeting = cursor.fetchone()
                        cursor.close()
                        
                        if meeting:
                            from .llm import answer_meeting_question
                            rag_answer = answer_meeting_question(meeting, user_query)
                            
                            return ChatResponse(
                                answer=rag_answer,
                                history=request.history + [
                                    {"role": "user", "content": original_query},
                                    {"role": "assistant", "content": rag_answer}
                                ],
                                source="meeting_detail_rag",
                                session_id=session_id
                            )
                    except Exception as e:
                        logger.error(f"RAG 처리 오류: {e}")

        # [0-2단계] Task 질문 체크
        # 컨텍스트 기반 Task 질문 감지
        previous_was_task = False
        previous_was_meeting_detail = False
        previous_meeting_id = None  

        if request.history and len(request.history) >= 2:
            last_response = request.history[-1].get('content', '') if request.history[-1].get('role') == 'assistant' else ''
            
            # 이전 답변이 Task 관련
            if '할 일' in last_response or '담당:' in last_response:
                previous_was_task = True
            
            # 이전 답변이 회의 상세 (📌 포함)
            if '📌' in last_response and '회의' in last_response:
                previous_was_meeting_detail = True

        # 컨텍스트에서 meeting_id 확인 (개수 확인 질문 제외)
        if context and context.get('selected_meeting_id') and not is_count_check:
            previous_meeting_id = context['selected_meeting_id']
            print(f"[DEBUG] 컨텍스트에 저장된 meeting_id: {previous_meeting_id}")

        # 강한 새 질문 신호 체크
        has_date = parse_date_from_query(user_query).get('type') is not None
        has_search_verb = any(kw in user_query for kw in ['찾아', '검색', '보여', '알려', '조회'])
        has_status = any(kw in user_query for kw in ['완료', '예정', '지난', '최근'])
        is_clear_new_query = has_date or has_search_verb or has_status or len(user_query) > 20

        # Task 패턴
        task_patterns = ['맡은 일', '담당', '해야 할', 'task', '액션', '할일', '할 일', '다른 사람', '다른사람', '누가', '누구']
        is_task_question = any(pattern in user_query.lower() for pattern in task_patterns)

        # "전체" 키워드 체크 (정확한 매칭)
        is_asking_all_tasks = False
        if '전체' in user_query or '모든' in user_query or '전부' in user_query:
            is_asking_all_tasks = True
        elif ' 다 ' in user_query or user_query.startswith('다 ') or user_query.endswith(' 다'):
            is_asking_all_tasks = True

        # 1-1. "이 회의", "그 회의", "저 회의" + Task 단어 (명시적)
        meeting_id_from_meeting_ref = None
        if any(word in user_query for word in ['이 회의', '그 회의', '저 회의', '해당 회의']):
            if any(kw in user_query for kw in ['할', '맡', '담당', '일', '해야', 'task', '사람', '누가', '누구']):  
                if context and context.get('selected_meeting_id'):
                    is_task_question = True
                    meeting_id_from_meeting_ref = context['selected_meeting_id']
                    print(f"[DEBUG] '이 회의' + Task 단어 감지 → meeting_id={meeting_id_from_meeting_ref}")

        # 1-2. 이전이 회의 상세 + 짧은 질문 + Task 관련 단어
        if not is_clear_new_query and previous_was_meeting_detail and len(user_query) <= 15:
            if any(kw in user_query for kw in ['할', '맡', '담당', '일', '해야', '사람', '누가', '누구']):  
                is_task_question = True
                print(f"[DEBUG] 암묵적 Task 질문 감지 (이전: 회의 상세)")

        # 1-3. 이전이 Task + 짧은 질문 + "다른 사람" 패턴
        if not is_clear_new_query and previous_was_task and len(user_query) <= 15:
            if any(word in user_query for word in ['다른', '누가', '누구', '사람']):
                is_task_question = True
                print(f"[DEBUG] 이전 Task + '다른 사람' 패턴 감지")

        # 2. 이전이 Task + 회의 언급
        elif not is_clear_new_query and previous_was_task and len(user_query) <= 20:
            if any(word in user_query for word in ['회의', '저기', '거기', '안에서', '에서', '저', '그']):
                is_task_question = True
                print(f"[DEBUG] 컨텍스트 기반 Task 질문 감지")

        # 3. "아니" + 회의 맥락 (정정 패턴)
        if any(word in user_query for word in ['아니', '그게 아니']):
            if context and context.get('selected_meeting_id') and len(user_query) <= 20:
                if any(word in user_query for word in ['회의', '저', '그', '거기', '저기', '안에서', '에서']):
                    is_task_question = True
                    print(f"[DEBUG] 정정 패턴 감지 → Task 질문")

        # 4. 컨텍스트에 meeting_id 있음 + 회의 범위 지정 + Task 단어
        elif not is_clear_new_query and context and context.get('selected_meeting_id') and len(user_query) <= 20:
            has_meeting_ref = any(word in user_query for word in ['저기', '거기', '안에서', '에서', '저', '그'])
            has_task_word = any(word in user_query for word in ['할', '일', '맡', '담당', '해야', 'task', '사람', '누가', '누구'])  
            
            if has_meeting_ref and has_task_word:
                is_task_question = True
                print(f"[DEBUG] 회의 컨텍스트 + 범위 지정어 + Task 단어 → Task 질문")

        if is_task_question:
            print(f"[DEBUG] Task 질문 감지")
            
            from .search import search_tasks
            
            # "전체" 키워드가 있으면 meeting_id 무시
            meeting_id = None
            if not is_asking_all_tasks:  
                # 컨텍스트에서 selected_meeting_id 확인
                meeting_id = meeting_id_from_meeting_ref
                if not meeting_id and context and context.get('selected_meeting_id'):
                    meeting_id = context['selected_meeting_id']
            
            if is_asking_all_tasks:
                print(f"[DEBUG] '전체' 키워드 감지 → meeting_id 무시")
            
            print(f"[DEBUG] Task 검색 meeting_id: {meeting_id}")
            
            message, tasks = search_tasks(user_query, user_id=user_id, meeting_id=meeting_id)
            return ChatResponse(
                answer=message,
                history=request.history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": message}
                ],
                source="task_query",
                session_id=session_id
            )

        # ========== 컨텍스트 유지 여부 판단 ==========
        should_use_context = False

        if context and context.get('state') == 'awaiting_selection':
            # 먼저 날짜 정보 체크!
            date_info_check = parse_date_from_query(user_query)
            has_date_info = date_info_check.get('type') is not None
            
            # 날짜 정보가 있으면 무조건 새 검색
            if has_date_info:
                should_use_context = False
                delete_context(session_id)
                context = None
                print(f"[DEBUG] 날짜 정보 감지 ({date_info_check.get('original')}) → 새로운 검색")
            else:
                # 명확한 선택 패턴 체크
                selection_patterns = [
                    r'^\d+$',  # 숫자만
                    r'^\d+번$',  # 1번, 2번
                    r'^(첫|마지막)',  # 첫 번째, 마지막
                ]
                
                is_clear_selection = any(re.match(p, user_query.strip()) for p in selection_patterns)
                
                # 새로운 검색 의도 체크
                search_keywords = ['찾아', '검색', '알려', '보여', '회의', '미팅', '있어', '있었', '있나']
                has_search_word = any(kw in user_query for kw in search_keywords)
                
                # 판단: "나머지"는 특별 처리
                if any(word in user_query.lower() for word in ['나머지', '더', '더보기', '추가', '계속']):
                    should_use_context = False  # 컨텍스트는 유지하되, handle_selection으로 안 넘김
                    print(f"[DEBUG] '나머지' 요청 감지 → 특별 처리")
                elif is_clear_selection:
                    should_use_context = True
                    print(f"[DEBUG] 선택 의도 감지 → 컨텍스트 사용")
                else:
                    # 새로운 검색으로 처리
                    should_use_context = False
                    delete_context(session_id)
                    print(f"[DEBUG] 새로운 검색 의도 감지 → 컨텍스트 무시")
                    
        # ========== 후속 질문 처리 (LLM 활용) ==========
        elif context and not is_obvious_pattern(user_query):
            # "나머지" 요청 감지 (정규식 + 오타 허용)
            query_lower = user_query.lower().replace(' ', '')  # 띄어쓰기 제거
            
            pagination_patterns = [
                r'나머.*',
                r'남은.*',
                r'더.*[보줘있알려]',
                r'추가.*',
                r'계속|이어서|다음',
                r'또.*[있뭐어]',
                r'그\s*외',
                r'더\s*[보줘]',
                r'완료.*나머',
                r'완료.*더',
                r'예정.*나머',
                r'예정.*더'
            ]
            
            # 오타 허용 키워드
            fuzzy_pagination = ['나머', '남머', '나미', '너머', '더보', '더줘', '더있', '더알', '추가로', '계속', '이어']
            
            is_pagination = (
                any(re.search(pattern, query_lower) for pattern in pagination_patterns) or
                any(fuzzy in query_lower for fuzzy in fuzzy_pagination)
            )
            
            # ========== 상태 키워드 체크 (새 검색) ==========
            status_keywords = ['완료된', '예정된', '진행중', '취소된']
            is_status_only = user_query.strip() in status_keywords  # "완료된"만 입력
            
            # 검색 의도가 명확하지 않고 짧은 질문 (단, 페이지네이션/상태 키워드 아닐 때만)
            if not is_pagination and not is_status_only and len(user_query) < 20 and not any(w in user_query for w in ['찾아', '검색', '회의', '뭐있어']):
                print(f"[DEBUG] 후속 질문 가능성 → LLM으로 확인")
                
                from .llm import answer_with_context
                llm_answer = answer_with_context(user_query, context)
                
                return ChatResponse(
                    answer=llm_answer,
                    history=request.history + [
                        {"role": "user", "content": original_query},
                        {"role": "assistant", "content": llm_answer}
                    ],
                    source="context_followup",
                    session_id=session_id
                )
            elif is_pagination:
                print(f"[DEBUG] 페이지네이션 요청 감지 → awaiting_selection 처리로 이동")
            elif is_status_only:
                print(f"[DEBUG] 상태 키워드만 입력 → 새 검색으로 처리")
                delete_context(session_id)
                # 아래 MySQL 검색으로 진행

        # === 컨텍스트 기반 선택 처리 ===
        if should_use_context and context.get('state') == 'awaiting_selection':
            print(f"[DEBUG] 컨텍스트 내 선택 처리: {user_query}")
            return handle_selection(user_query, context, request, session_id)

        # ========== 통계 결과 후속 질문 처리 (Phase 3) ==========
        if context and context.get('state') == 'count_result':
            print(f"[DEBUG] 통계 결과 컨텍스트 있음")
            
            meetings = context.get('meetings', [])
            total_count = context.get('total_count', 0)
            
            # ========== 0-1. 숫자만 입력 (번호 선택) ==========
            if user_query.strip().isdigit():
                selected_number = int(user_query.strip())
                print(f"[DEBUG] 번호 선택: {selected_number}번")
                
                shown_completed = context.get('shown_completed', 3)
                shown_scheduled = context.get('shown_scheduled', 3)
                
                # 번호 → 회의 매핑
                if 1 <= selected_number <= shown_completed:
                    # 완료된 회의 선택
                    completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
                    selected_meeting = completed_meetings[selected_number - 1]
                    print(f"[DEBUG] 완료된 회의 선택: {selected_number}번 → {selected_meeting.get('title')}")
                
                elif shown_completed < selected_number <= (shown_completed + shown_scheduled):
                    # 예정된 회의 선택
                    scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
                    scheduled_index = selected_number - shown_completed - 1
                    selected_meeting = scheduled_meetings[scheduled_index]
                    print(f"[DEBUG] 예정된 회의 선택: {selected_number}번 → {selected_meeting.get('title')}")
                
                # ========== 상태 없이 숫자만 입력 ==========
                else:
                    # 완료/예정 둘 다 해당 번호 있는지 체크
                    has_completed = 1 <= selected_number <= shown_completed
                    has_scheduled = 1 <= selected_number <= shown_scheduled
                    
                    if has_completed and has_scheduled:

                        # 명확화 질문
                        context['last_source'] = 'ambiguous_number'
                        context['last_ambiguous_number'] = selected_number
                        save_context(session_id, context)
                        
                        # 둘 다 있음 → 명확화 질문
                        return ChatResponse(
                            answer=f"완료된 회의와 예정된 회의 모두 {selected_number}번이 있어요! 🤔\n\n어떤 회의를 보시겠어요?\n\n💬 \"완료 {selected_number}\" - 완료된 {selected_number}번 회의\n💬 \"예정 {selected_number}\" - 예정된 {selected_number}번 회의",
                            source="ambiguous_number",
                            session_id=session_id
                        )
                    
                    elif has_completed:
                        # 완료만 있음
                        completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
                        selected_meeting = completed_meetings[selected_number - 1]
                        print(f"[DEBUG] 완료 {selected_number}번 선택 (자동): {selected_meeting.get('title')}")
                    
                    elif has_scheduled:
                        # 예정만 있음
                        scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
                        scheduled_index = selected_number - shown_completed - 1
                        selected_meeting = scheduled_meetings[scheduled_index]
                        print(f"[DEBUG] 예정 {selected_number}번 선택 (자동): {selected_meeting.get('title')}")
                    
                    else:
                        # 둘 다 없음
                        total_shown = shown_completed + shown_scheduled
                        return ChatResponse(
                            answer=f"❌ {selected_number}번은 없어요!\n\n완료: 1~{shown_completed}번\n예정: 1~{shown_scheduled}번\n중에서 선택해주세요! 😊",
                            source="invalid_number",
                            session_id=session_id
                        )

                # 페르소나 템플릿 적용
                answer = format_single_meeting_with_persona(selected_meeting, user_job_normalized)

                return ChatResponse(
                    answer=answer,
                    history=request.history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": answer}
                    ],
                    source="meeting_selection_by_number",
                    session_id=session_id
)
                        
            # ========== 1. "나머지", "더" 요청 감지 (오타 허용) ==========
            more_keywords = ['나머지', '나머지도', '남은', '남은거', '더', '더보기', '더보여', 
                            '더있어', '더줘', '더알려', '추가', '추가로', '계속', '이어서', 
                            '다음', '다른', '또', '그외', '외', '그밖', '더있나', '더있니', 
                            '또뭐', '또있어', '나머', '남머', '나미', '더보']

            more_patterns = [
                r'나머.*',
                r'남은.*',
                r'더.*[보줘있알려]',
                r'추가.*',
                r'계속|이어서|다음',
                r'또.*[있뭐어]',
                r'그\s*외',
                r'더\s*[보줘]'
            ]

            wants_more = any(keyword in user_query for keyword in more_keywords) or \
                        any(re.search(pattern, user_query) for pattern in more_patterns)

            # 오타 허용 (부분 매칭)
            if not wants_more:
                fuzzy_more = ['나머', '남머', '나미', '너머', '더보', '더줘', '더있', '더알', '추가']
                if any(x in user_query for x in fuzzy_more):
                    wants_more = True
                    print(f"[DEBUG] 유사 단어 감지 (오타 허용)")
                    
            # 숫자 패턴 감지 ("3개", "5개", "두 개")
            number_match = re.search(r'(\d+)개', user_query)
            korean_numbers = {'한': 1, '두': 2, '세': 3, '네': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10}
            korean_match = None
            for korean, num in korean_numbers.items():
                if korean in user_query and '개' in user_query:
                    korean_match = num
                    break
            
            if number_match or korean_match or '몇개' in user_query:
                wants_more = True
            
            if wants_more and len(meetings) > 5:
                print(f"[DEBUG] 통계 결과 나머지 요청: '{user_query}'")
                
                # ========== 현재 어디까지 보여줬는지 추적 ==========
                last_shown_index = context.get('last_shown_index', 5)  # 기본값: 5개까지 봄
                
                # 요청한 개수 파싱 (기본값: 5개씩)
                requested_count = 5
                if number_match:
                    requested_count = int(number_match.group(1))
                elif korean_match:
                    requested_count = korean_match
                
                # 다음 범위 계산
                start_idx = last_shown_index
                end_idx = min(start_idx + requested_count, len(meetings))
                
                remaining_meetings = meetings[start_idx:end_idx]
                
                if not remaining_meetings:
                    answer = "더 이상 회의가 없어요! 😊\n\n이미 모든 회의를 보여드렸습니다."
                    return ChatResponse(
                        answer=answer,
                        history=request.history,
                        source="no_more_meetings",
                        session_id=session_id
                    )
                
                # ========== 상세 포맷으로 보여주기 ==========
                answer = f"나머지 회의들이에요! 📋\n\n"
                
                for i, meeting in enumerate(remaining_meetings):
                    actual_number = start_idx + i + 1
                    emoji = f"📌 {actual_number}."
                    
                    title = meeting.get('title', '제목 없음')
                                                        
                    scheduled_at = meeting.get('scheduled_at')
                    if isinstance(scheduled_at, str):
                        scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                    date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                    
                    # summary 또는 description
                    summary = meeting.get('summary', '')
                    if not summary or summary.strip() == '':
                        summary = meeting.get('description', '내용 없음')
                    
                    # 1-2문장 (80자)
                    lines = summary.split('.')[:2]
                    display_text = '. '.join([line.strip() for line in lines if line.strip()])
                    if len(display_text) > 80:
                        display_text = display_text[:80] + "..."
                    
                    answer += f"{emoji} {title} {date_str}\n"
                    answer += f"   - {display_text}\n\n"
                
                # 남은 개수 계산
                shown_total = end_idx
                remaining_count = total_count - shown_total
                
                if remaining_count > 0:
                    answer += f"💡 이 외에도 {remaining_count}개가 더 있어요!\n"
                    answer += "\"더 보여줘\" 또는 \"나머지\" 라고 하시면 계속 볼 수 있어요.\n\n"
                else:
                    answer += "✅ 모든 회의를 보여드렸어요!\n\n"
                
                answer += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
                answer += f"예: 번호({start_idx + 1}, {start_idx + 2}), 제목(디자인 회의) 😊"
                
                # ========== 컨텍스트 업데이트 (진행 상황 저장) ==========
                context['state'] = 'awaiting_selection'
                context['last_shown_index'] = end_idx  # 어디까지 봤는지 저장
                save_context(session_id, context)
                
                return ChatResponse(
                    answer=answer,
                    history=request.history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": answer}
                    ],
                    source="count_remaining",
                    session_id=session_id
                )
            
            # ========== 2. "보여줘", "자세히" 등 상세 요청 ==========
            show_keywords = ['보여', '자세히', '상세', '리스트', '목록']
            wants_details = any(keyword in user_query for keyword in show_keywords)
            
            if wants_details:
                print(f"[DEBUG] 통계 결과 상세 요청: '{user_query}'")
                meetings = context.get('meetings', [])
                total_count = context.get('total_count', 0)
                
                if meetings:
                    # ========== Phase 2-A: 페르소나 정렬 적용 ==========
                    if ENABLE_PERSONA and len(meetings) > 1:
                        meetings = search_with_persona(meetings, user_job_normalized)
                        print(f"[DEBUG] Phase 2-A (통계 결과): {user_job_normalized} 관련도 순으로 정렬")
                    
                    # 여러 회의 포맷으로 보여주기
                    answer, shown_completed, shown_scheduled = format_multiple_meetings_short(
                        meetings[:10],
                        user_query,
                        total_count if total_count > 10 else None,
                        None,  # date_info
                        None   # status
                    )
                    
                    # ========== 컨텍스트를 awaiting_selection으로 변경 ==========
                    # datetime → str 변환 (정렬된 meetings 사용)
                    meetings_serializable = []
                    for meeting in meetings[:10]:
                        meeting_copy = {}
                        for key, value in meeting.items():
                            if isinstance(value, datetime):
                                meeting_copy[key] = value.isoformat()
                            else:
                                meeting_copy[key] = value
                        meetings_serializable.append(meeting_copy)
                    
                    context['state'] = 'awaiting_selection'
                    context['meetings'] = meetings_serializable  # 정렬된 결과로 업데이트
                    save_context(session_id, context)
                    
                    return ChatResponse(
                        answer=answer,
                        history=request.history + [
                            {"role": "user", "content": user_query},
                            {"role": "assistant", "content": answer}
                        ],
                        source="count_details",
                        session_id=session_id
                    )

        # ============================================================
        
        # ========== awaiting_selection 처리 (회의 선택 대기) ==========
        if context and context.get('state') == 'awaiting_selection':
            print(f"[DEBUG] 컨텍스트 있음 (state: {context.get('state')})")
            
            meetings = context.get('meetings', [])
            
            if meetings:
                # ========== 0-0. 상태 필터링 ("완료", "예정" 단독 입력) ==========
                status_filter_keywords = {
                    '완료': 'COMPLETED',
                    '완료된': 'COMPLETED', 
                    '예정': 'SCHEDULED',
                    '예정된': 'SCHEDULED'
                }

                # 명확화 질문 직후인지 체크
                last_source = context.get('last_source')
                print(f"[DEBUG] last_source: {last_source}, last_number: {context.get('last_ambiguous_number')}")
                if last_source == 'ambiguous_number' and user_query.strip() in ['완료', '예정']:
                    # "완료" 또는 "예정"만 입력 시 → 이전 번호 재사용
                    last_number = context.get('last_ambiguous_number')
                    if last_number:
                        user_query = f"{user_query} {last_number}"
                        print(f"[DEBUG] 명확화 질문 직후 → 쿼리 확장: {user_query}")
                        
                # ========== 쿼리 확장 적용 후 다시 체크 ==========
                query_to_check = user_query.strip()  # 확장된 쿼리 사용

                if query_to_check in status_filter_keywords:
                    target_status = status_filter_keywords[query_to_check]
                    print(f"[DEBUG] 상태 필터링 요청: {target_status}")

                    # 해당 상태의 회의만 필터링
                    filtered_meetings = [m for m in meetings if m.get('status') == target_status]
                    
                    if not filtered_meetings:
                        status_name = "완료된" if target_status == 'COMPLETED' else "예정된"
                        return ChatResponse(
                            answer=f"❌ {status_name} 회의가 없어요!",
                            source='no_meetings_with_status',
                            session_id=session_id
                        )
                    
                    # 기존 shown 값 가져오기
                    if target_status == 'COMPLETED':
                        current_offset = context.get('shown_completed', 0)
                    else:
                        current_offset = context.get('shown_scheduled', 0)
                    
                    # 오프셋 기준으로 다음 배치 가져오기
                    remaining_meetings = filtered_meetings[current_offset:]
                    
                    if not remaining_meetings:
                        status_name = "완료된" if target_status == 'COMPLETED' else "예정된"
                        return ChatResponse(
                            answer=f"✅ {status_name} 회의를 모두 보여드렸어요! 😊",
                            source='no_more_with_status',
                            session_id=session_id
                        )
                    
                    # 다음 5개 (또는 남은 개수)
                    next_batch = remaining_meetings[:5]
                    start_number = current_offset + 1
                    
                    answer = f"나머지 {'완료된' if target_status == 'COMPLETED' else '예정된'} 회의예요! 📋\n\n"

                    for i, meeting in enumerate(next_batch):
                        actual_number = start_number + i
                        
                        scheduled_at = meeting.get('scheduled_at')
                        if isinstance(scheduled_at, str):
                            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                        
                        date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                        title = meeting.get('title', '제목 없음')
                        
                        summary = meeting.get('summary', '')
                        if not summary or summary.strip() == '':
                            summary = meeting.get('description', '내용 없음')
                        
                        lines = summary.split('.')[:2]
                        display_text = '. '.join([line.strip() for line in lines if line.strip()])
                        if len(display_text) > 80:
                            display_text = display_text[:80] + "..."
                        
                        answer += f" 📌 {actual_number}. {title} {date_str}\n"
                        answer += f"   - {display_text}\n\n"
                    
                    # 남은 개수 계산
                    total_shown = current_offset + len(next_batch)
                    remaining_count = len(filtered_meetings) - total_shown
                    
                    if remaining_count > 0:
                        answer += f"💡 {remaining_count}개가 더 있어요!\n"
                        answer += f"\"{'완료된' if target_status == 'COMPLETED' else '예정된'} 나머지\" 또는 \"나머지\" 라고 하시면 계속 볼 수 있어요.\n\n"
                    else:
                        answer += "✅ 모든 회의를 보여드렸어요!\n\n"

                    answer += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
                    answer += f"예: 번호({start_number}, {start_number+1}), 날짜, 제목 😊"
                    
                    # 컨텍스트 업데이트
                    if target_status == 'COMPLETED':
                        context['shown_completed'] = total_shown
                    else:
                        context['shown_scheduled'] = total_shown
                    
                    save_context(session_id, context)
                    
                    return ChatResponse(
                        answer=answer,
                        source='status_filter_in_context',
                        session_id=session_id
                    )
                
                # ========== 0-1. 번호 선택 ==========
                # "완료 2", "예정 2", "2" 패턴 감지
                number_pattern = r'(?:(완료|예정)\s*)?(\d+)'
                number_match = re.match(number_pattern, user_query.strip())

                if number_match:
                    status_prefix = number_match.group(1)  # "완료" 또는 "예정" 또는 None
                    selected_number = int(number_match.group(2))
                    
                    completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
                    scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
                    
                    shown_completed = context.get('shown_completed', 3)
                    shown_scheduled = context.get('shown_scheduled', 3)
                    
                    # ========== 상태 명시된 경우 ==========
                    if status_prefix == '완료':
                        if 1 <= selected_number <= shown_completed:
                            selected_meeting = completed_meetings[selected_number - 1]
                            print(f"[DEBUG] 완료 {selected_number}번 선택: {selected_meeting.get('title')}")
                        else:
                            return ChatResponse(
                                answer=f"❌ 완료 {selected_number}번은 없어요!\n완료 1번부터 {shown_completed}번까지 선택할 수 있어요. 😊",
                                source="invalid_number",
                                session_id=session_id
                            )
                    
                    elif status_prefix == '예정':
                        if 1 <= selected_number <= shown_scheduled:
                            selected_meeting = scheduled_meetings[selected_number - 1]
                            print(f"[DEBUG] 예정 {selected_number}번 선택: {selected_meeting.get('title')}")
                        else:
                            return ChatResponse(
                                answer=f"❌ 예정 {selected_number}번은 없어요!\n예정 1번부터 {shown_scheduled}번까지 선택할 수 있어요. 😊",
                                source="invalid_number",
                                session_id=session_id
                            )
                    
                    # ========== 상태 없이 숫자만 입력 ==========
                    else:
                        # 완료/예정 둘 다 해당 번호 있는지 체크
                        has_completed = 1 <= selected_number <= shown_completed
                        has_scheduled = 1 <= selected_number <= shown_scheduled
                        
                        if has_completed and has_scheduled:

                            # 명확화 질문
                            context['last_source'] = 'ambiguous_number'
                            context['last_ambiguous_number'] = selected_number
                            save_context(session_id, context)

                            # 둘 다 있음 → 명확화 질문
                            return ChatResponse(
                                answer=f"완료된 회의와 예정된 회의 모두 {selected_number}번이 있어요! 🤔\n\n어떤 회의를 보시겠어요?\n\n💬 \"완료 {selected_number}\" - 완료된 {selected_number}번 회의\n💬 \"예정 {selected_number}\" - 예정된 {selected_number}번 회의",
                                source="ambiguous_number",
                                session_id=session_id
                            )
                        
                        elif has_completed:
                            # 완료만 있음
                            selected_meeting = completed_meetings[selected_number - 1]
                            print(f"[DEBUG] 완료 {selected_number}번 선택 (자동): {selected_meeting.get('title')}")
                        
                        elif has_scheduled:
                            # 예정만 있음
                            selected_meeting = scheduled_meetings[selected_number - 1]
                            print(f"[DEBUG] 예정 {selected_number}번 선택 (자동): {selected_meeting.get('title')}")
                        
                        else:
                            # 둘 다 없음
                            return ChatResponse(
                                answer=f"❌ {selected_number}번은 없어요!\n\n완료: 1~{shown_completed}번\n예정: 1~{shown_scheduled}번\n중에서 선택해주세요! 😊",
                                source="invalid_number",
                                session_id=session_id
                            )
                    
                    # 페르소나 템플릿 적용
                    answer = format_single_meeting_with_persona(selected_meeting, user_job_normalized)

                    # 컨텍스트 저장
                    new_context = {
                        'state': 'meeting_selected',
                        'selected_meeting_id': selected_meeting['id'],
                        'meeting_title': selected_meeting.get('title', ''),
                        'meetings': context.get('meetings', []),
                        'shown_completed': context.get('shown_completed', 3),
                        'shown_scheduled': context.get('shown_scheduled', 3),
                    }
                    save_context(session_id, new_context)

                    return ChatResponse(
                        answer=answer,
                        history=request.history + [
                            {"role": "user", "content": user_query},
                            {"role": "assistant", "content": answer}
                        ],
                        source="meeting_selection_by_number",
                        session_id=session_id
                    )
                
                # ========== 0-2. "나머지", "더" 요청 감지 (완료/예정 분리 + 정규식) ==========
                query_lower = user_query.lower().replace(' ', '')  # 띄어쓰기 제거

                # 1. 완료된 회의 나머지 요청 (정규식 + 오타 허용)
                completed_patterns = [
                    r'완료.*나머',
                    r'완료.*더',
                    r'완료.*추가',
                    r'완료.*남',
                ]

                if any(re.search(pattern, query_lower) for pattern in completed_patterns):
                    print(f"[DEBUG] 완료된 회의 나머지 요청")
                    
                    completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
                    shown_completed = context.get('shown_completed', 3)
                    remaining = completed_meetings[shown_completed:]
                    
                    if not remaining:
                        return ChatResponse(
                            answer="✅ 완료된 회의를 모두 보여드렸어요! 😊",
                            source='no_more_completed',
                            session_id=session_id
                        )
                    
                    next_batch = remaining[:5]
                    answer = "나머지 완료된 회의예요! 📋\n\n"
                    
                    # ========== 완료된 회의만 표시, 번호는 shown_completed + 1부터 ==========
                    start_number = shown_completed + 1
                    
                    for i, meeting in enumerate(next_batch):
                        actual_number = start_number + i  # 4, 5, 6, 7, 8
                        
                        scheduled_at = meeting.get('scheduled_at')
                        if isinstance(scheduled_at, str):
                            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                        
                        date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                        title = meeting.get('title', '제목 없음')
                        
                        summary = meeting.get('summary', '')
                        if not summary or summary.strip() == '':
                            summary = meeting.get('description', '내용 없음')
                        
                        lines = summary.split('.')[:2]
                        display_text = '. '.join([line.strip() for line in lines if line.strip()])
                        if len(display_text) > 80:
                            display_text = display_text[:80] + "..."
                        
                        answer += f"📌 {actual_number}. {title} {date_str}\n"
                        answer += f"   - {display_text}\n\n"
                    
                    total_shown = shown_completed + len(next_batch)
                    remaining_count = len(completed_meetings) - total_shown
                    
                    if remaining_count > 0:
                        answer += f"💡 완료된 회의가 {remaining_count}개 더 있어요!\n"
                        answer += "\"완료된 나머지\" 라고 하시면 계속 볼 수 있어요.\n\n"
                    else:
                        answer += "✅ 완료된 회의를 모두 보여드렸어요!\n\n"
                    
                    answer += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
                    answer += f"예: 번호({start_number}, {start_number+1}), 날짜, 제목 😊"
                    
                    # 컨텍스트 업데이트
                    context['shown_completed'] = total_shown
                    save_context(session_id, context)
                    
                    return ChatResponse(
                        answer=answer,
                        source='more_completed_meetings',
                        session_id=session_id
                    )

                # 2. 예정된 회의 나머지 요청 (정규식 + 오타 허용)
                scheduled_patterns = [
                    r'예정.*나머',
                    r'예정.*더',
                    r'예정.*추가',
                    r'예정.*남',
                ]

                if any(re.search(pattern, query_lower) for pattern in scheduled_patterns):
                    print(f"[DEBUG] 예정된 회의 나머지 요청")
                    
                    scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
                    shown_scheduled = context.get('shown_scheduled', 3)
                    remaining = scheduled_meetings[shown_scheduled:]
                    
                    if not remaining:
                        return ChatResponse(
                            answer="✅ 예정된 회의를 모두 보여드렸어요! 😊",
                            source='no_more_scheduled',
                            session_id=session_id
                        )
                    
                    next_batch = remaining[:5]
                    answer = "나머지 예정된 회의예요! 📋\n\n"
                    
                    # ========== 예정된 회의만 표시, 번호는 shown_scheduled + 1부터 ==========
                    start_number = shown_scheduled + 1
                    
                    for i, meeting in enumerate(next_batch):
                        actual_number = start_number + i  # 4, 5, 6, 7, 8
                        
                        scheduled_at = meeting.get('scheduled_at')
                        if isinstance(scheduled_at, str):
                            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                        
                        date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                        title = meeting.get('title', '제목 없음')
                        
                        summary = meeting.get('summary', '')
                        if not summary or summary.strip() == '':
                            summary = meeting.get('description', '내용 없음')
                        
                        lines = summary.split('.')[:2]
                        display_text = '. '.join([line.strip() for line in lines if line.strip()])
                        if len(display_text) > 80:
                            display_text = display_text[:80] + "..."
                        
                        answer += f"📌 {actual_number}. {title} {date_str}\n"
                        answer += f"   - {display_text}\n\n"
                    
                    total_shown = shown_scheduled + len(next_batch)
                    remaining_count = len(scheduled_meetings) - total_shown
                    
                    if remaining_count > 0:
                        answer += f"💡 예정된 회의가 {remaining_count}개 더 있어요!\n"
                        answer += "\"예정된 나머지\" 라고 하시면 계속 볼 수 있어요.\n\n"
                    else:
                        answer += "✅ 예정된 회의를 모두 보여드렸어요!\n\n"
                    
                    answer += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
                    answer += f"예: 번호({start_number}, {start_number+1}), 날짜, 제목 😊"
                    
                    # 컨텍스트 업데이트
                    context['shown_scheduled'] = total_shown
                    save_context(session_id, context)
                    
                    return ChatResponse(
                        answer=answer,
                        source='more_scheduled_meetings',
                        session_id=session_id
                    )

                # 3. 일반 "나머지" - 키워드 + 정규식 + 오타 허용
                more_keywords = ['나머지', '나머지도', '남은', '남은거', '더', '더보기', '더보여', 
                                '더있어', '더줘', '더알려', '추가', '추가로', '계속', '이어서', 
                                '다음', '또', '그외', '외', '그밖', '더있나', '더있니', 
                                '또뭐', '또있어', '나머', '남머', '나미', '더보']

                more_patterns = [
                    r'나머.*',
                    r'남은.*',
                    r'더.*[보줘있알려]',
                    r'추가.*',
                    r'계속|이어서|다음',
                    r'또.*[있뭐어]',
                    r'그\s*외',
                    r'더\s*[보줘]'
                ]

                wants_more = any(keyword in user_query for keyword in more_keywords) or \
                            any(re.search(pattern, user_query) for pattern in more_patterns)

                # 오타 허용 (띄어쓰기 제거 버전에서도 체크)
                if not wants_more:
                    fuzzy_more = ['나머', '남머', '나미', '너머', '더보', '더줘', '더있', '더알']
                    if any(x in query_lower for x in fuzzy_more):
                        wants_more = True
                        print(f"[DEBUG] 유사 단어 감지 (오타 허용)")

                if wants_more:
                    print(f"[DEBUG] 일반 '나머지' 요청")
                    
                    completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
                    scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
                    
                    shown_completed = context.get('shown_completed', 3)
                    shown_scheduled = context.get('shown_scheduled', 3)
                    
                    has_more_completed = len(completed_meetings) > shown_completed
                    has_more_scheduled = len(scheduled_meetings) > shown_scheduled
                    
                    # ========== 둘 다 나머지 있음 → 선택 요청 ==========
                    if has_more_completed and has_more_scheduled:
                        answer = "완료된 회의와 예정된 회의 모두 나머지가 있어요! 😊\n\n"
                        answer += "어떤 회의를 더 보시겠어요?\n\n"
                        answer += "💬 \"완료된 나머지\" - 완료된 회의 계속 보기\n"
                        answer += "💬 \"예정된 나머지\" - 예정된 회의 계속 보기"
                        
                        return ChatResponse(
                            answer=answer,
                            source='both_have_more',
                            session_id=session_id
                        )
                    
                    # ========== 완료된 회의만 나머지 있음 → 자동 표시 ==========
                    elif has_more_completed:
                        print(f"[DEBUG] 완료된 회의만 나머지 있음 → 자동 표시")
                        
                        remaining = completed_meetings[shown_completed:]
                        next_batch = remaining[:5]
                        
                        # 시작 번호 계산: 완료 shown + 1
                        start_number = shown_completed + 1
                        
                        answer = "나머지 완료된 회의예요! 📋\n\n"
                        
                        for i, meeting in enumerate(next_batch):
                            actual_number = start_number + i
                            
                            scheduled_at = meeting.get('scheduled_at')
                            if isinstance(scheduled_at, str):
                                scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                            
                            date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                            title = meeting.get('title', '제목 없음')
                            
                            summary = meeting.get('summary', '')
                            if not summary or summary.strip() == '':
                                summary = meeting.get('description', '내용 없음')
                            
                            lines = summary.split('.')[:2]
                            display_text = '. '.join([line.strip() for line in lines if line.strip()])
                            if len(display_text) > 80:
                                display_text = display_text[:80] + "..."
                            
                            answer += f"📌 {actual_number}. {title} {date_str}\n"
                            answer += f"   - {display_text}\n\n"
                        
                        total_shown = shown_completed + len(next_batch)
                        remaining_count = len(completed_meetings) - total_shown
                        
                        if remaining_count > 0:
                            answer += f"💡 완료된 회의가 {remaining_count}개 더 있어요!\n"
                            answer += "\"나머지\" 또는 \"완료된 나머지\" 라고 하시면 계속 볼 수 있어요.\n\n"
                        else:
                            answer += "✅ 완료된 회의를 모두 보여드렸어요!\n\n"
                        
                        answer += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
                        answer += f"예: 번호({start_number}, {start_number+1}), 날짜, 제목 😊"
                        
                        context['shown_completed'] = total_shown
                        save_context(session_id, context)
                        
                        return ChatResponse(
                            answer=answer,
                            source='auto_more_completed',
                            session_id=session_id
                        )
                        
                    # ========== 예정된 회의만 나머지 있음 → 자동 표시 ==========
                    elif has_more_scheduled:
                        print(f"[DEBUG] 예정된 회의만 나머지 있음 → 자동 표시")
                        
                        remaining = scheduled_meetings[shown_scheduled:]
                        next_batch = remaining[:5]
                        
                        # 시작 번호 계산: 완료 전체 + 예정 shown + 1
                        start_number = shown_scheduled + 1

                        answer = "나머지 예정된 회의예요! 📋\n\n"
                        
                        for i, meeting in enumerate(next_batch):
                            actual_number = start_number + i
                            
                            scheduled_at = meeting.get('scheduled_at')
                            if isinstance(scheduled_at, str):
                                scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                            
                            date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
                            title = meeting.get('title', '제목 없음')
                            
                            summary = meeting.get('summary', '')
                            if not summary or summary.strip() == '':
                                summary = meeting.get('description', '내용 없음')
                            
                            lines = summary.split('.')[:2]
                            display_text = '. '.join([line.strip() for line in lines if line.strip()])
                            if len(display_text) > 80:
                                display_text = display_text[:80] + "..."
                            
                            answer += f"📌 {actual_number}. {title} {date_str}\n"
                            answer += f"   - {display_text}\n\n"
                        
                        total_shown = shown_scheduled + len(next_batch)
                        remaining_count = len(scheduled_meetings) - total_shown
                        
                        if remaining_count > 0:
                            answer += f"💡 예정된 회의가 {remaining_count}개 더 있어요!\n"
                            answer += "\"나머지\" 또는 \"예정된 나머지\" 라고 하시면 계속 볼 수 있어요.\n\n"
                        else:
                            answer += "✅ 예정된 회의를 모두 보여드렸어요!\n\n"
                        
                        answer += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
                        answer += f"예: 번호({start_number}, {start_number+1}), 날짜, 제목 😊"
                        
                        context['shown_scheduled'] = total_shown
                        save_context(session_id, context)
                        
                        return ChatResponse(
                            answer=answer,
                            source='auto_more_scheduled',
                            session_id=session_id
                        )
                    
                    # ========== 둘 다 나머지 없음 ==========
                    else:
                        return ChatResponse(
                            answer="✅ 모든 회의를 보여드렸어요! 😊",
                            source='no_more_any',
                            session_id=session_id
                        )
    
        # ========== 1. 상태 키워드 감지 ==========
            if meetings:
                # ========== "나머지" 요청 먼저 체크 ==========
                if any(word in user_query.lower() for word in ['나머지', '더', '더보기', '추가', '계속']):
                    print("[DEBUG] '나머지' 회의 요청 감지")
                    shown_count = context.get('shown_count', 5)
                    remaining = meetings[shown_count:]
                    
                    if remaining:
                        print(f"[DEBUG] 나머지 {len(remaining)}개 회의 반환")
                        response_text = f"나머지 {len(remaining)}개 회의입니다:\n\n"
                        for idx, meeting in enumerate(remaining, start=shown_count+1):
                            response_text += f"{idx}. {meeting['title']}\n"
                            response_text += f"   📅 {meeting['scheduled_at']}\n"
                            if meeting.get('participants'):
                                response_text += f"   👥 {', '.join(meeting['participants'])}\n"
                            response_text += "\n"
                        
                        # shown_count 업데이트
                        context['shown_count'] = len(meetings)
                        save_context(session_id, context)
                        
                        return ChatResponse(
                            answer=response_text,
                            history=request.history + [
                                {"role": "user", "content": user_query},
                                {"role": "assistant", "content": response_text}
                            ],
                            source="remaining_meetings",
                            session_id=session_id
                        )
                    else:
                        no_more = "이미 모든 회의를 보여드렸습니다."
                        return ChatResponse(
                            answer=no_more,
                            history=request.history + [
                                {"role": "user", "content": user_query},
                                {"role": "assistant", "content": no_more}
                            ],
                            source="no_more_meetings",
                            session_id=session_id
                        )
    
                # ========== 1. 상태 키워드 감지 (최우선!) ==========
                status_keywords = ['예정', '완료', '진행중', '취소', 'scheduled', 'completed', 'recording']
                has_status_keyword = any(keyword in user_query for keyword in status_keywords)
                
                if has_status_keyword:
                    print(f"[DEBUG] 상태 키워드 감지 → 새로운 검색: '{user_query}'")
                    delete_context(session_id)
                    # 아래 MySQL 검색으로 진행
                
                # ========== 2. 검색 의도 있는지 체크 ==========
                elif any(keyword in user_query for keyword in ['찾아', '검색', '있어', '있었어', '있나', '뭐', '어떤', '미팅']) or intent == 'meeting_search':
                    print(f"[DEBUG] 검색 의도 감지: '{user_query}'")

                    # 컨텍스트 매칭 점수 계산
                    korean_tokens = re.findall(r'[가-힣]{2,}', user_query)
                    english_tokens = re.findall(r'[A-Za-z0-9]+', user_query)
                    all_tokens = korean_tokens + english_tokens
                    
                    # 불용어/검색어 제거
                    excluded = ['회의', '알려', '알려줘', '보여', '보여줘', '찾아', '검색', '있어', '있었어', '있나', '관련', '뭐가', '어떤']
                    meaningful_tokens = [t for t in all_tokens if len(t) >= 2 and t not in excluded]
                    
                    # 컨텍스트와 매칭 시도
                    best_match_score = 0
                    for meeting in meetings:
                        title = meeting.get('title', '').lower()
                        description = (meeting.get('description') or '').lower()
                        
                        if meaningful_tokens:
                            match_count = sum(1 for token in meaningful_tokens if token in title or token in description)
                            score = match_count / len(meaningful_tokens) if meaningful_tokens else 0
                            best_match_score = max(best_match_score, score)
                    
                    # 매칭 점수가 높으면 (80% 이상) → 선택 시도
                    if best_match_score >= 0.8:
                        print(f"[DEBUG] 검색 의도 있지만 강한 매칭 ({best_match_score:.2f}) → 선택 시도: '{user_query}'")
                        selection_result = handle_selection(user_query, context, request, session_id)
                        
                        # None이면 선택 실패 → 새로운 검색
                        if selection_result is None:
                            print(f"[DEBUG] 선택 실패 → 새로운 검색 시작")
                            delete_context(session_id)
                            # 아래 검색 로직으로 계속 진행
                        else:
                            return selection_result
                    else:
                        # 매칭 점수 낮음 → 새로운 검색
                        print(f"[DEBUG] 검색 의도 감지 + 약한 매칭 ({best_match_score:.2f}) → 새로운 검색: '{user_query}'")
                        delete_context(session_id)
                        # 아래 MySQL 검색으로 진행
                
                # 3. 검색 의도 없음 → 날짜/키워드로 선택 시도
                else:
                    # 날짜 범위 표현이면 새로운 검색
                    date_range_keywords = ['부터', '까지', '사이', '동안', '이후', '이전']
                    if any(keyword in user_query for keyword in date_range_keywords):
                        print(f"[DEBUG] 날짜 범위 검색 감지 → 새로운 검색: '{user_query}'")
                        delete_context(session_id)
                        # 아래 MySQL 검색으로 진행
                    
                    # 단일 날짜 패턴 (범위 아님)
                    elif re.search(r'^\d{1,2}월\s*\d{1,2}일$|^\d{1,2}일$', user_query.strip()):
                        print(f"[DEBUG] 단일 날짜 감지 → 선택 시도: '{user_query}'")
                        selection_result = handle_selection(user_query, context, request, session_id)
                        
                        if selection_result is None:
                            print(f"[DEBUG] 선택 실패 → 새로운 검색 시작")
                            delete_context(session_id)
                            # 아래 검색 로직으로 계속 진행
                        else:
                            return selection_result
                        
                    # 키워드로 컨텍스트 매칭
                    meetings = context.get('meetings', [])
                    user_query_lower = user_query.lower()
                    
                    for meeting in meetings:
                        title = meeting.get('title', '').lower()
                        description = (meeting.get('description') or '').lower()
                        
                        # 입력이 제목/설명에 포함되면 선택
                        if user_query_lower in title or user_query_lower in description:
                            print(f"[DEBUG] 컨텍스트 직접 매칭 → 선택 시도: '{user_query}'")
                            selection_result = handle_selection(user_query, context, request, session_id)
                            
                            if selection_result is None:
                                print(f"[DEBUG] 선택 실패 → 새로운 검색 시작")
                                delete_context(session_id)
                                # 아래 검색 로직으로 계속 진행
                            else:
                                return selection_result
    
                    # 매칭 실패 → 선택 시도
                    print(f"[DEBUG] 컨텍스트 있음 + 검색 의도 없음 → 선택 시도: '{user_query}'")
                    selection_result = handle_selection(user_query, context, request, session_id)

                    if selection_result is None:
                        print(f"[DEBUG] 선택 실패 → 새로운 검색 시작")
                        delete_context(session_id)
                        # 아래 검색 로직으로 계속 진행
                    else:
                        return selection_result

        # === 0단계: 오프토픽 필터링 ===
        # ========== 페이지네이션 키워드 정규식 패턴 ==========
        pagination_keywords = ['나머지', '나머지도', '남은', '남은거', '더', '더보기', '더보여', 
                            '더있어', '더줘', '더알려', '추가', '추가로', '계속', '이어서', 
                            '다음', '다른', '또', '그외', '외', '그밖', '더있나', '더있니', 
                            '또뭐', '또있어', '나머', '남머', '나미', '더보',
                            '줘봐', '줘', '보여줘']

        pagination_patterns = [
            r'나머.*',
            r'남은.*',
            r'더.*[보줘있알려]',
            r'추가.*',
            r'계속|이어서|다음',
            r'또.*[있뭐어]',
            r'그\s*외',
            r'더\s*[보줘]',
            r'줘\s*봐',
            r'보여\s*줘'
        ]

        has_pagination = (any(kw in user_query for kw in pagination_keywords) or 
                        any(re.search(pattern, user_query) for pattern in pagination_patterns))

        if context and context.get('meetings') and has_pagination:
            print(f"[DEBUG] 페이지네이션 키워드 감지 → 오프토픽 체크 스킵")
            # 오프토픽 체크 건너뛰고 아래로 진행
        else:
            # 인사말이나 의미 없는 입력 체크
            simple_greetings = ['ㅎㅇ', 'ㅎㅇㅎㅇ', 'ㅎ2', 'ㄱㄷ', 'ㅂ2', 'ㅂㅂ', 'ㄴㄴ']
            if user_query.strip() in simple_greetings:
                answer = "안녕하세요! 😊 회의록 검색을 도와드릴게요.\n궁금한 회의를 물어보세요!"
                return ChatResponse(
                    answer=answer,
                    history=request.history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": answer}
                    ],
                    source="greeting",
                    session_id=session_id
                )

            if is_off_topic_query(user_query):
                # 예외: 키워드가 있고 "회의" 단어가 포함되어 있으면 회의 검색 시도
                if ('회의' in user_query or '미팅' in user_query) and (keywords and len(keywords) > 0):
                    print(f"[DEBUG] 오프토픽이지만 회의 키워드 있음 → 회의 검색 계속 진행")
                else:
                    print(f"\n🚫 오프토픽 → 회의록 검색 전용 안내")
                    answer = get_off_topic_response()
                    
                    return ChatResponse(
                        answer=answer,
                        history=request.history + [
                            {"role": "user", "content": user_query},
                            {"role": "assistant", "content": answer}
                        ],
                        source="off_topic",
                        session_id=session_id
                    )

        # === Intent 처리가 안 된 경우에만 MySQL 검색 진행 ===
        # (위에서 task_search, participant_search는 이미 처리됨)
        
        # === 1단계: MySQL 검색 ===
        date_info = parse_date_from_query(user_query)
        status = parse_status_from_query(user_query)
        
        from .search import search_meetings_direct

        search_response, meetings = search_meetings_direct(
            user_query, date_info, status, user_job_normalized, selected_meeting_id, user_id
        )

        # ========== 완화 성공 체크 (최우선!) ==========
        if search_response and search_response.startswith("[FALLBACK_SUCCESS]"):
            # 마커 제거
            final_message = search_response.replace("[FALLBACK_SUCCESS]", "")
            print(f"[DEBUG] 단계적 완화 성공 감지 → 컨텍스트 저장 후 반환")
            
            # ========== 컨텍스트 저장 (선택 가능하도록!) ==========
            if meetings and len(meetings) > 0:
                # datetime → str 변환
                meetings_serializable = []
                for meeting in meetings:
                    meeting_copy = {}
                    for key, value in meeting.items():
                        if isinstance(value, datetime):
                            meeting_copy[key] = value.isoformat()
                        else:
                            meeting_copy[key] = value
                    meetings_serializable.append(meeting_copy)
                
                shown_completed, shown_scheduled = calculate_shown_counts(meetings[:5])

                # 컨텍스트 저장
                context_data = {
                    'state': 'awaiting_selection',
                    'shown_completed': shown_completed,
                    'shown_scheduled': shown_scheduled,
                    'meetings': meetings_serializable,
                    'total_count': len(meetings),
                    'shown_completed': 3,
                    'shown_scheduled': 3,
                    'original_query': user_query
                }
                save_context(session_id, context_data)
                print(f"[DEBUG] 완화 성공 → 컨텍스트 저장: {len(meetings)}개")
            
            return ChatResponse(
                answer=final_message,
                history=request.history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": final_message}
                ],
                source="fallback_success",
                session_id=session_id
            )
        
        # MySQL 완전 실패
        if not search_response:
            default_msg = "오류가 발생했어요. 😢\n잠시 후 다시 시도해주세요!"
            
            return ChatResponse(
                answer=default_msg,
                history=request.history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": default_msg}
                ],
                source="error",
                session_id=session_id
            )
        
        # === 2단계: 실패 메시지 체크 ===
        # meetings 리스트로만 판단 (메시지 텍스트 체크 X)
        if not meetings or len(meetings) == 0:
            print(f"⚠️ MySQL 검색 실패 (결과 없음)")
            
            # 컨텍스트가 있었다면 후속 질문으로 처리
            if context and context.get('state') == 'meeting_selected' and selected_meeting_id:
                selected_title = context.get('meeting_title', '해당 회의')
                fallback_msg = f"{selected_title}에 대한 추가 정보를 찾을 수 없었어요.\n다시 질문해주시거나, 다른 회의를 검색해보세요! 😊"
            else:
                fallback_msg = search_response
            
            return ChatResponse(
                answer=fallback_msg,
                history=request.history + [
                    {"role": "user", "content": original_query},
                    {"role": "assistant", "content": fallback_msg}
                ],
                source="not_found",
                session_id=session_id
            )
        
        # === 3단계: 여러 회의 처리 (컨텍스트 저장!) ===
        total = len(meetings)

        if total > 1:
            print(f"[DEBUG] {total}개 회의 발견 → 컨텍스트 저장")
            
            # ========== 상태별 분리 후 재정렬 ==========
            completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
            scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
            
            # 완료 + 예정 순서로 병합
            reordered_meetings = completed_meetings + scheduled_meetings
            
            # datetime → str 변환
            meetings_serializable = []
            for meeting in reordered_meetings:  # 재정렬된 순서로 저장
                meeting_copy = {}
                for key, value in meeting.items():
                    if isinstance(value, datetime):
                        meeting_copy[key] = value.isoformat()
                    else:
                        meeting_copy[key] = value
                meetings_serializable.append(meeting_copy)
            
            # ========== 상태별 분리 포맷팅 적용 (컨텍스트 저장 전에) ==========
            final_answer, shown_completed, shown_scheduled = format_multiple_meetings_short(
                meetings,
                user_query,
                total,
                date_info,
                status
            )

            # 컨텍스트 저장 (실제 표시된 개수로!)
            context = {
                'state': 'awaiting_selection',
                'shown_completed': shown_completed,
                'shown_scheduled': shown_scheduled,
                'meeting_list': meetings_serializable,
                'meetings': meetings_serializable,
                'last_shown_index': min(5, total),
                'offset': min(5, total),
                'shown_count': min(5, total),
                'total_count': total,
                'original_query': user_query
            }

            save_context(session_id, context)
            print(f"[DEBUG] 컨텍스트 저장 완료: {len(meetings_serializable)}개 회의 (shown_completed={shown_completed}, shown_scheduled={shown_scheduled})")

            return ChatResponse(
                answer=final_answer,
                history=request.history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": final_answer}
                ],
                source="multiple_meetings",
                session_id=session_id
            )
        
        # === 3.5단계: 개수 확인 질문 처리 ===
        if total == 1 and context and context.get('state') == 'meeting_selected':
            
            # "그거 하나야?", "더 없어?", "다른 거는?" 같은 질문
            if is_count_question(original_query):
                meeting = meetings[0]
                meeting_title = meeting.get('title', '해당 회의')
                
                # 날짜 범위 표시
                date_range_text = ""
                if date_info and date_info.get('original'):
                    date_range_text = date_info['original']
                else:
                    date_range_text = "해당 기간"
                
                count_response = f"네, {date_range_text}까지 진행한 회의는 {meeting_title} 1개입니다. 😊"
                
                return ChatResponse(
                    answer=count_response,
                    history=request.history + [
                        {"role": "user", "content": original_query},
                        {"role": "assistant", "content": count_response}
                    ],
                    source="count_confirmation",
                    session_id=session_id
                )
        
        # === 3.6단계: 확인 질문 처리 ===
        if context and context.get('state') == 'meeting_selected' and is_confirmation:
            meeting_title = context.get('meeting_title', '해당 회의')
            
            # 날짜 정보 추출
            if meetings and len(meetings) >= 1:
                scheduled_at = meetings[0].get('scheduled_at', '')
                try:
                    if isinstance(scheduled_at, str):
                        dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                    else:
                        dt = scheduled_at
                    date_str = dt.strftime('%Y년 %m월 %d일')
                except:
                    date_str = "해당 날짜"
            else:
                date_str = "해당 날짜"
            
            confirmation_response = f"네, 맞습니다! 📌\n\n가장 최근 {meeting_title}는 {date_str}에 진행되었습니다. 😊"
            
            return ChatResponse(
                answer=confirmation_response,
                history=request.history + [
                    {"role": "user", "content": original_query},
                    {"role": "assistant", "content": confirmation_response}
                ],
                source="confirmation",
                session_id=session_id
            )
        
        # === 4단계: 단일 회의 (이미 search.py에서 템플릿 적용됨) ===
        print(f"\n✅ 단일 회의 발견")
        final_answer = search_response  # search.py에서 이미 페르소나 템플릿 적용됨

        # 단일 회의도 컨텍스트 저장 (meeting_id 저장!)
        if meetings and len(meetings) == 1:
            meeting = meetings[0]
            meeting_id = meeting.get('id')
            
            context = {
                'state': 'meeting_selected',
                'selected_meeting_id': meeting_id,
                'meeting_title': meeting.get('title', ''),
                'original_query': user_query,
                'search_status': status
            }
            save_context(session_id, context)
            print(f"[DEBUG] 단일 회의 컨텍스트 저장: meeting_id={meeting_id}")

        return ChatResponse(
            answer=final_answer,
            history=request.history + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_answer}
            ],
            source="single_meeting",
            session_id=session_id
        )
                
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = "서버 오류가 발생했어요. 잠시 후 다시 시도해주세요. 🙏"
        
        return ChatResponse(
            answer=error_msg,
            history=request.history + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": error_msg}
            ],
            source="error"
        )
    
def is_obvious_pattern(user_query: str) -> bool:
    obvious_patterns = [
        user_query.strip().isdigit(),
        bool(re.match(r'^\d{1,2}월\s?\d{1,2}일$', user_query.strip())),
        (len(user_query) > 8 and 
         ('회의' in user_query or '미팅' in user_query) and 
         not any(w in user_query for w in ['?', '뭐', '어떤', '있어', '저', '그', '이']) and
         not ('에서' in user_query and len(user_query) < 15)),  # ← "회의에서" 같은 짧은 질문 제외
    ]
    return any(obvious_patterns)

def needs_llm_analysis(user_query: str, context: dict) -> bool:
    """
    LLM 분석이 필요한지 확인 (최소화)
    """
    # ========== 0. awaiting_selection 상태면 LLM 안 씀 ==========
    if context and context.get('state') == 'awaiting_selection':
        return False
    
    # 1. 명확한 오타가 있으면 LLM 필요
    if any(char in user_query for char in ['ㅅ', 'ㅈ', 'ㄱ', 'ㄴ', 'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ']):
        return True
    
    # 2. 컨텍스트 있고 대명사만 쓴 짧은 질문 (5자 이하)
    if context and context.get('state') == 'meeting_selected' and len(user_query) <= 5:
        pronouns = ['그거', '저거', '이거', '거기', '여기']
        if any(p in user_query for p in pronouns):
            return True
    
    # 3. 그 외는 LLM 안 씀
    return False