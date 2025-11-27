"""
컨텍스트 기반 선택 처리
- 번호 선택
- 날짜 선택
- 키워드 선택
"""
import re
import logging
from datetime import datetime
from .models import ChatRequest, ChatResponse
from .formatting import format_single_meeting, format_single_meeting_with_persona
from .context import save_context, delete_context
from .config import ENABLE_PERSONA

logger = logging.getLogger(__name__)

# ============================================================
# 선택 처리
# ============================================================

def handle_selection(user_input: str, context: dict, 
                    request: ChatRequest, session_id: str) -> ChatResponse:
    """사용자가 회의를 선택했을 때 처리 (번호, 제목, 날짜, 키워드)"""
    
    meetings = context.get('meetings', [])
    if not meetings:
        return ChatResponse(
            answer="선택할 회의가 없어요. 다시 검색해주세요! 😊",
            history=request.history,
            source="no_meetings",
            session_id=session_id
        )
    
    user_input_lower = user_input.lower().strip()
    selected_meeting = None
    selection_method = None
    matched_meetings = []
    
    # 1. 숫자로 선택
    number_pattern = r'(?:(완료|예정)\s*)?(\d+)'
    number_match = re.match(number_pattern, user_input.strip())

    if number_match:
        status_prefix = number_match.group(1)
        selected_number = int(number_match.group(2))
        
        shown_completed = context.get('shown_completed', 0)
        shown_scheduled = context.get('shown_scheduled', 0)
        
        print(f"[DEBUG] 번호 선택 체크: number={selected_number}")
        
        # ========== 먼저 변수 정의! ==========
        completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
        scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
        
        print(f"[DEBUG] 완료={len(completed_meetings)}개, 예정={len(scheduled_meetings)}개")
        
        # ========== 상태별 분리 표시 확인 ==========
        is_status_separated = (shown_completed > 0 or shown_scheduled > 0)
        
        if is_status_separated:
            # 먼저 변수 정의
            completed_meetings = [m for m in meetings if m.get('status') == 'COMPLETED']
            scheduled_meetings = [m for m in meetings if m.get('status') == 'SCHEDULED']
            
            # 그 다음 로그 출력
            print(f"[DEBUG] 번호 선택 체크: number={selected_number}")
            print(f"[DEBUG] 완료={len(completed_meetings)}개, 예정={len(scheduled_meetings)}개")
            
            if status_prefix == '완료':
                if 1 <= selected_number <= shown_completed:
                    selected_meeting = completed_meetings[selected_number - 1]
                    selection_method = f"완료 {selected_number}번"
                else:
                    return ChatResponse(
                        answer=f"❌ 완료 {selected_number}번은 없어요!",
                        source="invalid_number",
                        session_id=session_id
                    )
            
            elif status_prefix == '예정':
                if 1 <= selected_number <= shown_scheduled:
                    selected_meeting = scheduled_meetings[selected_number - 1]
                    selection_method = f"예정 {selected_number}번"
                else:
                    return ChatResponse(
                        answer=f"❌ 예정 {selected_number}번은 없어요!",
                        source="invalid_number",
                        session_id=session_id
                    )
            
            else:
                # 숫자만 입력
                has_completed = (completed_meetings and 
                            1 <= selected_number <= len(completed_meetings))
                has_scheduled = (scheduled_meetings and 
                            1 <= selected_number <= len(scheduled_meetings))
                
                print(f"[DEBUG] has_completed={has_completed}, has_scheduled={has_scheduled}")
                
                if has_completed and has_scheduled:
                    # 모호함
                    context['last_source'] = 'ambiguous_number'
                    context['last_ambiguous_number'] = selected_number
                    save_context(session_id, context)

                    return ChatResponse(
                        answer=f"완료된 회의와 예정된 회의 모두 {selected_number}번이 있어요! 🤔\n\n어떤 회의를 보시겠어요?\n\n💬 \"완료 {selected_number}\"\n💬 \"예정 {selected_number}\"",
                        source="ambiguous_number",
                        session_id=session_id
                    )
                
                elif has_completed:
                    selected_meeting = completed_meetings[selected_number - 1]
                    selection_method = f"{selected_number}번 (완료)"
                
                elif has_scheduled:
                    selected_meeting = scheduled_meetings[selected_number - 1]
                    selection_method = f"{selected_number}번 (예정)"
                
                else:
                    return ChatResponse(
                        answer=f"❌ {selected_number}번은 없어요!",
                        source="invalid_number",
                        session_id=session_id
                    )
        
        else:
            # 일반 다중 회의 - 연속 번호
            if 1 <= selected_number <= len(meetings):
                selected_meeting = meetings[selected_number - 1]
                selection_method = f"{selected_number}번"
                print(f"[DEBUG] 번호 선택: {selected_number}번")
            else:
                return ChatResponse(
                    answer=f"❌ {selected_number}번은 없어요!\n1번부터 {len(meetings)}번까지 선택할 수 있어요. 😊",
                    source="invalid_number",
                    session_id=session_id
                )
    
    # 2. 날짜로 선택 (예: "10월 20일", "20일", "20일꺼")
    if not selected_meeting:
        # "X월 Y일" 패턴
        date_match = re.search(r'(\d{1,2})월\s*(\d{1,2})일', user_input)
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            
            # 해당 날짜의 모든 회의 찾기
            matched_meetings = []
            for i, meeting in enumerate(meetings):
                scheduled_at = meeting.get('scheduled_at')
                if isinstance(scheduled_at, str):
                    scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                
                if scheduled_at and scheduled_at.month == month and scheduled_at.day == day:
                    matched_meetings.append((i, meeting))
            
            # 매칭 결과 처리
            if len(matched_meetings) == 1:
                # 1개만 매칭 → 바로 선택
                selected_meeting = matched_meetings[0][1]
                selection_method = f"{month}월 {day}일"
                print(f"[DEBUG] 날짜 선택: {month}월 {day}일 (1개 매칭)")
            elif len(matched_meetings) > 1:
                # 여러 개 매칭 → 연도가 다른 경우!
                print(f"[DEBUG] 날짜 선택: {month}월 {day}일 (여러 개 매칭: {len(matched_meetings)}개)")
                
                response_msg = f"{month}월 {day}일에 회의가 {len(matched_meetings)}개 있어요! 🗓️\n"
                response_msg += "연도가 다른 것 같아요. 확인해주세요!\n\n"
                
                for idx, (original_idx, meeting) in enumerate(matched_meetings, 1):
                    title = meeting.get('title', '제목 없음')
                    scheduled_at = meeting.get('scheduled_at')
                    if isinstance(scheduled_at, str):
                        scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                    
                    date_str = scheduled_at.strftime('%Y년 %m월 %d일') if scheduled_at else '날짜 정보 없음'
                    description = meeting.get('description', '')
                    if len(description) > 40:
                        description = description[:40] + "..."
                    
                    emoji = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'][idx - 1] if idx <= 10 else f"{idx}️⃣"
                    response_msg += f"{emoji} {title} ({date_str})\n"
                    response_msg += f"   - {description}\n\n"
                
                response_msg += "어떤 회의를 보시겠어요?\n"
                response_msg += "예: 번호(1, 2), 연도 포함 날짜(2025년 10월 20일) 😊"
                
                # 매칭된 회의들만 컨텍스트에 저장 (다시 선택하도록)
                matched_meetings_list = [m for _, m in matched_meetings]
                context_data = {
                    'state': 'awaiting_selection',
                    'meetings': matched_meetings_list,
                    'original_query': user_input
                }
                save_context(session_id, context_data)
                
                return ChatResponse(
                    answer=response_msg,
                    history=request.history + [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response_msg}
                    ],
                    source="multiple_date_matches",
                    session_id=session_id
                )
        
        # "X일" 패턴 (예: "20일", "20일꺼")
        if not selected_meeting:
            day_match = re.search(r'(\d{1,2})일', user_input)
            if day_match:
                day = int(day_match.group(1))
                
                # 해당 날짜의 모든 회의 찾기
                matched_meetings = []
                for i, meeting in enumerate(meetings):
                    scheduled_at = meeting.get('scheduled_at')
                    if isinstance(scheduled_at, str):
                        scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                    
                    if scheduled_at and scheduled_at.day == day:
                        matched_meetings.append((i, meeting))
                
                # 매칭 결과 처리
                if len(matched_meetings) == 1:
                    # 1개만 매칭 → 바로 선택
                    selected_meeting = matched_meetings[0][1]
                    selection_method = f"{day}일"
                    print(f"[DEBUG] 날짜 선택: {day}일 (1개 매칭)")
                elif len(matched_meetings) > 1:
                    # 여러 개 매칭 → 목록 보여주고 다시 선택
                    print(f"[DEBUG] 날짜 선택: {day}일 (여러 개 매칭: {len(matched_meetings)}개)")
                    
                    response_msg = f"{day}일에 회의가 {len(matched_meetings)}개 있어요! 🗓️\n\n"
                    
                    for idx, (original_idx, meeting) in enumerate(matched_meetings, 1):
                        title = meeting.get('title', '제목 없음')
                        scheduled_at = meeting.get('scheduled_at')
                        if isinstance(scheduled_at, str):
                            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                        
                        date_str = scheduled_at.strftime('%Y년 %m월 %d일') if scheduled_at else '날짜 정보 없음'
                        description = meeting.get('description', '')
                        if len(description) > 40:
                            description = description[:40] + "..."
                        
                        emoji = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'][idx - 1] if idx <= 10 else f"{idx}️⃣"
                        response_msg += f"{emoji} {title} ({date_str})\n"
                        response_msg += f"   - {description}\n\n"
                    
                    response_msg += "어떤 회의를 보시겠어요?\n"
                    response_msg += "예: 번호(1, 2) 😊"
                    
                    # 매칭된 회의들만 컨텍스트에 저장 (다시 선택하도록)
                    matched_meetings_list = [m for _, m in matched_meetings]
                    context_data = {
                        'state': 'awaiting_selection',
                        'meetings': matched_meetings_list,
                        'original_query': user_input
                    }
                    save_context(session_id, context_data)
                    
                    return ChatResponse(
                        answer=response_msg,
                        history=request.history + [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": response_msg}
                        ],
                        source="multiple_date_matches",
                        session_id=session_id
                    )
    
    # 3. 제목/키워드로 선택 (예: "디자인", "디자인 시스템", "AI회의")
    if not selected_meeting:
        import difflib
        
        # 회의 제목과의 유사도 계산
        matched_meetings = []  # (meeting, score) 튜플 리스트
        user_input_lower = user_input.lower().strip()
        
        # ========== "회의" 제거 함수 ==========
        def remove_meeting_word(text):
            return re.sub(r'회의|미팅', '', text).strip()
        
        # ========== 검색 유도 불용어 체크 (기존 그대로) ==========
        search_stopwords = ['최근', '이번주', '지난주', '회의', '미팅', '뭐', '어떤', '있어', '있었어', '있나', '찾아', '검색', '더', '나머지']
        
        tokens = user_input_lower.split()
        search_word_count = len([t for t in tokens if t in search_stopwords])
        
        if tokens and search_word_count / len(tokens) > 0.6:
            print(f"[DEBUG] 키워드 선택 스킵: 검색 유도 단어가 대부분 ({search_word_count}/{len(tokens)})")
            pass
        
        # ========== 키워드 매칭 로직 (수정) ==========
        else:
            # "회의" 제거 후 비교
            user_query_clean = remove_meeting_word(user_input_lower)
            
            for i, meeting in enumerate(meetings):
                title_original = meeting.get('title', '').lower()
                title_clean = remove_meeting_word(title_original)
                
                # 1. 부분 문자열 포함 체크 (정확 매칭)
                if user_query_clean in title_clean or title_clean in user_query_clean:
                    matched_meetings.append((meeting, 1.0))  # 100% 매칭
                    print(f"  - '{meeting.get('title')}' 부분 매칭 (100%)")
                    continue
                
                # 2. difflib 유사도 계산 (기존 로직)
                ratio = difflib.SequenceMatcher(None, user_query_clean, title_clean).ratio()
                
                print(f"  - '{meeting.get('title')}' 유사도: {ratio:.2%} ('{user_query_clean}' vs '{title_clean}')")
                
                # 70% 이상 유사하면 매칭
                if ratio >= 0.7:
                    matched_meetings.append((meeting, ratio))

        # 매칭 결과 처리
        if len(matched_meetings) == 0:
            # 매칭 없음
            pass  # 아래 invalid_selection으로
        
        elif len(matched_meetings) == 1:
            # 1개만 → 바로 선택
            selected_meeting = matched_meetings[0][0]
            selection_method = "키워드"
            print(f"[DEBUG] 키워드 선택: '{user_input}' (점수: {matched_meetings[0][1]:.2f}, 1개 매칭)")
            
        else: # matched_meetings > 1 인 경우만 실행
            # 여러 개 → 점수 순 정렬 후 목록 표시
            matched_meetings.sort(key=lambda x: x[1], reverse=True)
            print(f"[DEBUG] 키워드 선택: '{user_input}' (여러 개 매칭: {len(matched_meetings)}개)")
            
            response_msg = f"'{user_input}' 관련 회의가 {len(matched_meetings)}개 있어요! 📋\n\n"
            
            for idx, (meeting, score) in enumerate(matched_meetings[:10], 1):  # 최대 10개
                title = meeting.get('title', '제목 없음')
                scheduled_at = meeting.get('scheduled_at')
                if isinstance(scheduled_at, str):
                    scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                
                date_str = scheduled_at.strftime('%Y년 %m월 %d일') if scheduled_at else '날짜 정보 없음'
                description = meeting.get('description', '')
                if len(description) > 40:
                    description = description[:40] + "..."
                
                emoji = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'][idx - 1] if idx <= 10 else f"{idx}️⃣"
                response_msg += f"{emoji} {title} ({date_str})\n"
                response_msg += f"   - {description}\n\n"
            
            if len(matched_meetings) > 10:
                response_msg += f"💡 나머지 {len(matched_meetings) - 10}개 회의도 있어요!\n\n"
            
            response_msg += "어떤 회의를 보시겠어요?\n"
            response_msg += "예: 번호(1, 2) 😊"
            
            # 매칭된 회의들만 컨텍스트에 저장 (다시 선택하도록)
            matched_meetings_list = [m for m, _ in matched_meetings[:10]]
            context_data = {
                'state': 'awaiting_selection',
                'meetings': matched_meetings_list,
                'original_query': user_input
            }
            save_context(session_id, context_data)
            
            return ChatResponse(
                answer=response_msg,
                history=request.history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response_msg}
                ],
                source="multiple_keyword_matches",
                session_id=session_id
            )
    
    # 선택된 회의가 없으면 → 새로운 검색으로 처리
    if not selected_meeting:
        print(f"[DEBUG] 선택 실패 (유사도 70% 미만) → 새로운 검색으로 전환")
        return None
    
    # 선택된 회의 정보 포맷
    print(f"[DEBUG] 선택 완료 ({selection_method}): {selected_meeting['title']}")

    # DB에서 전체 정보 다시 조회 (meeting_result, participants 포함)
    from .database import get_db_connection
    with get_db_connection() as conn:
        if conn:
            cursor = conn.cursor()
            
            # meeting + meeting_result JOIN
            cursor.execute("""
                SELECT m.*, mr.summary, mr.agenda, mr.purpose, 
                    mr.importance_level, mr.importance_reason
                FROM meeting m
                LEFT JOIN meeting_result mr ON m.id = mr.meeting_id
                WHERE m.id = %s
            """, (selected_meeting['id'],))
            full_meeting = cursor.fetchone()
            
            if full_meeting:
                # participants 조회
                cursor.execute("""
                    SELECT name FROM participant WHERE meeting_id = %s
                """, (selected_meeting['id'],))
                participants = cursor.fetchall()
                full_meeting['participants'] = [p['name'] for p in participants]
                
                selected_meeting = full_meeting
                
    # ========== Phase 2-A: 페르소나 템플릿 적용 ==========
    user_job_raw = getattr(request, 'user_job', 'NONE')
    if not user_job_raw or user_job_raw == 'NONE':
        user_job_raw = getattr(request, 'job', 'NONE')

    # 정규화 (대문자 변환)
    user_job = user_job_raw.upper() if user_job_raw else 'NONE'

    # 유효한 직무만 허용
    valid_jobs = ['NONE', 'PROJECT_MANAGER', 'FRONTEND_DEVELOPER', 
                'BACKEND_DEVELOPER', 'DATABASE_ADMINISTRATOR', 'SECURITY_DEVELOPER']
    if user_job not in valid_jobs:
        user_job = 'NONE'

    print(f"[DEBUG] Phase 2-A: user_job (원본: {user_job_raw}, 정규화: {user_job})")

    if ENABLE_PERSONA and user_job != 'NONE':
        meeting_info = format_single_meeting_with_persona(selected_meeting, user_job)
        print(f"[DEBUG] Phase 2-A: {user_job}용 템플릿 적용 (선택)")
    else:
        meeting_info = format_single_meeting(selected_meeting)
        print(f"[DEBUG] 기본 템플릿 적용 (선택)")
        
    # 선택 완료 후 - 컨텍스트 업데이트 (회의 리스트 유지!)
    new_context = {
        'state': 'meeting_selected',
        'selected_meeting_id': selected_meeting['id'],
        'meeting_title': selected_meeting.get('title', ''),
        'selected_meeting': selected_meeting,
        'meetings': context.get('meetings', []),  # ← 회의 리스트 유지!
        'shown_completed': context.get('shown_completed', 3),
        'shown_scheduled': context.get('shown_scheduled', 3),
        'original_query': context.get('original_query', '')
    }
    save_context(session_id, new_context)
    print(f"[DEBUG] 컨텍스트 업데이트 (회의 리스트 유지): {len(context.get('meetings', []))}개")
        
    return ChatResponse(
        answer=meeting_info,
        history=request.history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": meeting_info}
        ],
        source="selected_meeting",
        session_id=session_id
    )