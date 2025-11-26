"""
템플릿 포맷팅 함수들
- 단일 회의 포맷
- 여러 회의 목록 포맷
- Phase 2-A: 직업별 페르소나 템플릿
"""
from datetime import datetime
import re

def format_importance(level: str, reason: str = None) -> str:
    """중요도 포맷팅"""
    if not level:
        return '정보 없음'
    level_map = {
        'HIGH': '🔴 높음',
        'MEDIUM': '🟡 보통', 
        'LOW': '🟢 낮음'
    }
    level_str = level_map.get(level, level)
    if reason:
        return f"{level_str} - {reason}"
    return level_str

# ============================================================
# 기본 포맷팅
# ============================================================

def format_date(dt) -> str:
    """날짜를 'YYYY년 MM월 DD일' 형식으로"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    return dt.strftime('%Y년 %m월 %d일') if dt else '날짜 정보 없음'

def format_single_meeting(meeting: dict) -> str:
    """단일 회의 기본 템플릿"""
    scheduled_at = meeting.get('scheduled_at')
    if isinstance(scheduled_at, str):
        scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
    
    date_str = scheduled_at.strftime('%Y년 %m월 %d일') if scheduled_at else '날짜 정보 없음'
    
    # 추가 정보 추출
    participants = meeting.get('participants', [])
    participants_str = ', '.join(participants) if participants else '정보 없음'
    purpose = meeting.get('purpose') or '정보 없음'
    agenda = meeting.get('agenda') or '정보 없음'
    importance_str = format_importance(meeting.get('importance_level'), meeting.get('importance_reason'))
    
    return f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}
👥 참가자: {participants_str}

🎯 회의 목적:
{purpose}

📋 안건:
{agenda}

⭐ 중요도: {importance_str}

💡 요약:
{meeting.get('summary') or '요약 없음'}"""


def format_multiple_meetings_short(results: list, user_query: str, total: int = None, date_info: dict = None, status: str = None) -> str:
    """여러 회의를 상태별로 분리하여 간단히 포맷팅 (완료 3개 + 예정 3개)"""
    from datetime import datetime
    
    # ========== 상태별 분리 ==========
    completed = [m for m in results if m.get('status') == 'COMPLETED']
    scheduled = [m for m in results if m.get('status') == 'SCHEDULED']
    
    print(f"[DEBUG] 상태별 분리: 완료 {len(completed)}개, 예정 {len(scheduled)}개")
    
    # 조건 텍스트 생성
    condition_text = ""
    if date_info and date_info.get('original'):
        condition_text = f"{date_info['original']} "
    
    response = f"네, {condition_text}회의로는 다음과 같은 것들이 있어요! 📋\n\n"
    
    idx = 1  # 연속 번호 시작
    
    # ========== 완료와 예정 개수 결정 ==========
    if len(completed) > 0 and len(scheduled) > 0:
        # 둘 다 있으면: 완료 3개 + 예정 3개
        completed_limit = 3
        scheduled_limit = 3
    elif len(completed) > 0:
        # 완료만 있으면: 완료 5개
        completed_limit = 5
        scheduled_limit = 0
    else:
        # 예정만 있으면: 예정 5개
        completed_limit = 0
        scheduled_limit = 5

    # ========== 완료된 회의 ==========
    if completed:
        response += "✅ 완료된 회의:\n\n"
        for i, meeting in enumerate(completed[:completed_limit], 1):
            scheduled_at = meeting.get('scheduled_at')
            if isinstance(scheduled_at, str):
                scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
            date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
            
            title = meeting.get('title', '제목 없음')
            
            summary = meeting.get('summary') or ''
            if not summary.strip():
                summary = meeting.get('description') or '내용 없음'
                
            lines = summary.split('.')[:2]
            display_text = '. '.join([line.strip() for line in lines if line.strip()])
            
            response += f"📌{i}. {title} {date_str}\n"
            response += f"   - {display_text}\n\n"
        
        # 완료된 회의 나머지 개수
        remaining_completed = len(completed) - completed_limit
        if remaining_completed > 0:
            response += f"💡 완료된 회의가 {remaining_completed}개 더 있어요!\n\n"

    # ========== 예정된 회의 ==========
    if scheduled:
        response += "📅 예정된 회의:\n\n"
        for i, meeting in enumerate(scheduled[:scheduled_limit], 1):
            scheduled_at = meeting.get('scheduled_at')
            if isinstance(scheduled_at, str):
                scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
            date_str = scheduled_at.strftime('(%Y년 %m월 %d일)') if scheduled_at else ''
            
            # 날짜 지남 체크
            is_past = False
            if scheduled_at and scheduled_at.date() < datetime.now().date():
                is_past = True
                date_str = date_str.replace(')', ' - 일정 지남)')
            
            title = meeting.get('title', '제목 없음')
            
            summary = meeting.get('summary') or ''
            if not summary.strip():
                summary = meeting.get('description') or '내용 없음'

            lines = summary.split('.')[:2]
            display_text = '. '.join([line.strip() for line in lines if line.strip()])
            
            emoji = '⚠️' if is_past else '📌'
            response += f"{emoji}{i}. {title} {date_str}\n"
            response += f"   - {display_text}\n\n"
        
        # 예정된 회의 나머지 개수
        remaining_scheduled = len(scheduled) - scheduled_limit
        if remaining_scheduled > 0:
            response += f"💡 예정된 회의가 {remaining_scheduled}개 더 있어요!\n\n"
            
    # ========== 마지막 안내 ==========
    # 나머지가 실제로 있을 때만 "나머지 보여줘" 멘트 표시
    remaining_completed = len(completed) - completed_limit
    remaining_scheduled = len(scheduled) - scheduled_limit

    if remaining_completed > 0 or remaining_scheduled > 0:
        response += "💬 \"나머지 보여줘\" 라고 하시면 계속 볼 수 있어요!\n\n"

    response += "더 자세히 알고 싶은 회의를 선택해주세요!\n"
    response += "예: 번호(완료 1, 예정 2), 제목(디자인 회의) 😊"

    return response, completed_limit, scheduled_limit

# ============================================================
# Phase 2-A: 페르소나 템플릿 (5개 직업군만)
# ============================================================
# DB의 실제 job: 'NONE', 'PROJECT_MANAGER', 'FRONTEND_DEVELOPER', 
#                'BACKEND_DEVELOPER', 'DATABASE_ADMINISTRATOR', 'SECURITY_DEVELOPER'

def extract_pm_tech_stack(meeting: dict) -> list:
    """프로젝트 관리 도구 추출 (PM용)"""
    tech_keywords = ['Jira', 'Asana', 'Trello', 'Notion', 'Confluence', 
                     'Monday', 'ClickUp', 'Slack', 'Teams', 'GitHub', 
                     'GitLab', 'Figma', 'Miro']
    tech_stack = []
    
    description = meeting.get('description', '')
    summary = meeting.get('summary', '')
    
    for keyword in tech_keywords:
        # 단어 경계 사용 (\b)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, description.lower()) or re.search(pattern, summary.lower()):
            tech_stack.append(keyword)
    
    return tech_stack

def extract_frontend_tech_stack(meeting: dict) -> list:
    """프론트엔드 기술 스택 추출"""
    tech_keywords = ['React', 'Vue', 'Angular', 'Next.js', 'Nuxt.js', 
                     'TypeScript', 'JavaScript', 'Svelte', 'Tailwind', 
                     'CSS', 'HTML', 'Redux', 'Zustand', 'Webpack', 'Vite']
    tech_stack = []
    
    description = meeting.get('description', '')
    summary = meeting.get('summary', '')
    
    for keyword in tech_keywords:
        # 단어 경계 사용 (\b)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, description.lower()) or re.search(pattern, summary.lower()):
            tech_stack.append(keyword)
    
    return tech_stack

def extract_backend_tech_stack(meeting: dict) -> list:
    """백엔드 기술 스택 추출"""
    tech_keywords = ['Spring Boot', 'Spring', 'Node.js', 'Express', 'FastAPI', 
                     'Django', 'Flask', 'NestJS', 'Java', 'Python', 
                     'Go', 'Rust', 'Kotlin', 'REST', 'GraphQL', 
                     'gRPC', 'Docker', 'Kubernetes', 'AWS', 'Azure']
    tech_stack = []
    
    description = meeting.get('description', '')
    summary = meeting.get('summary', '')
    
    for keyword in tech_keywords:
        # 단어 경계 사용 (\b)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, description.lower()) or re.search(pattern, summary.lower()):
            tech_stack.append(keyword)
    
    return tech_stack

def extract_dba_tech_stack(meeting: dict) -> list:
    """데이터베이스 기술 스택 추출"""
    tech_keywords = ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 
                     'SQL Server', 'MariaDB', 'Elasticsearch', 'Cassandra', 
                     'DynamoDB', 'SQLite', 'Neo4j', 'Snowflake']
    tech_stack = []
    
    description = meeting.get('description', '')
    summary = meeting.get('summary', '')
    
    for keyword in tech_keywords:
        # 단어 경계 사용 (\b)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, description.lower()) or re.search(pattern, summary.lower()):
            tech_stack.append(keyword)
    
    return tech_stack

def extract_security_tech_stack(meeting: dict) -> list:
    """보안 도구/기술 추출"""
    tech_keywords = ['SSL', 'TLS', 'OAuth', 'JWT', 'SAML', 
                     'Firewall', 'WAF', 'IDS', 'IPS', 'VPN', 
                     'Nessus', 'Burp Suite', 'Wireshark', 'Metasploit', 
                     'OpenSSL', 'Snort']
    tech_stack = []
    
    description = meeting.get('description', '')
    summary = meeting.get('summary', '')
    
    for keyword in tech_keywords:
        # 단어 경계 사용 (\b)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, description.lower()) or re.search(pattern, summary.lower()):
            tech_stack.append(keyword)
    
    return tech_stack

def extract_simple_info(meeting: dict, keywords: list) -> str:
    """간단한 정보 추출"""
    description = meeting.get('description') or ''
    lines = description.split('\n')
    
    results = []
    for line in lines:
        if any(keyword in line for keyword in keywords):
            results.append(f"{line.strip()}")
    
    return '\n'.join(results) if results else '   없음'

# ============================================================
def format_project_manager_meeting(meeting: dict) -> str:
    """PROJECT_MANAGER용 회의 템플릿"""
    scheduled_at = meeting.get('scheduled_at', '')
    try:
        if isinstance(scheduled_at, str):
            dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        else:
            dt = scheduled_at
        date_str = dt.strftime('%Y년 %m월 %d일')
    except:
        date_str = str(scheduled_at)[:10] if scheduled_at else '날짜 없음'
    
    # 추가 정보 추출
    participants = meeting.get('participants', [])
    participants_str = ', '.join(participants) if participants else '정보 없음'
    purpose = meeting.get('purpose') or '정보 없음'
    agenda = meeting.get('agenda') or '정보 없음'
    importance_str = format_importance(meeting.get('importance_level'), meeting.get('importance_reason'))
    
    summary = meeting.get('summary', '')
    goal = summary.split('.')[0].strip() if summary else '없음'
    
    tech_stack = extract_pm_tech_stack(meeting)
    planning_info = extract_simple_info(meeting, ['기획', '전략', '로드맵', '목표', '계획', '일정', '마일스톤'])
    
    template = f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}
👥 참가자: {participants_str}
📊 사용 도구: {', '.join(tech_stack) if tech_stack else '정보 없음'}

🎯 회의 목적:
{purpose}

📋 안건:
{agenda}

⭐ 중요도: {importance_str}

💡 요약:
{meeting.get('summary') or '없음'}

📊 PM 주요사항:
{planning_info}
"""
    return template

# ============================================================
def format_frontend_developer_meeting(meeting: dict) -> str:
    """FRONTEND_DEVELOPER용 회의 템플릿"""
    scheduled_at = meeting.get('scheduled_at', '')
    try:
        if isinstance(scheduled_at, str):
            dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        else:
            dt = scheduled_at
        date_str = dt.strftime('%Y년 %m월 %d일')
    except:
        date_str = str(scheduled_at)[:10] if scheduled_at else '날짜 없음'
    
    # 추가 정보 추출
    participants = meeting.get('participants', [])
    participants_str = ', '.join(participants) if participants else '정보 없음'
    purpose = meeting.get('purpose') or '정보 없음'
    agenda = meeting.get('agenda') or '정보 없음'
    importance_str = format_importance(meeting.get('importance_level'), meeting.get('importance_reason'))
    
    tech_stack = extract_frontend_tech_stack(meeting)
    ui_info = extract_simple_info(meeting, ['ui', 'ux', '화면', '컴포넌트', 'react', 'vue', 'frontend', '프론트'])
    
    template = f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}
👥 참가자: {participants_str}
💻 기술 스택: {', '.join(tech_stack) if tech_stack else '정보 없음'}

🎯 회의 목적:
{purpose}

📋 안건:
{agenda}

⭐ 중요도: {importance_str}

💡 요약:
{meeting.get('summary') or '없음'}

🎨 UI/UX 작업사항:
{ui_info}
"""
    return template

# ============================================================
def format_backend_developer_meeting(meeting: dict) -> str:
    """BACKEND_DEVELOPER용 회의 템플릿"""
    tech_stack = extract_backend_tech_stack(meeting)
    
    scheduled_at = meeting.get('scheduled_at', '')
    try:
        if isinstance(scheduled_at, str):
            dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        else:
            dt = scheduled_at
        date_str = dt.strftime('%Y년 %m월 %d일')
    except:
        date_str = str(scheduled_at)[:10] if scheduled_at else '날짜 없음'
    
    # 추가 정보 추출
    participants = meeting.get('participants', [])
    participants_str = ', '.join(participants) if participants else '정보 없음'
    purpose = meeting.get('purpose') or '정보 없음'
    agenda = meeting.get('agenda') or '정보 없음'
    importance_str = format_importance(meeting.get('importance_level'), meeting.get('importance_reason'))
    
    backend_tasks = extract_simple_info(meeting, ['api', '서버', '백엔드', 'backend', '데이터베이스', '배포'])
    
    template = f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}
👥 참가자: {participants_str}
💻 기술 스택: {', '.join(tech_stack) if tech_stack else '정보 없음'}

🎯 회의 목적:
{purpose}

📋 안건:
{agenda}

⭐ 중요도: {importance_str}

💡 요약:
{meeting.get('summary') or '없음'}

🔧 백엔드 작업사항:
{backend_tasks}
"""
    return template

# ============================================================
def format_database_administrator_meeting(meeting: dict) -> str:
    """DATABASE_ADMINISTRATOR용 회의 템플릿"""
    scheduled_at = meeting.get('scheduled_at', '')
    try:
        if isinstance(scheduled_at, str):
            dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        else:
            dt = scheduled_at
        date_str = dt.strftime('%Y년 %m월 %d일')
    except:
        date_str = str(scheduled_at)[:10] if scheduled_at else '날짜 없음'
    
    # 추가 정보 추출
    participants = meeting.get('participants', [])
    participants_str = ', '.join(participants) if participants else '정보 없음'
    purpose = meeting.get('purpose') or '정보 없음'
    agenda = meeting.get('agenda') or '정보 없음'
    importance_str = format_importance(meeting.get('importance_level'), meeting.get('importance_reason'))
    
    tech_stack = extract_dba_tech_stack(meeting)
    db_tasks = extract_simple_info(meeting, ['데이터베이스', 'database', 'db', 'sql', '쿼리', '최적화', '인덱스', 'mysql'])
    
    template = f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}
👥 참가자: {participants_str}
🗄️ DB 기술: {', '.join(tech_stack) if tech_stack else '정보 없음'}

🎯 회의 목적:
{purpose}

📋 안건:
{agenda}

⭐ 중요도: {importance_str}

💡 요약:
{meeting.get('summary') or '없음'}

💾 데이터베이스 작업사항:
{db_tasks}
"""
    return template

# ============================================================
def format_security_developer_meeting(meeting: dict) -> str:
    """SECURITY_DEVELOPER용 회의 템플릿"""
    scheduled_at = meeting.get('scheduled_at', '')
    try:
        if isinstance(scheduled_at, str):
            dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        else:
            dt = scheduled_at
        date_str = dt.strftime('%Y년 %m월 %d일')
    except:
        date_str = str(scheduled_at)[:10] if scheduled_at else '날짜 없음'
    
    # 추가 정보 추출
    participants = meeting.get('participants', [])
    participants_str = ', '.join(participants) if participants else '정보 없음'
    purpose = meeting.get('purpose') or '정보 없음'
    agenda = meeting.get('agenda') or '정보 없음'
    importance_str = format_importance(meeting.get('importance_level'), meeting.get('importance_reason'))
    
    tech_stack = extract_security_tech_stack(meeting)
    security_tasks = extract_simple_info(meeting, ['보안', 'security', '취약점', '암호화', '인증', '권한'])
    
    template = f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}
👥 참가자: {participants_str}
🔒 보안 도구: {', '.join(tech_stack) if tech_stack else '정보 없음'}

🎯 회의 목적:
{purpose}

📋 안건:
{agenda}

⭐ 중요도: {importance_str}

💡 요약:
{meeting.get('summary') or '없음'}

🛡️ 보안 작업사항:
{security_tasks}
"""
    return template

# ============================================================

def format_single_meeting_with_persona(meeting: dict, user_job: str) -> str:
    """Job에 따라 다른 템플릿 적용 (실제 DB job enum에 맞춤)"""
    
    """직무별 페르소나 템플릿 적용"""
    # NONE이면 기본 템플릿 사용
    if not user_job or user_job == 'NONE':
        return format_single_meeting_basic(meeting)
    
    # 정규화 (대문자 변환)
    user_job = user_job.upper() if user_job else 'NONE'

    # 유효한 직무만 허용
    valid_jobs = ['NONE', 'PROJECT_MANAGER', 'FRONTEND_DEVELOPER', 
                'BACKEND_DEVELOPER', 'DATABASE_ADMINISTRATOR', 'SECURITY_DEVELOPER']
    if user_job not in valid_jobs:
        user_job = 'NONE'
    
    # DB의 실제 ENUM 값에 맞춤
    if user_job == 'PROJECT_MANAGER':
        return format_project_manager_meeting(meeting)
    elif user_job == 'FRONTEND_DEVELOPER':
        return format_frontend_developer_meeting(meeting)
    elif user_job == 'BACKEND_DEVELOPER':
        return format_backend_developer_meeting(meeting)
    elif user_job == 'DATABASE_ADMINISTRATOR':
        return format_database_administrator_meeting(meeting)
    elif user_job == 'SECURITY_DEVELOPER':
        return format_security_developer_meeting(meeting)
    else:
        # NONE이거나 인식 못한 경우 기본 템플릿
        return format_single_meeting(meeting)
    
def format_single_meeting_basic(meeting: dict) -> str:
    """직무가 NONE일 때 사용하는 기본 템플릿"""
    scheduled_at = meeting.get('scheduled_at', '')
    try:
        if isinstance(scheduled_at, str):
            dt = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        else:
            dt = scheduled_at
        date_str = dt.strftime('%Y년 %m월 %d일')
    except:
        date_str = str(scheduled_at)[:10] if scheduled_at else '날짜 없음'
    
    response = f"""📌 {meeting.get('title', '제목 없음')}
📅 날짜: {date_str}

📝 회의 설명:
{meeting.get('description', '설명 없음')}

💡 회의 요약:
{meeting.get('summary', '요약 없음')}"""
    
    return response

# ============================================================

def format_my_tasks(tasks: list, status_text: str = "") -> str:
    """내 할 일 목록 포맷팅"""
    total = len(tasks)
    
    if status_text:
        response = f"📋 {status_text} 일 {total}개:\n\n"
    else:
        response = f"📋 {total}개의 할 일이 있어요!\n\n"
    
    for i, task in enumerate(tasks[:10], 1):
        title = task.get('title', '제목 없음')
        meeting_title = task.get('meeting_title', '회의 없음')
        due_date = task.get('due_date')
        status = task.get('status', 'TODO')
        
        status_emoji = "✅" if status == 'COMPLETED' else "⏳"
        
        if due_date:
            due_str = f"📅 {due_date.strftime('%m월 %d일')}"
        else:
            due_str = "📅 기한 없음"
        
        response += f"{status_emoji} {i}. {title}\n"
        response += f"   회의: {meeting_title}\n"
        response += f"   {due_str}\n\n"
    
    if total > 10:
        response += f"💡 최대 10개까지만 보여드려요!\n"
        response += f"   더 자세히 보려면 '회의 선택 → 할 일 조회'를 이용해주세요.\n"
    
    return response


def format_assignee_tasks(tasks: list, name: str, status_text: str = "") -> str:
    """특정 담당자의 할 일 목록 포맷팅"""
    total = len(tasks)
    
    if status_text:
        response = f"📋 {name}님이 {status_text} 일 {total}개:\n\n"
    else:
        response = f"📋 {name}님이 담당한 일 {total}개:\n\n"
    
    for i, task in enumerate(tasks[:10], 1):
        title = task.get('title', '제목 없음')
        meeting_title = task.get('meeting_title', '회의 없음')
        due_date = task.get('due_date')
        status = task.get('status', 'TODO')
        
        status_emoji = "✅" if status == 'COMPLETED' else "⏳"
        
        if due_date:
            due_str = f"📅 {due_date.strftime('%m월 %d일')}"
        else:
            due_str = "📅 기한 없음"
        
        response += f"{status_emoji} {i}. {title}\n"
        response += f"   회의: {meeting_title}\n"
        response += f"   {due_str}\n\n"
    
    if total > 10:
        response += f"💡 최대 10개까지만 보여드려요!\n"
        response += f"   더 자세히 보려면 '회의 선택 → 할 일 조회'를 이용해주세요.\n"
    
    return response

def format_meeting_tasks(tasks: list, meeting_title: str = None, exclude_self: bool = False) -> str:
    """특정 회의의 할 일 목록 포맷팅"""
    total = len(tasks)
    
    # 0개일 때 명시적 메시지
    if total == 0:
        if exclude_self:
            # "다른 사람은?" 질문에 대한 응답
            if meeting_title:
                return f"📋 {meeting_title}에서 다른 사람이 맡은 할 일은 없어요! 😊"
            else:
                return f"📋 이 회의에서 다른 사람이 맡은 할 일은 없어요! 😊"
        else:
            if meeting_title:
                return f"📋 {meeting_title}에서 정한 할 일이 없어요! 😊"
            else:
                return f"📋 이 회의에서 정한 할 일이 없어요! 😊"
            
    # 1개 이상일 때
    if meeting_title:
        response = f"📋 {meeting_title}에서 정한 할 일: {total}개\n\n"
    else:
        response = f"📋 이 회의에서 정한 할 일: {total}개\n\n"

    for i, task in enumerate(tasks, 1):
        title = task.get('title', '제목 없음')
        assignee = task.get('assignee_name', '미정')
        due_date = task.get('due_date')
        status = task.get('status', 'TODO')
        
        status_emoji = "✅" if status == 'COMPLETED' else "⏳"
        
        if due_date:
            due_str = f"📅 {due_date.strftime('%m월 %d일')}"
        else:
            due_str = "📅 기한 없음"
        
        response += f"{status_emoji} {i}. {title}\n"
        response += f"   담당: {assignee}\n"
        response += f"   {due_str}\n\n"
    
    return response

# ============================================================
# Participant 포맷팅
# ============================================================

def format_meeting_participants(meeting: dict, participants: list) -> str:
    """
    특정 회의의 참석자 목록 포맷팅
    
    meeting: {'title': ..., 'scheduled_at': ...}
    participants: [{'name': ..., 'speaker_id': ..., 'job': ...}, ...]
    """
    from datetime import datetime
    
    title = meeting['title']
    scheduled_at = meeting['scheduled_at']
    
    # 날짜 포맷팅
    if isinstance(scheduled_at, str):
        scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
    date_str = scheduled_at.strftime('%Y년 %m월 %d일')
    
    # 메시지 생성
    message = f"네, {title} 회의 참석자를 알려드릴게요! 👥\n\n"
    message += f"📅 {date_str}\n\n"
    
    for p in participants:
        name = p['name']
        speaker_id = p.get('speaker_id', '')
        job = p.get('job', 'NONE')
        
        # 직무 한글 변환
        job_kr = {
            'PROJECT_MANAGER': '기획자',
            'FRONTEND_DEVELOPER': '프론트엔드',
            'BACKEND_DEVELOPER': '백엔드',
            'DATABASE_ADMINISTRATOR': 'DBA',
            'SECURITY_DEVELOPER': '보안',
            'NONE': ''
        }.get(job, '')
        
        # 정보 조합
        info_parts = [name]
        if job_kr:
            info_parts.append(f"({job_kr})")
        participant_info = " ".join(info_parts)
        message += f"• {participant_info}\n"
    
    return message.strip()


def format_person_meetings(user: dict, meetings: list) -> str:
    """
    특정 사람이 참석한 회의 목록 포맷팅
    
    user: {'id': ..., 'name': ..., 'job': ...}
    meetings: [{'id': ..., 'title': ..., 'scheduled_at': ..., 'status': ..., 'role': ...}, ...]
    """
    from datetime import datetime
    
    name = user['name']
    
    # 단일 회의 부분 수정
    if len(meetings) == 1:
        meeting = meetings[0]
        title = meeting['title']
        scheduled_at = meeting['scheduled_at']
        status = meeting['status']
        # role 관련 코드 삭제!
        
        # 날짜 포맷팅
        if isinstance(scheduled_at, str):
            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        date_str = scheduled_at.strftime('%Y년 %m월 %d일')
        
        # 상태 한글 변환
        status_kr = {
            'COMPLETED': '완료됨',
            'SCHEDULED': '예정',
            'RECORDING': '진행중'
        }.get(status, status)
        
        message = f"네, {name}님이 참석한 회의가 있어요! 📌\n\n"
        message += f"{title}\n"
        message += f"📅 {date_str} ({status_kr})\n\n"
        
        if meeting.get('description'):
            message += f"💡 {meeting['description']}"
        
        return message.strip()
    
    else:
        # 여러 회의
        message = f"네, {name}님이 참석한 회의는 총 {len(meetings)}개예요! 📋\n\n"
        
        for i, meeting in enumerate(meetings[:10], 1):
            title = meeting['title']
            scheduled_at = meeting['scheduled_at']
            status = meeting['status']
            
            # 날짜 포맷팅
            if isinstance(scheduled_at, str):
                scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
            date_str = scheduled_at.strftime('%m월 %d일')
            
            # 상태 이모지
            status_emoji = {
                'COMPLETED': '✅',
                'SCHEDULED': '📅',
                'RECORDING': '🔴'
            }.get(status, '📌')
            
            message += f"{i}. {status_emoji} {title} ({date_str})\n"
        
        if len(meetings) > 10:
            message += f"\n💡 이 외에도 {len(meetings) - 10}개가 더 있어요!"
        
        message += "\n\n번호를 말씀해주시면 자세히 알려드릴게요! 😊"
        
        return message.strip()
    
def calculate_shown_counts(meetings: list) -> tuple:
    """
    회의 목록에서 실제 보여줄 개수 계산
    
    Returns:
        (shown_completed, shown_scheduled)
    """
    completed = [m for m in meetings if m.get('status') == 'COMPLETED']
    scheduled = [m for m in meetings if m.get('status') == 'SCHEDULED']
    
    # format_multiple_meetings_short와 동일한 로직
    if len(completed) > 0 and len(scheduled) > 0:
        # 둘 다 있으면: 완료 3개 + 예정 3개
        return min(3, len(completed)), min(3, len(scheduled))
    elif len(completed) > 0:
        # 완료만 있으면: 완료 5개
        return min(5, len(completed)), 0
    else:
        # 예정만 있으면: 예정 5개
        return 0, min(5, len(scheduled))
    

def format_meeting_tasks(tasks: list, meeting_title: str = None) -> str:
    """특정 회의의 Task 포맷팅 (전체 할일)"""
    
    if not tasks or len(tasks) == 0:
        if meeting_title:
            return f"📌 {meeting_title}에서 정한 할 일이 없어요! 😊"
        return "📌 이 회의에서 정한 할 일이 없어요! 😊"
    
    title_text = f"📌 {meeting_title}의 할 일" if meeting_title else "📌 할 일 목록"
    answer = f"{title_text} ({len(tasks)}개)\n\n"
    
    for i, task in enumerate(tasks, 1):
        status_emoji = "✅" if task.get('status') == 'COMPLETED' else "⏳"
        title = task.get('title', '제목 없음')
        assignee = task.get('assignee_name') or task.get('assignee_real_name', '미정')
        due_date = task.get('due_date')
        
        answer += f"{status_emoji} {i}. {title}\n"
        answer += f"   담당자: {assignee}\n"
        
        if due_date:
            if hasattr(due_date, 'strftime'):
                answer += f"   마감일: {due_date.strftime('%m월 %d일')}\n"
            else:
                answer += f"   마감일: {due_date}\n"
        else:
            answer += f"   마감일: 미정\n"
        
        answer += "\n"
    
    return answer.strip()