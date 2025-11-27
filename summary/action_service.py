# action_service.py
# "내 할 일 생성" 기능을 전담합니다.

import os
import asyncio
import httpx
import re
import uuid
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from fastapi import HTTPException

# .env 파일 로드
load_dotenv()

# --- 환경 변수 ---
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_API_URL = os.getenv("CLOVA_STUDIO_URL")

# --- 상수 (JOB_PERSONAS) ---
# "액션 아이템" 프롬프트는 'general'만 사용하지만, call_hyperclova 함수가 참조하므로 포함합니다.
JOB_PERSONAS = {
    'PROJECT_MANAGER': '당신은 프로젝트 관리자(PM)입니다. 일정, 리소스, 주요 결정사항을 중요하게 봅니다.',
    'FRONTEND_DEVELOPER': '당신은 프론트엔드 개발자입니다. UI/UX, API 연동, 사용자 인터랙션을 중요하게 봅니다.',
    'BACKEND_DEVELOPER': '당신은 백엔드 개발자입니다. API 설계, 데이터베이스, 서버 아키텍처, 성능을 중요하게 봅니다.',
    'DATABASE_ADMINISTRATOR': '당신은 데이터베이스 관리자(DBA)입니다. 데이터 모델링, 쿼리, 데이터 무결성을 중요하게 봅니다.',
    'SECURITY_DEVELOPER': '당신은 보안 전문가입니다. 인증, 인가, 데이터 암호화, 취약점을 중요하게 봅니다.',
    'general': '당신은 회의록 작성 전문가입니다. 회의 내용을 명확하고 간결하게 요약합니다.'
}

# --- Pydantic DTO (이 서비스와 관련된 모델들) ---
# 참고: summary_service.py에도 이 DTO들이 중복으로 필요합니다.
class Transcript(BaseModel):
    speaker: str
    time: str = ""
    text: str

class ActionRequest(BaseModel):
    transcripts: List[Transcript]
    speakerMapping: Dict[str, str] = {}
    meetingDate: str = datetime.now().strftime("%Y-%m-%d")
    userJob: str = 'general'
    currentUserName: Optional[str] = None

class ActionItem(BaseModel):
    title: str
    assignee: str
    deadline: str
    addedToCalendar: bool = False
    source: str = 'ai'

class ActionResponse(BaseModel):
    success: bool
    actions: Optional[List[ActionItem]] = None
    error: Optional[str] = None

# --- 공통 헬퍼 함수 (utils) ---
# 참고: summary_service.py에도 이 함수들이 중복으로 필요합니다.

def generate_request_id():
    return f"meeting-{int(time.time() * 1000)}-{uuid.uuid4().hex[:9]}"

def convert_relative_date(relative_date: str, meeting_date_str: str) -> str:
    if not relative_date or not meeting_date_str:
        return relative_date

    try:
        if 'T' in meeting_date_str:
            meeting = datetime.fromisoformat(meeting_date_str.replace('Z', '+00:00')).date()
        else:
            meeting = datetime.strptime(meeting_date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            meeting = datetime.strptime(meeting_date_str, "%Y-%m-%d").date()
        except ValueError:
            return relative_date

    month_end_match = re.search(r'(\d{1,2})월\s*말', relative_date)
    if month_end_match:
        try:
            target_month = int(month_end_match.group(1))
            target_year = meeting.year
            
            if target_month < meeting.month:
                target_year += 1
                
            result_date = datetime(target_year, target_month, 1) + relativedelta(day=31)
            return result_date.strftime("%Y-%m-%d")
        except:
            pass

    # 연도 보정
    if re.match(r'\d{4}-\d{2}-\d{2}', relative_date):
        try:
            parsed_date = datetime.strptime(relative_date, "%Y-%m-%d").date()
            if parsed_date.year < meeting.year: 
                new_date = parsed_date.replace(year=meeting.year)
                return new_date.strftime("%Y-%m-%d")
            return relative_date
        except:
            pass

    # 말일/월말/지지난달 말 등 처리
    if "말일" in relative_date or "월말" in relative_date or ("달" in relative_date and "말" in relative_date):
        if "다음" in relative_date:
            return (meeting + relativedelta(months=1, day=31)).strftime("%Y-%m-%d")
        elif "지지난" in relative_date: 
            return (meeting - relativedelta(months=2, day=31)).strftime("%Y-%m-%d")
        elif "지난" in relative_date:
            return (meeting - relativedelta(months=1, day=31)).strftime("%Y-%m-%d")
        else:
            return (meeting + relativedelta(day=31)).strftime("%Y-%m-%d")

    # 요일 처리
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    day_match = re.search(r"([월화수목금토일])요일", relative_date)
    target_weekday = day_map.get(day_match.group(1)) if day_match else None

    if target_weekday is None:
        if relative_date == "오늘": return meeting.strftime("%Y-%m-%d")
        if relative_date == "내일": return (meeting + relativedelta(days=1)).strftime("%Y-%m-%d")
        if relative_date == "모레": return (meeting + relativedelta(days=2)).strftime("%Y-%m-%d")
        return relative_date

    current_weekday = meeting.weekday()
    days_since_sunday = (current_weekday + 1) % 7
    this_week_sunday = meeting - relativedelta(days=days_since_sunday)

    if "이번 주" in relative_date and target_weekday is not None:
        target_index = (target_weekday + 1) % 7
        result = this_week_sunday + relativedelta(days=target_index)
        return result.strftime("%Y-%m-%d")

    if "다음 주" in relative_date and target_weekday is not None:
        next_week_sunday = this_week_sunday + relativedelta(days=7)
        target_index = (target_weekday + 1) % 7
        result = next_week_sunday + relativedelta(days=target_index)
        return result.strftime("%Y-%m-%d")

    if target_weekday is not None and "다음" not in relative_date and "이번" not in relative_date:
        target_index = (target_weekday + 1) % 7
        candidate = this_week_sunday + relativedelta(days=target_index)
        if candidate <= meeting:
            candidate += relativedelta(days=7)
        return candidate.strftime("%Y-%m-%d")

    return relative_date

def parse_actions(
    actions_text: str, 
    speaker_mapping: Dict[str, str], 
    meeting_date: str, 
    convert_dates: bool = False,
    source: str = 'ai' 
) -> List[ActionItem]:

    actions = []
    lines = actions_text.split('\n')

    time_regex = re.compile(r'((?:오전|오후)?\s*\d{1,2}시(?:\s*\d{1,2}분)?)')
    
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'다음\s*주\s*[월화수목금토일]요일',
        r'이번\s*주\s*[월화수목금토일]요일',
        r'[월화수목금토일]요일(까지)?',
        r'오늘|내일|모레',
        r'이번\s*달\s*말일',
        r'[0-9]{1,2}월\s*[0-9]{1,2}일',
        r'[0-9]{1,2}월\s*말(일)?',
        r'월말',
    ]
    date_regex = re.compile("|".join(date_patterns))

    for line in lines:
        trimmed = line.strip()
        # 리스트 형태가 아니면 스킵
        if not (trimmed.startswith('-') or trimmed.startswith('•') or re.match(r'^\d+\.', trimmed)):
            continue

        text = re.sub(r'^[-•\d.)\s]+', '', trimmed).strip()

        assignee = ''
        raw_assignee = ''
        time_str = ""
        deadline = ''

        # 1. 담당자 파싱
        assignee_match = re.search(r'\(([^)]+)\)', text)
        if assignee_match:
            potential_assignee = assignee_match.group(1).strip()
            if not time_regex.fullmatch(potential_assignee):
                if potential_assignee in ('팀 담당', '담당자 미지정'):
                    raw_assignee = potential_assignee
                else:
                    raw_assignee = re.sub(r'\s*담당$', '', potential_assignee).strip()
                text = text.replace(assignee_match.group(0), '').strip()

        if raw_assignee and speaker_mapping:
            assignee = speaker_mapping.get(raw_assignee, raw_assignee)
        elif raw_assignee: 
            assignee = raw_assignee

        # 2. 날짜 파싱 (대괄호 안의 내용 우선 확인)
        found_date_str = ""
        bracket_matches = list(re.finditer(r'\[([^\]]+)\]', text))
        
        for match in bracket_matches:
            content = match.group(1).strip()
            if date_regex.fullmatch(content) or "요일" in content or "말일" in content or "주" in content or "월 말" in content:
                found_date_str = content
                text = text.replace(match.group(0), '', 1).strip()
                break

        if not found_date_str:
            date_match = date_regex.search(text)
            if date_match:
                found_date_str = date_match.group(0)
                text = text.replace(found_date_str, '').strip()

        if found_date_str:
            if convert_dates:
                deadline = convert_relative_date(found_date_str, meeting_date)
            else:
                deadline = found_date_str
        
        # 날짜 파싱 후 빈 대괄호 제거
        text = re.sub(r'\[\s*\]', '', text).strip()

        # 3. 시간 파싱
        time_match = time_regex.search(text)
        if time_match:
            time_str = time_match.group(1).strip()
            text = text.replace(time_match.group(0), '').strip()

        # 시간 추출 후 남은 잔여 텍스트 제거
        text = re.sub(r'[\[\(]\s*(?:이전|이후|경|쯤|안으로)\s*[\]\)]', '', text).strip()

        # 4. 텍스트 클리닝
        text = re.sub(r'까지|합니다|해야\s*합니다', '', text).strip()
        text = re.sub(r'[.,;]$', '', text).strip()
        text = re.sub(r'(을|를|은|는|이|가|와|과|에|에서)$', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        
        if time_str:
            text += f" ({time_str})"

        if "할 일 없음" in text or "없습니다" in text:
            continue

        if text:
            item = ActionItem(
                title=text,
                assignee=assignee,
                deadline=deadline,
                addedToCalendar=False,
                source=source
            )
            actions.append(item)

    return actions

# --- 공통 AI 호출 함수 ---
async def call_hyperclova(
    client: httpx.AsyncClient, 
    conversation_text: str, 
    task_type: str, 
    user_job: str = 'general',
    user_name: Optional[str] = None,
    participants: List[str] = [],
    meeting_date: str = ""
) -> str:

    persona_general = JOB_PERSONAS['general']
    persona_user = JOB_PERSONAS.get(user_job, persona_general)

    # 직무별 팀 키워드 정리
    team_keywords = []
    
    if user_job == 'BACKEND_DEVELOPER':
        team_keywords = ['백엔드 개발팀', '백엔드팀', '개발팀', '서버팀', '백엔드'] 
    elif user_job == 'FRONTEND_DEVELOPER':
        team_keywords = ['프론트엔드 개발팀', '프론트엔드팀', '프론트팀', '프론트엔드', 'UI/UX팀', '클라이언트팀']
    elif user_job == 'DATABASE_ADMINISTRATOR':
        team_keywords = ['DBA팀', 'DBA', 'DB팀', '데이터베이스팀', '데이터팀']
    elif user_job == 'SECURITY_DEVELOPER':
        team_keywords = ['보안팀', '정보보안팀', '보안']
    elif user_job == 'PROJECT_MANAGER':
        team_keywords = ['PM', '기획팀', '프로젝트 관리팀']

    team_keyword_string = ""
    if team_keywords:
        team_keyword_string = f"'{', '.join(team_keywords)}'과(와) 같은"

    participants_str = ", ".join(participants) if participants else "정보 없음"

    prompts = {
        '액션아이템': f"당신은 [{persona_user}]의 관점에서 회의록을 작성하는 비서입니다.\n"
                    f"당신의 이름(사용자)은 '{user_name}'입니다.\n"
                    f"현재 회의 참가자 명단: [{participants_str}]\n"
                    f"**회의 기준 날짜: {meeting_date}**\n\n"
                    
                    f"## 필독: 작업 지침\n"
                    f"회의 대화를 분석하여 **명확하게 합의된 '내 할 일(To-Do)'만** 추출하세요.\n"
                    f"**추측해서 생성하지 마세요.** 대화에 없는 내용은 절대 만들지 마세요.\n"
                    f"할 일이 전혀 발견되지 않으면 반드시 '할 일 없음'이라고만 출력하세요.\n\n"
                    
                    f"1. **유형 1 (본인 의지)**: '{user_name}'(당신)이 직접 하겠다고 말한 작업.\n"
                    f"2. **유형 2 (지시 받음)**: 타인이 '{user_name}'에게 지시하거나 요청한 작업.\n"
                    f"3. **유형 3 (팀 업무)**: '{user_name}'이 속한 팀({team_keyword_string}) 전체에 할당된 작업.\n\n"
                    
                    f"## 필수 포맷 규칙 (엄격 준수)\n"
                    f"- 모든 항목은 반드시 '-' (하이픈)으로 시작하세요.\n"
                    f"- **팀 업무(유형 3)**인 경우: 반드시 내용 맨 앞에 `[팀명]`을 대괄호로 감싸서 붙이세요.\n"
                    f"- 형식: `- [팀명(선택)] 작업 내용 (담당자) [마감기한]`\n\n"
                    
                    f"## 작성 예시 (Good Case vs Bad Case)\n"
                    f"(Good) - [백엔드팀] 쿼리 성능 개선 (담당자 미지정) [이번 주 금요일]\n"
                    f"(Good) - 클라우드 비용 보고서 제출 ({user_name}) [내일]\n"
                    f"(Bad) - 회의록 정리 및 공유 ({user_name}) [내일] -> (설명: 대화에 회의록 정리하겠다는 말이 없으면 절대 쓰지 말 것)\n"
                    f"(Good) - 할 일 없음 -> (설명: 대화가 단순 잡담이거나 구체적인 할 일이 없을 때)\n\n"
                    
                    f"## [중요] 세부 규칙\n"
                    f"1. **할 일이 없는 경우**:\n"
                    f"   - 간식 논의, 단순 인사, 잡담 등 구체적 업무가 없으면 반드시 '할 일 없음'이라고만 적으세요.\n"
                    f"   - '회의록 작성', '정리' 같은 상투적인 내용을 임의로 추가하지 마세요.\n"
                    f"2. **날짜 표기**:\n"
                    f"   - 대화에 날짜가 언급된 경우에만 `[내일]` 처럼 적으세요.\n"
                    f"   - **날짜가 없으면 대괄호 []를 아예 쓰지 마세요.** 빈 괄호 `[]`도 금지입니다.\n\n"
                    
                    f"## 회의 대화\n{conversation_text}\n\n"
                    f"'{user_name}'님의 할 일 목록:"
    }


    if task_type not in prompts:
        raise ValueError(f"지원하지 않는 task_type입니다: {task_type}")

    system_content = persona_user
    token_settings = { '액션아이템': 600 }
    current_max_tokens = token_settings.get(task_type, 600)

    headers = {
        'Authorization': f'Bearer {CLOVA_API_KEY}',
        'Content-Type': 'application/json',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': generate_request_id()
    }

    body = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompts[task_type]}
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": current_max_tokens,
        "temperature": 0.3,
        "repeatPenalty": 5.0,
        "stopBefore": [],
        "includeAiFilters": True
    }

    try:
        response = await client.post(CLOVA_API_URL, headers=headers, json=body, timeout=30.0)
        response.raise_for_status() 
        data = response.json()

        if data.get("status") and data["status"].get("code") != "20000":
            raise HTTPException(status_code=500, detail=f"HyperCLOVA API 오류: {data['status'].get('message')}")

        result_text = data.get("result", {}).get("message", {}).get("content", "") or \
                      data.get("result", {}).get("text", "") 
        
        if not result_text.strip().startswith("-"):
            return ""

        return result_text.strip()

    except httpx.HTTPStatusError as e:
        print(f"HyperCLOVA API 호출 오류: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"HyperCLOVA API 오류: {e.response.text}")
    except Exception as e:
        print(f"API 호출 중 알 수 없는 오류: {e}")
        raise HTTPException(status_code=500, detail=f"API 호출 중 알 수 없는 오류: {e}")


# --- "내 할 일 생성" 비즈니스 로직 ---
# main.py가 이 함수를 호출합니다.
async def generate_all_actions_service(request: ActionRequest) -> List[ActionItem]:
    user_name = request.currentUserName
    user_job = request.userJob

    if not user_name:
        return []

    participants_list = list(set(request.speakerMapping.values()))

    conversation_lines = []
    for t in request.transcripts:
        display_name = request.speakerMapping.get(t.speaker, t.speaker)
        if display_name == user_name:
            speaker_display = user_name
        else:
            speaker_display = display_name
        conversation_lines.append(f"{speaker_display} ({t.time}): {t.text}")

    conversation_text = "\n".join(conversation_lines)

    print(f"[{user_name}] 액션 아이템 생성 요청 (참가자: {participants_list})")

    target_date = request.meetingDate
    if "T" in target_date:
        target_date = target_date.split("T")[0]

    async with httpx.AsyncClient() as client:
        try:
            actions_text = await call_hyperclova(
                client, 
                conversation_text, 
                '액션아이템', 
                user_job, 
                user_name, 
                participants_list, 
                target_date
            )

            if not actions_text:
                return []

            all_actions = parse_actions(
                actions_text, 
                request.speakerMapping, 
                request.meetingDate, 
                convert_dates=True, 
                source='ai' 
            )
            
            final_actions = []
            for action in all_actions:
                assignee = action.assignee
                
                # 1. 대괄호 태그가 있는지 확인 (예: [백엔드팀])
                has_team_tag = re.match(r'^\[.*(?:팀|부서|파트).*\]', action.title)
                
                # 2. 대괄호 없이 'OO팀'으로 시작하는지 확인 (예: 백엔드팀은, 개발팀이)
                starts_with_team = re.match(r'^\s*\S+(?:팀|부서|파트)', action.title)

                # 3. 본문에 '팀 전체', '백엔드팀' 등 키워드가 포함되어 있는지 확인
                has_group_keyword = (
                    "팀 전체" in action.title 
                    or "부서 전체" in action.title 
                    or "팀원들" in action.title 
                    or "팀 내" in action.title
                    or "백엔드팀" in action.title
                    or "프론트엔드팀" in action.title
                    or "개발팀" in action.title
                )

                # 조건: 팀 태그가 있거나, 팀으로 시작하거나, 그룹 키워드가 있는 경우
                if has_team_tag or starts_with_team or has_group_keyword:
                    # '담당자 미지정'으로 강제 설정
                    action.assignee = "담당자 미지정"
                    
                    # 태그가 없었다면 강제로 [팀 업무] 태그 추가
                    if not has_team_tag:
                        action.title = f"[팀 업무] {action.title}"
                        
                    final_actions.append(action)

                # 4. 개인 업무 (참가자 명단에 있는 경우)
                elif assignee in participants_list:
                    final_actions.append(action)
                
                # 5. 미지정 업무
                elif assignee in ['담당자 미지정', '미지정', '']:
                    action.assignee = "담당자 미지정" 
                    final_actions.append(action)

                # 6. 그 외
                else:
                    action.assignee = "담당자 미지정" 
                    final_actions.append(action)

            return final_actions

        except Exception as e:
            print(f"액션 아이템 생성 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))