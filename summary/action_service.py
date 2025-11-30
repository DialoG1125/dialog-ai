# action_service.py
# "내 할 일 생성" 기능을 전담합니다.

import os
import asyncio
import httpx
import re
import uuid
import time
from datetime import datetime
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
JOB_PERSONAS = {
    'PROJECT_MANAGER': '당신은 프로젝트 관리자(PM)입니다. 일정, 리소스, 주요 결정사항을 중요하게 봅니다.',
    'FRONTEND_DEVELOPER': '당신은 프론트엔드 개발자입니다. UI/UX, API 연동, 사용자 인터랙션을 중요하게 봅니다.',
    'BACKEND_DEVELOPER': '당신은 백엔드 개발자입니다. API 설계, 데이터베이스, 서버 아키텍처, 성능을 중요하게 봅니다.',
    'DATABASE_ADMINISTRATOR': '당신은 데이터베이스 관리자(DBA)입니다. 데이터 모델링, 쿼리, 데이터 무결성을 중요하게 봅니다.',
    'SECURITY_DEVELOPER': '당신은 보안 전문가입니다. 인증, 인가, 데이터 암호화, 취약점을 중요하게 봅니다.',
    'general': '당신은 회의록 작성 전문가입니다. 회의 내용을 명확하고 간결하게 요약합니다.'
}

# --- Pydantic DTO ---
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

# --- 공통 헬퍼 함수 ---
def generate_request_id():
    return f"meeting-{int(time.time() * 1000)}-{uuid.uuid4().hex[:9]}"

def parse_actions(
    actions_text: str, 
    speaker_mapping: Dict[str, str], 
    source: str = 'ai' 
) -> List[ActionItem]:

    actions = []
    lines = actions_text.split('\n')
    
    # 날짜 추출용 정규식 (대괄호 포함 여부 상관없이 YYYY-MM-DD 추출)
    date_regex = re.compile(r'(\d{4}-\d{2}-\d{2})')
    # 담당자 추출용 정규식 (소괄호 안의 내용 추출)
    assignee_regex = re.compile(r'\(([^)]+)\)')

    for line in lines:
        trimmed = line.strip()
        # 리스트 마커 제거 (- 또는 숫자.)
        if not (trimmed.startswith('-') or trimmed.startswith('•') or re.match(r'^\d+\.', trimmed)):
            continue
        
        # 순수 텍스트만 남기기 위해 마커 제거
        text = re.sub(r'^[-•\d.)\s]+', '', trimmed).strip()

        assignee = ''
        raw_assignee = ''
        deadline = ''

        # 1. 날짜 파싱 및 제목에서 제거 (가장 먼저 수행)
        date_match = date_regex.search(text)
        if date_match:
            deadline = date_match.group(1)
            # 날짜 패턴을 포함한 대괄호/괄호까지 찾아서 지우기
            text = re.sub(r'\[\s*' + deadline + r'\s*.*?\]', '', text)
            text = re.sub(r'\(\s*' + deadline + r'\s*.*?\)', '', text)
            # 괄호 없이 날짜만 있어도 제거
            text = text.replace(deadline, '')

        # 2. 담당자 파싱 및 제목에서 제거
        assignee_matches = list(assignee_regex.finditer(text))
        for match in assignee_matches:
            content = match.group(1).strip()
            
            # "10시", "오후 2시" 같은 시간 표현은 담당자가 아님 -> 건너뜀
            if re.match(r'.*\d시.*', content):
                continue
            
            # 담당자 후보 확인
            if content in ('팀 담당', '담당자 미지정'):
                raw_assignee = content
            else:
                raw_assignee = re.sub(r'\s*담당$', '', content).strip()
            
            # 제목에서 해당 부분 "(홍길동)" 제거
            text = text.replace(match.group(0), '')
            break # 첫 번째 매칭된 사람을 담당자로 간주

        # 담당자 이름 매핑
        if raw_assignee and speaker_mapping:
            assignee = speaker_mapping.get(raw_assignee, raw_assignee)
        elif raw_assignee: 
            assignee = raw_assignee
        else:
            assignee = "담당자 미지정"

        # 3. 텍스트 최종 정리 (남은 괄호, 특수문자 정리)
        text = re.sub(r'\[\s*\]', '', text) # 빈 대괄호 제거
        text = re.sub(r'\(\s*\)', '', text) # 빈 소괄호 제거
        text = re.sub(r'[.,;]$', '', text)  # 끝 문장부호 제거
        text = re.sub(r'\s+', ' ', text).strip() # 다중 공백 정리

        if "할 일 없음" in text or "없습니다" in text or not text:
            continue

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
    
    participants_str = ", ".join(participants) if participants else "정보 없음"

    # 프롬프트 설정 (포맷 강제)
    prompts = {
        '액션아이템': (
            f"당신은 회의록 작성 AI입니다.\n"
            f"회의 날짜: {meeting_date}\n"
            f"참석자: {participants_str}\n\n"
            f"**지시사항:**\n"
            f"1. 회의 내용을 분석하여 '구체적인 할 일(Action Item)'을 추출하세요.\n"
            f"2. 모든 날짜 표현(내일, 다음주 등)은 회의 날짜({meeting_date})를 기준으로 `YYYY-MM-DD` 포맷으로 변환하세요.\n"
            f"3. 담당자가 명확하지 않으면 '담당자 미지정'이라고 적으세요.\n"
            f"4. **반드시 아래 포맷을 정확히 지키세요.** 다른 말은 하지 마세요.\n\n"
            f"**필수 출력 포맷:**\n"
            f"- 할 일 내용 (담당자이름) [YYYY-MM-DD]\n"
            f"- 할 일 내용 (담당자이름) [YYYY-MM-DD]\n\n"
            f"**예시:**\n"
            f"- API 명세서 작성 (김철수) [2025-11-30]\n"
            f"- [백엔드팀] DB 스키마 설계 (박영희) [2025-12-01]\n"
            f"- 회식 장소 예약 (담당자 미지정) [2025-12-05]\n\n"
            f"**회의 대화:**\n{conversation_text}\n\n"
            f"**결과:**"
        )
    }

    if task_type not in prompts:
        raise ValueError(f"지원하지 않는 task_type입니다: {task_type}")

    headers = {
        'Authorization': f'Bearer {CLOVA_API_KEY}',
        'Content-Type': 'application/json',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': generate_request_id()
    }

    body = {
        "messages": [
            {"role": "system", "content": persona_general},
            {"role": "user", "content": prompts[task_type]}
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 800,
        "temperature": 0.1,
        "repeatPenalty": 5.0,
        "includeAiFilters": True
    }

    try:
        response = await client.post(CLOVA_API_URL, headers=headers, json=body, timeout=30.0)
        response.raise_for_status() 
        data = response.json()
        
        result_text = data.get("result", {}).get("message", {}).get("content", "") or \
                      data.get("result", {}).get("text", "") 
        
        if not result_text.strip().startswith("-"):
            return "할 일 없음"

        return result_text.strip()

    except Exception as e:
        print(f"HyperCLOVA API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"API 오류: {str(e)}")

# --- 내 할 일 생성 로직 ---
async def generate_all_actions_service(request: ActionRequest) -> List[ActionItem]:
    user_name = request.currentUserName
    
    participants_list = list(set(request.speakerMapping.values()))
    conversation_lines = []
    for t in request.transcripts:
        display_name = request.speakerMapping.get(t.speaker, t.speaker)
        conversation_lines.append(f"{display_name} ({t.time}): {t.text}")
    conversation_text = "\n".join(conversation_lines)

    target_date = request.meetingDate.split("T")[0]

    async with httpx.AsyncClient() as client:
        try:
            actions_text = await call_hyperclova(
                client, 
                conversation_text, 
                '액션아이템', 
                'general', 
                user_name, 
                participants_list, 
                target_date 
            )

            if not actions_text or "할 일 없음" in actions_text:
                return []

            final_actions = parse_actions(
                actions_text, 
                request.speakerMapping, 
                source='ai' 
            )

            return final_actions

        except Exception as e:
            print(f"액션 아이템 생성 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))