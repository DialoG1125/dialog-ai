# summary_service.py
# "AI 요약 생성" (회의 목적, 안건, 요약, 중요도, 키워드) 기능을 전담합니다.

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

class SummaryRequest(BaseModel):
    transcripts: List[Transcript]
    meetingDate: str = datetime.now().strftime("%Y-%m-%d") 
    speakerMapping: Dict[str, str] = {} 
    userJob: str = 'general'

class Importance(BaseModel):
    level: str
    reason: str

class ActionItem(BaseModel):
    title: str
    assignee: str
    deadline: str
    addedToCalendar: bool = False
    source: str = 'ai'

class Summary(BaseModel):
    purpose: str
    agenda: str
    overallSummary: str
    importance: str
    keywords: List[str]
    actions: List[ActionItem]

class SummaryResponse(BaseModel):
    success: bool
    summary: Optional[Summary] = None
    error: Optional[str] = None

# --- 공통 헬퍼 함수 ---
def generate_request_id():
    return f"meeting-{int(time.time() * 1000)}-{uuid.uuid4().hex[:9]}"

def map_importance_to_enum(korean_level: str) -> str:
    if '높음' in korean_level or '긴급' in korean_level or '중요' in korean_level or 'HIGH' in korean_level:
        return "HIGH"
    elif '낮음' in korean_level or 'LOW' in korean_level:
        return "LOW"
    else:
        return "MEDIUM"

def analyze_importance(summary: str) -> Importance:
    level = '보통'
    lower_summary = summary.lower()

    if '높음' in lower_summary or 'high' in lower_summary:
        level = '높음'
    elif '낮음' in lower_summary or 'low' in lower_summary:
        level = '낮음'
    elif '보통' in lower_summary or 'medium' in lower_summary:
        level = '보통'

    # 중요도 텍스트에서 불필요한 패턴 제거 및 사유 추출
    clean_reason = re.sub(r'^[\*\-\s]*(중요도(\s*평가)?\s*[:\-]?\s*)?[\*\s]*(HIGH|MEDIUM|LOW|높음|보통|낮음)[\*\s]*[-:]?\s*', '', summary, flags=re.IGNORECASE).strip()
    clean_reason = re.sub(r'^[\*\s]*이유\s*[:\-]?\s*', '', clean_reason, flags=re.IGNORECASE).strip()
    
    if not clean_reason:
        clean_reason = summary

    return Importance(level=level, reason=clean_reason)

# --- 공통 AI 호출 함수 ---
async def call_hyperclova(
    client: httpx.AsyncClient, 
    conversation_text: str, 
    task_type: str, 
    user_job: str = 'general',
    user_name: Optional[str] = None
) -> str:

    persona_general = JOB_PERSONAS['general']
    persona_user = JOB_PERSONAS.get(user_job, persona_general)

    # 프롬프트 설정 (중요도 기준 명시)
    prompts = {
        '회의목적': f"다음 회의의 핵심 목적을 '명사형 종결 어미'(~함, ~논의 등)로 끝나는 한 문장으로 요약하세요.\n\n[대화]\n{conversation_text}\n\n회의 목적:",
        '주요안건': f"회의에서 논의된 주요 안건 3~5가지를 쉼표(,)로 구분하여 단답형으로 나열하세요. (예: 예산 확정, 일정 조율)\n\n[대화]\n{conversation_text}\n\n주요 안건:",
        '전체요약': f"회의 내용을 서술형으로 자연스럽게 요약하세요. 결정된 사항과 향후 계획 위주로 3문장 내외로 작성하세요.\n\n[대화]\n{conversation_text}\n\n전체 요약:",
        
        '중요도': (
            f"[{persona_user}] 관점에서 이 회의의 중요도를 엄격하게 평가하세요.\n\n"
            f"### [판단 기준표]\n"
            f"1. **높음 (HIGH)**: 서비스 장애(500 에러), 긴급 핫픽스, 보안 사고, 매출 직결 이슈, 데드라인 임박.\n"
            f"2. **보통 (MEDIUM)**: 정기 스프린트 회의, 일반적인 기능 개발 논의, 일정 조율, 업무 분배.\n"
            f"3. **낮음 (LOW)**: 단순 스터디 계획, 회식 장소 선정, 가벼운 아이디어 회의, 잡담 위주.\n\n"
            f"### 출력 형식\n"
            f"[높음/보통/낮음] - [판단 사유 한 문장]\n\n"
            f"[대화]\n{conversation_text}\n\n평가:"
        ),
        
        '키워드': f"[{persona_user}] 관점에서 가장 중요한 핵심 명사 키워드 5개를 쉼표로 구분해 추출하세요.\n\n[대화]\n{conversation_text}\n\n키워드:"
    }

    system_content = persona_general
    if task_type in ('중요도', '키워드'):
        system_content = persona_user

    token_settings = {
        '회의목적': 100, '주요안건': 100, '전체요약': 200,
        '중요도': 100, '키워드': 50
    }
    current_max_tokens = token_settings.get(task_type, 500)

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
        return result_text.strip()

    except Exception as e:
        print(f"API 호출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"API 호출 오류: {e}")

# --- AI 요약 생성 로직 ---
async def create_summary(request: SummaryRequest) -> Summary:
    conversation_lines = []
    for t in request.transcripts:
        display_name = request.speakerMapping.get(t.speaker, t.speaker)
        conversation_lines.append(f"{display_name} ({t.time}): {t.text}")
    conversation_text = "\n".join(conversation_lines)

    user_job = request.userJob
    print(f"[{user_job}] 요약 생성 시작")

    async with httpx.AsyncClient() as client:
        try:
            tasks = [
                call_hyperclova(client, conversation_text, '회의목적', user_job),
                call_hyperclova(client, conversation_text, '주요안건', user_job),
                call_hyperclova(client, conversation_text, '전체요약', user_job),
                call_hyperclova(client, conversation_text, '중요도', user_job),
                call_hyperclova(client, conversation_text, '키워드', user_job)
            ]
            results = await asyncio.gather(*tasks)
            purpose_raw, agenda_raw, summary_raw, importance_raw, keywords_raw = results

            def clean_text(text, prefix_pattern):
                text = re.sub(r'\*\*|__|\#\#|\#', '', text)
                text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)
                return text.replace('"', '').replace("'", "").strip()

            purpose = clean_text(purpose_raw, r'^(회의\s*)?목적\s*[:\-]?\s*')
            agenda = clean_text(agenda_raw, r'^(주요\s*)?안건\s*[:\-]?\s*')
            summary_text = clean_text(summary_raw, r'^(전체\s*)?요약\s*[:\-]?\s*')

            importance_obj = analyze_importance(importance_raw)
            enum_value = map_importance_to_enum(importance_obj.level)
            
            clean_reason = re.sub(r'^[\-\s]*', '', importance_obj.reason)
            final_summary = f"{summary_text}\n\n(중요도 판정 사유: {clean_reason})"

            keywords_text = clean_text(keywords_raw, r'^키워드\s*[:\-]?\s*')
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]

            summary_data = Summary(
                purpose=purpose,
                agenda=agenda,
                overallSummary=final_summary,
                importance=enum_value,
                keywords=keywords,
                actions=[]
            )
            return summary_data

        except Exception as e:
            print(f"요약 생성 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))