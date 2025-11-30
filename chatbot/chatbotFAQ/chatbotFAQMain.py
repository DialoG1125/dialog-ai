from dotenv import load_dotenv
import os
from pathlib import Path

# 환경변수 로드
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Dict, Any, Optional, List

# ================== 설정 ==================
TERMS_DB_FILE = os.getenv('TERMS_DB_FILE', '/app/chatbot/chatbotFAQ/data/terms_database.json')

# HyperCLOVA X API
CLOVA_STUDIO_URL = os.getenv('CLOVA_STUDIO_URL')
CLOVA_API_KEY = os.getenv('CLOVA_API_KEY')

# 챗봇 빌더 API
CHATBOT_API_URL = os.getenv('CHATBOT_API_URL')
CHATBOT_SECRET_KEY = os.getenv('CHATBOT_SECRET_KEY')

# 서버 설정
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))

# ================== FastAPI 앱 생성 ==================
app = FastAPI(title="IT 용어 챗봇 API (비용 효율)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== 데이터 로드 ==================
terms_db = {}

def load_terms_database():
    """용어 데이터베이스 로드 (명시적 호출용)"""
    global terms_db
    
    if terms_db:  # 이미 로드됨
        return
    
    # 절대 경로로 변환
    if TERMS_DB_FILE.startswith('./'):
        base_path = Path(__file__).parent
        file_path = base_path / TERMS_DB_FILE.replace('./', '')
    else:
        file_path = Path(TERMS_DB_FILE)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            terms_db = json.load(f)
        print(f"✅ {len(terms_db)}개 용어 로드 완료 (경로: {file_path})")
    except FileNotFoundError:
        print(f"❌ {file_path} 파일이 없습니다!")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()

# 초기 로드 시도
load_terms_database()

# ================== 유연한 검색 함수 ==================
def search_term_flexible(query: str) -> Optional[Dict[str, Any]]:
    import re
    
    query_lower = query.lower().strip()
    
    # 질문어 제거
    cleaned = re.sub(
        r'(가\s*뭐야|이\s*뭐야|뭐야|뭔가요|알려줘|설명해줘|이란|란)\??$',
        '',
        query_lower
    ).strip()
    
    print(f"🔍 검색어: '{query}' → '{cleaned}'")
    
    # 1단계: 정확한 매칭
    if cleaned in terms_db:
        print(f"✅ 정확 매칭: '{cleaned}'")
        return terms_db[cleaned]
    
    # 2단계: 유사어 완전 일치
    for key, value in terms_db.items():
        synonyms = value.get('synonyms', [])
        if any(cleaned == syn.lower() for syn in synonyms):
            print(f"✅ 유사어 매칭: '{cleaned}' → '{value['name']}'")
            return value
    
    # 3단계: 부분 매칭 (띄어쓰기 무시) - 더 엄격하게!
    query_no_space = cleaned.replace(" ", "")
    
    # 3글자 이상이고, 길이가 비슷해야 함!
    if len(query_no_space) >= 3:
        for key, value in terms_db.items():
            key_no_space = key.replace(" ", "")
            
            # 길이 체크: 검색어와 키의 길이 차이가 3자 이내
            length_diff = abs(len(query_no_space) - len(key_no_space))
            
            if length_diff <= 3:
                # 양방향 포함 체크
                if key_no_space in query_no_space or query_no_space in key_no_space:
                    print(f"✅ 부분 매칭: '{key}' ← '{query}'")
                    return value
    
    # 4단계: 토큰 기반 유사도
    def get_tokens(text):
        return set(re.findall(r'[가-힣a-zA-Z0-9]+', text.lower()))
    
    query_tokens = get_tokens(cleaned)
    best_match = None
    best_score = 0
    
    for key, value in terms_db.items():
        all_text = key + " " + " ".join(value.get('synonyms', []))
        term_tokens = get_tokens(all_text)
        
        intersection = len(query_tokens & term_tokens)
        union = len(query_tokens | term_tokens)
        
        if union > 0:
            score = intersection / union
            if score > best_score and score > 0.2:
                best_score = score
                best_match = value
    
    if best_match:
        print(f"✅ 토큰 매칭: 유사도 {best_score:.2f}")
    
    return best_match

# ================== 요청/응답 모델 ==================
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    answer: str
    history: List[Dict[str, Any]]
    source: str

# ================== 엔드포인트 ==================

@app.get("/")
def root():
    unique_terms = len(set([v['name'] for v in terms_db.values()]))
    return {
        "status": "ok",
        "service": "IT 용어 챗봇 (비용 효율)",
        "unique_terms": unique_terms,
        "search_keys": len(terms_db),
        "fallback_order": ["JSON (무료)", "챗봇 빌더 (저렴)", "HyperCLOVA X (비쌈)"]
    }

# ================== 시스템 프롬프트 ==================
SYSTEM_PROMPT = """
🚨 중요: IT/AI/프로그래밍 용어 전문 챗봇

당신은 IT 용어만 설명하는 전문 챗봇입니다.

## IT 용어 정의 (반드시 이 의미로만 답변!)
- LLM = Large Language Model (대형 언어 모델, ChatGPT 같은 AI)
- API = Application Programming Interface (소프트웨어 간 통신 규칙)
- RAG = Retrieval-Augmented Generation (검색 증강 생성, AI가 외부 지식 참조)
- GPU = Graphics Processing Unit (그래픽 처리 장치)

❌ 절대 금지: 법률, 의학, 경영 용어 설명
❌ 절대 금지: 존재하지 않는 용어 지어내기

## 답변 규칙
1. **간결하게**: 3~5문장
2. **친근한 톤**: 존댓말이지만 편안하게 + 이모지 1~2개 사용
3. **IT 예시**: AI/프로그래밍 관련 예시만
4. **형식 금지**: [예시/활용] 같은 제목 쓰지 말기

## 이모지 가이드
😊 🤗 (친근함)
💡 ✨ (아이디어/핵심)
📚 📖 (설명/학습)
🎯 🔍 (핵심/검색)
💻 ⚙️ (기술/개발)
✅ ❌ (옳음/그름)
😢 🙏 (미안함/양해)

## 좋은 예시

### LLM 질문 시:
"LLM(Large Language Model)은 수십억 개의 텍스트로 학습한 초대형 AI 언어 모델이에요. 📚

ChatGPT, Claude, Gemini 같은 챗봇이 바로 LLM 기반이고, 코드 생성, 번역, 문서 작성 등을 자동으로 처리할 수 있어요.

요즘 개발자들이 코딩 도우미로 많이 쓰고 있답니다! ✨"

### RAG 질문 시:
"RAG(Retrieval-Augmented Generation)는 AI가 답변할 때 외부 데이터베이스에서 정보를 먼저 검색한 후 답변을 생성하는 기술이에요. 📚

예를 들어, 회사 내부 문서나 최신 뉴스를 참고해서 더 정확한 답변을 만들 수 있어요. ChatGPT가 인터넷 검색 기능을 쓰는 것도 RAG 방식이죠.

LLM의 오래된 지식 문제를 해결할 수 있어서 요즘 많이 쓰인답니다! ✨"

## 금지사항
❌ IT 외 분야 설명
❌ 존재하지 않는 용어 만들기
❌ [예시] [활용] 같은 섹션 제목
❌ 5문장 초과
❌ 추측성 정보

## 모를 때
"죄송해요, IT 용어 데이터베이스에서 찾지 못했어요. 😢
다른 표현으로 질문해주시겠어요?"
""".strip()

# ================== CLOVA Proxy 엔드포인트 ==================
@app.post("/clova_proxy")
async def clova_proxy(request: dict):
    """CLOVA Studio Proxy (HyperCLOVA X 직접 호출)"""
    import requests
    import uuid
    
    try:
        # CLOVA Chatbot 형식 → Studio 형식 변환
        bubbles = request.get('bubbles', [])
        user_message = ""
        
        if bubbles and len(bubbles) > 0:
            user_message = bubbles[0].get('data', {}).get('description', '')
        
        if not user_message:
            raise HTTPException(status_code=400, detail="메시지가 비어있습니다")
        
        # Studio API 형식으로 변환
        studio_request = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 300,
            "temperature": 0.5,
            "repeatPenalty": 5.0,
            "stopBefore": [],
            "includeAiFilters": True
        }
                
        headers = {
            'Authorization': f'Bearer {CLOVA_API_KEY}',
            'Content-Type': 'application/json',
            'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
        }
        
        response = requests.post(
            CLOVA_STUDIO_URL,
            headers=headers,
            json=studio_request,
            timeout=30
        )
        
        if response.status_code == 200:
            # Studio 응답 → Chatbot 형식으로 변환
            studio_response = response.json()
            
            # Studio 응답에서 답변 추출
            answer = ""
            if 'result' in studio_response:
                message = studio_response['result'].get('message', {})
                answer = message.get('content', '')
            
            # Chatbot 형식으로 반환
            return {
                "bubbles": [{
                    "type": "text",
                    "data": {
                        "description": answer or "답변을 생성할 수 없습니다."
                    }
                }]
            }
        else:
            print(f"❌ CLOVA 오류: {response.text}")
            raise HTTPException(status_code=502, detail="CLOVA Studio API 오류")
    
    except Exception as e:
        print(f"❌ Proxy 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(e))

# ================== /chat 엔드포인트 (비용 효율 폴백) ==================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    비용 효율적인 폴백 채팅 엔드포인트
    """
    # 데이터 로드 확인
    if not terms_db:
        print("⚠️ terms_db 비어있음 - 재로드 시도")
        load_terms_database()
        
        if not terms_db:
            return ChatResponse(
                answer="죄송해요, 용어 데이터베이스를 로드할 수 없어요. 😢",
                history=request.history + [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": "데이터베이스 로드 실패"}
                ],
                source="error"
            )
    
    try:        
        # === 1단계: JSON 검색 (무료, 가장 빠름!) ===
        print(f"\n📚 [1단계] JSON 검색: '{request.message}'")
        result = search_term_flexible(request.message)
        
        if result:
            answer = result['answer']
            term_name = result['name']
            
            # 유사어 정보 추가
            if result.get('synonyms'):
                synonyms_text = ", ".join(result['synonyms'][:3])
                answer += f"\n\n💡 관련 용어: {synonyms_text}"
            
            print(f"✅ JSON에서 찾음: '{term_name}'")
            
            new_history = request.history + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": answer}
            ]
            
            return ChatResponse(
                answer=answer,
                history=new_history,
                source="json"
            )
        
        print(f"❌ JSON에 없음")
        
        # === 2단계: 챗봇 빌더 시도 ===
        print(f"\n💬 [2단계] 챗봇 빌더 호출")
        
        try:
            import requests
            import hashlib
            import hmac
            import base64
            import time
            
            timestamp = int(time.time() * 1000)
            
            body = {
                "version": "v2",
                "userId": "fastapi-user",
                "timestamp": timestamp,
                "bubbles": [{
                    "type": "text",
                    "data": {"description": request.message}
                }],
                "event": "send"
            }
            
            # Request Body를 JSON 문자열로 변환
            body_string = json.dumps(body)
            
            # HMAC 서명 생성
            secret_key_bytes = CHATBOT_SECRET_KEY.encode('utf-8')
            body_bytes = body_string.encode('utf-8')
            
            signature = base64.b64encode(
                hmac.new(secret_key_bytes, body_bytes, digestmod=hashlib.sha256).digest()
            ).decode('utf-8')
            
            # API 호출
            headers = {
                'Content-Type': 'application/json',
                'X-NCP-CHATBOT_SIGNATURE': signature
            }
            
            response = requests.post(
                CHATBOT_API_URL,
                headers=headers,
                data=body_string,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                chatbot_answer = data.get('bubbles', [{}])[0].get('data', {}).get('description', '')
                
                print(f"📥 챗봇 빌더 답변: '{chatbot_answer[:100]}...'")
                
                # 1) 무효 키워드 체크
                fallback_keywords = ['모르', '찾을 수 없', '없습니다', '제공', '구체적', '이해하지', '죄송']
                has_fallback_keyword = any(keyword in chatbot_answer for keyword in fallback_keywords)
                
                # 2) 답변 길이 체크 (너무 짧으면 무효)
                is_too_short = len(chatbot_answer) < 50
                
                # 3) 질문 관련성 체크 (질문의 핵심 단어가 답변에 있는지)
                import re
                query_keywords = set(re.findall(r'[가-힣a-zA-Z]{2,}', request.message.lower()))
                answer_keywords = set(re.findall(r'[가-힣a-zA-Z]{2,}', chatbot_answer.lower()))
                
                # 질문 키워드 중 하나라도 답변에 있어야 함
                has_relevance = bool(query_keywords & answer_keywords)
                
                print(f"🔍 무효 키워드: {has_fallback_keyword}")
                print(f"🔍 답변 길이: {len(chatbot_answer)} ({'너무 짧음' if is_too_short else 'OK'})")
                print(f"🔍 관련성: {has_relevance} (질문: {query_keywords}, 답변: {list(answer_keywords)[:3]}...)")
                
                # 유효성 판단
                is_valid = (
                    chatbot_answer and 
                    not has_fallback_keyword and 
                    not is_too_short and
                    has_relevance
                )
                
                if is_valid:
                    print(f"✅ 챗봇 빌더 성공: 유효한 답변")
                    
                    new_history = request.history + [
                        {"role": "user", "content": request.message},
                        {"role": "assistant", "content": chatbot_answer}
                    ]
                    
                    return ChatResponse(
                        answer=chatbot_answer,
                        history=new_history,
                        source="chatbot_builder"
                    )
                else:
                    print(f"⚠️ 챗봇 빌더 답변 무효 → HyperCLOVA X 시도")
            
            print(f"❌ 챗봇 빌더 실패: {response.status_code}")
        
        except Exception as e:
            print(f"❌ 챗봇 빌더 오류: {e}")
        
        # === 3단계: HyperCLOVA X (최후의 수단) ===
        print(f"\n🤖 [3단계] HyperCLOVA X 호출")
        
        try:
            import requests
            import uuid
            
            studio_request = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": request.message
                    }
                ],
                "topP": 0.8,
                "topK": 0,
                "maxTokens": 300,
                "temperature": 0.5,
                "repeatPenalty": 5.0,
                "stopBefore": [],
                "includeAiFilters": True
            }
            
            headers = {
                'Authorization': f'Bearer {CLOVA_API_KEY}',
                'Content-Type': 'application/json',
                'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
            }
            
            response = requests.post(
                CLOVA_STUDIO_URL,
                headers=headers,
                json=studio_request,
                timeout=30
            )
            
            if response.status_code == 200:
                studio_response = response.json()
                
                answer = ""
                if 'result' in studio_response:
                    message = studio_response['result'].get('message', {})
                    answer = message.get('content', '')
                
                if answer:
                    print(f"✅ HyperCLOVA X 성공")
                    
                    new_history = request.history + [
                        {"role": "user", "content": request.message},
                        {"role": "assistant", "content": answer}
                    ]
                    
                    return ChatResponse(
                        answer=answer,
                        history=new_history,
                        source="hyperclova_x"
                    )
            
            print(f"❌ HyperCLOVA X 실패: {response.status_code}")
        
        except Exception as e:
            print(f"❌ HyperCLOVA X 오류: {e}")
        
        # === 모든 방법 실패 ===
        print(f"\n❌ 모든 방법 실패")
        
        # 추천 용어 제안
        import re
        query_tokens = set(re.findall(r'[가-힣a-zA-Z0-9]+', request.message.lower()))
        
        similar = []
        for k, v in terms_db.items():
            term_tokens = set(re.findall(r'[가-힣a-zA-Z0-9]+', k.lower()))
            if query_tokens & term_tokens:
                similar.append(v['name'])
        
        similar = list(set(similar))[:3]
        
        answer = f"'{request.message}'에 대한 정확한 설명을 찾지 못했어요. 😢"
        
        if similar:
            answer += f"\n\n혹시 이런 용어를 찾으시나요?\n"
            answer += "\n".join([f"• {s}" for s in similar])
        else:
            answer += "\n\n다른 표현으로 다시 질문해주시겠어요?"
        
        new_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": answer}
        ]
        
        return ChatResponse(
            answer=answer,
            history=new_history,
            source="not_found"
        )
    
    except Exception as e:
        print(f"❌ 전체 오류: {e}")
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

# ================== 서버 실행 (맨 아래 수정) ==================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 IT 용어 챗봇 서버 시작 (비용 효율 버전)")
    print("="*70)
    print(f"📊 로드된 용어: {len(set([v['name'] for v in terms_db.values()]))}개")
    print(f"🔍 검색 키: {len(terms_db)}개 (유사어 포함)")
    print(f"🌐 서버 주소: http://{HOST}:{PORT}")
    print(f"")
    print(f"💡 /chat 폴백 순서 (비용 효율적):")
    print(f"   1️⃣ JSON 검색 (무료, 0.1초)")
    print(f"   2️⃣ 챗봇 빌더 (저렴, 1초)")
    print(f"   3️⃣ HyperCLOVA X (비쌈, 2초)")
    print(f"")
    print(f"🔗 /clova_proxy: HyperCLOVA X 직접 호출")
    print("="*70 + "\n")
    
    uvicorn.run(app, host=HOST, port=PORT)