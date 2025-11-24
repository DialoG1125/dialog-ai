# -*- coding: utf-8 -*-
"""
CLOVA Speech API + AI 요약/할일 + 챗봇 통합 FastAPI 서버 (DialoG)
- 실시간 STT / 발화자 분석
- AI 요약 / 할 일 생성
- 회의록 검색 챗봇 / FAQ 챗봇
"""

import sys
from pathlib import Path
import os
import asyncio
import json
import queue
import uvicorn

# ========== 경로 설정 (챗봇 및 STT 모듈 호환성) ==========
# stt/nest 폴더 등을 모듈 경로로 인식시키기 위해 추가
sys.path.insert(0, str(Path(__file__).parent / "stt" / "nest"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# --- 내부 모듈: STT 관련 ---
from stt.sttStreaming import ClovaSpeechRecognizer
from stt.sttSpeaker import ClovaSpeakerAnalyzer, convert_language_code

# --- 내부 모듈: AI 요약/할일 관련 ---
from summary.summary_service import (
    create_summary, 
    SummaryRequest, 
    SummaryResponse
)
from summary.action_service import (
    generate_all_actions_service, 
    ActionRequest, 
    ActionResponse
)

# --- 내부 모듈: 챗봇 관련 ---
# chatbotSearchMain에서 chat_endpoint 함수 import
from chatbot.chatbotSearch.chatbotSearchMain import chat as chatbot_chat_endpoint
from chatbot.chatbotSearch.models import ChatRequest, ChatResponse

# chatbotFAQMain에서 FAQ chat_endpoint 함수 import  
from chatbot.chatbotFAQ.chatbotFAQMain import chat as chatbot_faq_endpoint


# ======================================================
# FastAPI 기본 설정
# ======================================================
app = FastAPI(title="Dialog Integrated API Server", version="10.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트엔드 연결 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# 1. 기본 정보 및 헬스 체크
# ======================================================
@app.get("/")
async def root():
    """API 정보"""
    return {
        "status": "Dialog Integrated API Server Running",
        "version": "10.0",
        "description": "STT + Speaker Analysis + AI Summary/Actions + Chatbot",
        "endpoints": {
            "stt_websocket": "/ws/realtime",
            "speaker_analyze": "/api/analyze/object",
            "ai_summary": "/summary/generate",
            "ai_actions": "/actions/generate",
            "chatbot_search": "/api/chat",
            "chatbot_faq": "/api/faq",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "service": "Dialog API"}


# ======================================================
# 2. 챗봇 엔드포인트
# ======================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """회의록 검색 챗봇"""
    try:
        result = await chatbot_chat_endpoint(request)
        
        # [옵션] 불필요한 history 데이터 제외 후 반환
        result.history = None
        
        print(f"🔹 챗봇 응답 완료: {result.model_dump(exclude_none=True)}")
        return result
    except Exception as e:
        print(f"❌ 챗봇 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/faq", response_model=ChatResponse)
async def faq_endpoint(request: ChatRequest):
    """FAQ 챗봇 (IT 용어)"""
    try:
        return await chatbot_faq_endpoint(request)
    except Exception as e:
        print(f"❌ FAQ 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# 3. AI 요약 및 할 일 생성 엔드포인트
# ======================================================
@app.post("/summary/generate", response_model=SummaryResponse)
async def summarize_meeting(request: SummaryRequest):
    """AI 요약 생성"""
    try:
        summary_data = await create_summary(request)
        return SummaryResponse(success=True, summary=summary_data)
    except Exception as e:
        print(f"❌ 요약 생성 오류: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"요약 생성 실패: {str(e)}")


@app.post("/actions/generate", response_model=ActionResponse)
async def generate_all_actions(request: ActionRequest):
    """AI 할 일 생성"""
    try:
        actions_list = await generate_all_actions_service(request)
        return ActionResponse(success=True, actions=actions_list)
    except Exception as e:
        print(f"❌ 액션 아이템 생성 오류: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"액션 아이템 생성 실패: {str(e)}")


# ======================================================
# 4. 실시간 STT WebSocket
# ======================================================
@app.websocket("/ws/realtime")
async def websocket_realtime_stt(websocket: WebSocket):
    """
    실시간 STT WebSocket 엔드포인트
    - gRPC 기반 CLOVA Speech Streaming
    - 실시간 텍스트 변환 및 Object Storage 업로드
    """
    await websocket.accept()
    recognizer = ClovaSpeechRecognizer()

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                data = json.loads(msg)

                # 🎙️ 녹음 시작
                if data["action"] == "start":
                    language = data.get("language", "ko")
                    recognizer.connect()
                    recognizer.start_recording()
                    recognizer.start_recognition(language)

                    await websocket.send_json({
                        "type": "status",
                        "message": "recording",
                        "info": "STT 시작 (녹음 및 업로드 준비 중)"
                    })

                # ⏸️ 녹음 일시정지
                elif data["action"] == "pause":
                    if recognizer.pause_recording():
                        await websocket.send_json({
                            "type": "status",
                            "message": "paused",
                            "info": "STT 일시정지됨"
                        })

                # ▶️ 녹음 재개
                elif data["action"] == "resume":
                    if recognizer.resume_recording():
                        await websocket.send_json({
                            "type": "status",
                            "message": "resumed",
                            "info": "STT 재개됨"
                        })

                # 🛑 녹음 중지
                elif data["action"] == "stop":
                    recognizer.stop_recording()
                    await websocket.send_json({
                        "type": "status",
                        "message": "stopping",
                        "info": "녹음 중지 중..."
                    })

            except asyncio.TimeoutError:
                pass

            # 결과 처리
            try:
                msg_type, payload = recognizer.result_queue.get_nowait()

                # 실시간 인식 데이터
                if msg_type == "data":
                    await websocket.send_json(payload)

                # 업로드 완료
                elif msg_type == "audio_uploaded":
                    await websocket.send_json({
                        "type": "audio_uploaded",
                        "file_url": payload,
                        "info": "Object Storage 업로드 완료"
                    })

                # 업로드 실패
                elif msg_type == "audio_upload_failed":
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Object Storage 업로드 실패: {payload}"
                    })

                # STT 종료
                elif msg_type == "done":
                    file_url = recognizer.get_uploaded_file_url()
                    
                    await websocket.send_json({
                        "type": "done",
                        "fullText": recognizer.full_text,
                        "sentences": recognizer.sentences,
                        "sentenceCount": len(recognizer.sentences),
                        "file_url": file_url,
                        "info": "STT 완료. Object Storage 업로드 완료"
                    })
                    
                    # 자동으로 발화자 분석 시작 (file_url이 있는 경우)
                    if file_url:
                        print(f"\n🚀 자동 발화자 분석 시작: {file_url}")
                        analyzer = ClovaSpeakerAnalyzer()
                        analysis_result = analyzer.analyze_audio_url_async(
                            file_url=file_url,
                            language="ko-KR",
                            speaker_min=-1,
                            speaker_max=-1
                        )
                        
                        if "token" in analysis_result:
                            await websocket.send_json({
                                "type": "speaker_analysis_started",
                                "token": analysis_result.get("token"),
                                "file_url": file_url,
                                "info": "발화자 분석 시작됨"
                            })
                        else:
                            await websocket.send_json({
                                "type": "speaker_analysis_error",
                                "error": analysis_result.get("error", "Unknown error"),
                                "info": "발화자 분석 시작 실패"
                            })
                    
                    break

                # STT 에러
                elif msg_type == "error":
                    await websocket.send_json({
                        "type": "error",
                        "message": payload.get("message", "Unknown error")
                    })

            except queue.Empty:
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("📡 WebSocket 연결 종료 (클라이언트 측)")
    except Exception as e:
        print(f"❌ WebSocket 예외 발생: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        recognizer.stop_recording()
        recognizer.disconnect()
        print("🧹 WebSocket 리소스 정리 완료")


# ======================================================
# 5. 발화자 분석 엔드포인트 (Object Storage & Local)
# ======================================================
@app.post("/api/analyze/object")
async def analyze_from_object_storage(
    file_url: str,
    language: str = "ko",
    speaker_min: int = -1,
    speaker_max: int = -1,
    callback_url: str = None
):
    """Object Storage URL 기반 비동기 발화자 분석"""
    try:
        print(f"\n🎧 CLOVA ExternalURL 분석 요청: {file_url}")
        
        analyzer = ClovaSpeakerAnalyzer()
        lang = convert_language_code(language)

        result = analyzer.analyze_audio_url_async(
            file_url=file_url,
            language=lang,
            speaker_min=speaker_min,
            speaker_max=speaker_max,
            callback_url=callback_url
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "status": "started",
            "token": result.get("token"),
            "file_url": file_url,
            "message": "CLOVA 비동기 분석 요청 성공"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze/{token}")
async def get_async_result(token: str):
    """비동기 발화자 분석 결과 조회"""
    analyzer = ClovaSpeakerAnalyzer()
    result = analyzer.get_async_result(token)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # 완료 시 로그 출력
    if result.get("status") == "COMPLETED":
        print(f"\n🎉 분석 완료! (Token: {token})")
        print(f"👥 화자 수: {result.get('totalSpeakers', 0)}명")

    return result


@app.post("/api/analyze")
async def analyze_speaker_sync(
    language: str = "ko",
    speaker_min: int = -1,
    speaker_max: int = -1
):
    """로컬 파일 동기 분석"""
    path = "recordings/session_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="로컬 오디오 파일 없음")

    analyzer = ClovaSpeakerAnalyzer()
    result = analyzer.analyze_audio_file(
        audio_file_path=path,
        language=convert_language_code(language),
        speaker_min=speaker_min,
        speaker_max=speaker_max
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/analyze/async")
async def analyze_speaker_async(
    language: str = "ko",
    speaker_min: int = -1,
    speaker_max: int = -1,
    callback_url: str = None
):
    """로컬 파일 비동기 분석"""
    path = "recordings/session_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="로컬 오디오 파일 없음")

    analyzer = ClovaSpeakerAnalyzer()
    result = analyzer.analyze_audio_file_async(
        audio_file_path=path,
        language=convert_language_code(language),
        speaker_min=speaker_min,
        speaker_max=speaker_max,
        callback_url=callback_url
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# ======================================================
# 6. 유틸리티 (다운로드)
# ======================================================
@app.get("/api/download/audio")
async def download_audio():
    """녹음된 오디오 파일 다운로드"""
    path = "recordings/session_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    return FileResponse(path=path, media_type="audio/wav", filename="session_audio.wav")


# ======================================================
# 서버 실행
# ======================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Dialog Integrated Server 시작! (STT + AI + Chatbot)")
    print("=" * 80)
    print("📡 [STT & Analysis]")
    print("   • ws://localhost:8000/ws/realtime            → 실시간 STT")
    print("   • POST /api/analyze/object                   → 발화자 분석 (URL)")
    print("   • GET  /api/analyze/{token}                  → 분석 결과 조회")
    print("📡 [AI Generation]")
    print("   • POST /summary/generate                     → AI 요약")
    print("   • POST /actions/generate                     → AI 할 일")
    print("📡 [Chatbot]")
    print("   • POST /api/chat                             → 회의록 검색 챗봇")
    print("   • POST /api/faq                              → FAQ 챗봇")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)