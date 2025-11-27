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

# --- STT 발화자 분석 관련 ---
from pydantic import BaseModel
from typing import Optional

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

from chatbot.chatbotSearch.models import ChatRequest as SearchChatRequest, ChatResponse
from chatbot.chatbotFAQ.chatbotFAQMain import ChatRequest as FAQChatRequest, chat as chatbot_faq_endpoint

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
async def chat_endpoint(request: SearchChatRequest):
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
async def faq_endpoint(request: FAQChatRequest):
    """FAQ 챗봇 (IT 용어)"""
    try:
        # FAQ 모듈 데이터 로드 확인
        from chatbot.chatbotFAQ.chatbotFAQMain import load_terms_database
        load_terms_database()
        
        return await chatbot_faq_endpoint(request)
    except Exception as e:
        print(f"❌ FAQ 오류: {e}")
        import traceback
        traceback.print_exc()
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
    - 발화자 구분 없음
    """
    ws_pcm_buffer = bytearray()    
    await websocket.accept()
    recognizer = ClovaSpeechRecognizer()
    is_connected = True
    
    # stop 호출 여부 플래그
    is_stopped = False

    try:
        while is_connected:
            # -------------------------
            # 1) WebSocket 메시지 수신
            # -------------------------
            try:
                msg = await websocket.receive()
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    print("🔌 WebSocket 연결이 이미 종료됨")
                    is_connected = False
                    break
                raise
            
            # 연결 종료 메시지 확인
            if msg["type"] == "websocket.disconnect":
                print("📡 WebSocket disconnect 메시지 수신")
                is_connected = False
                break

            # 텍스트 메시지 처리
            if msg["type"] == "websocket.receive" and msg.get("text"):
                try:
                    data = json.loads(msg["text"])

                    # STT 시작
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

                    # 일시정지
                    elif data["action"] == "pause":
                        if recognizer.pause_recording():
                            await websocket.send_json({
                                "type": "status",
                                "message": "paused",
                                "info": "STT 일시정지됨"
                            })

                    # 재개
                    elif data["action"] == "resume":
                        if recognizer.resume_recording():
                            await websocket.send_json({
                                "type": "status",
                                "message": "resumed",
                                "info": "STT 재개됨"
                            })

                    # # 중지
                    # elif data["action"] == "stop":
                    #     recognizer.stop_recording()
                    #     await websocket.send_json({
                    #         "type": "status",
                    #         "message": "stopping",
                    #         "info": "녹음 중지 중..."
                    #     })

                    elif data["action"] == "stop":
                        if not is_stopped:
                            is_stopped = True
                            recognizer.stop_recording()
                        
                        await websocket.send_json({
                            "type": "status",
                            "message": "stopping",
                            "info": "녹음 중지 중..."
                        })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"text parse error: {str(e)}"
                    })

            # 바이너리(PCM) 데이터 처리
            # WebSocket 통신 중에 byte 깨짐 확인
            if msg["type"] == "websocket.receive" and msg.get("bytes"):
                chunk = msg.get("bytes")
                if chunk:
                    ws_pcm_buffer.extend(chunk)

                    FRAME = 320  # 16kHz 16bit 10ms PCM

                    while len(ws_pcm_buffer) >= FRAME:
                        frame = ws_pcm_buffer[:FRAME]
                        del ws_pcm_buffer[:FRAME]
                        recognizer.add_audio_data(bytes(frame))


            # -------------------------
            # 2) recognizer 결과 처리
            # -------------------------
            try:
                msg_type, payload = recognizer.result_queue.get_nowait()

                # 실시간 인식 결과
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
                    
                    print("\n" + "=" * 80)
                    print("🎤 STT 처리 완료")
                    print("=" * 80)
                    print(f"   📁 파일 URL: {file_url if file_url else '❌ 없음'}")
                    print(f"   📝 전체 텍스트: {recognizer.full_text[:100]}{'...' if len(recognizer.full_text) > 100 else ''}")
                    print(f"   📊 문장 수: {len(recognizer.sentences)}개")
                    print("=" * 80 + "\n")

                    await websocket.send_json({
                        "type": "done",
                        "fullText": recognizer.full_text,
                        "sentences": recognizer.sentences,
                        "sentenceCount": len(recognizer.sentences),
                        "file_url": file_url,
                        "info": "STT 완료. Object Storage 업로드 완료"
                    })

                    break  # 루프 종료

                elif msg_type == "error":
                    await websocket.send_json({
                        "type": "error",
                        "message": payload.get("message", "Unknown error")
                    })

            except queue.Empty:
                await asyncio.sleep(0.005)

    except WebSocketDisconnect:
        print("📡 WebSocket 연결 종료 (클라이언트 측)")
    except Exception as e:
        print(f"❌ WebSocket 예외 발생: {e}")
        import traceback
        print(traceback.format_exc())
        
        # WebSocket이 아직 연결되어 있을 때만 에러 메시지 전송
        try:
            if is_connected:
                await websocket.send_json({"type": "error", "message": str(e)})
        except Exception as send_error:
            print(f"⚠️ 에러 메시지 전송 실패 (이미 연결 종료됨): {send_error}")

    # finally:
    #     recognizer.stop_recording()
    #     recognizer.disconnect()
    #     print("🧹 WebSocket 리소스 정리 완료")

    finally:
        if not is_stopped:
            is_stopped = True
            recognizer.stop_recording()
    
        recognizer.disconnect()
        print("🧹 WebSocket 리소스 정리 완료")

# ======================================================
# 5. 발화자 분석 요청 모델
# ======================================================
class SpeakerAnalysisRequest(BaseModel):
    file_url: str
    language: str = "ko"
    speaker_min: int = 2
    speaker_max: int = 10
    callback_url: Optional[str] = None

# ======================================================
# 6. 발화자 분석 엔드포인트
# ======================================================
@app.post("/api/analyze/object")
async def analyze_from_object_storage(request: SpeakerAnalysisRequest):
    """
    Object Storage URL → CLOVA ExternalURL 호출 (resultToObs=True)
    JSON은 버킷에 자동 저장됨.
    """
    print("\n" + "=" * 80)
    print("CLOVA ExternalURL 비동기 발화자 분석 요청 시작")
    print(f"file_url = {request.file_url}")
    print("=" * 80)

    # WAV 파일명 파싱
    original_filename = request.file_url.split("/")[-1]
    print(f"추출된 파일명: {original_filename}")

    analyzer = ClovaSpeakerAnalyzer()
    lang = convert_language_code(request.language)

    result = analyzer.analyze_audio_url_async(
        file_url=request.file_url,
        language=lang,
        speaker_min=request.speaker_min,
        speaker_max=request.speaker_max,
        callback_url=request.callback_url
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    token = result.get("token")

    print(f"CLOVA 요청 완료! token={token}")
    print("=" * 80)

    return {
        "status": "started",
        "token": token,
        "original_filename": original_filename,
        "message": "CLOVA 비동기 분석 요청 성공"
    }

# ======================================================
# 7. 발화자 분석 엔드포인트
# ======================================================
@app.get("/api/analyze/{token}")
async def get_async_result(token: str, filename: str):
    """
    resultToObs=True → JSON 파일을 직접 Object Storage에서 fetch하여 반환
    """
    print("\n" + "=" * 80)
    print(f"비동기 결과 조회: token={token}, filename={filename}")
    print("=" * 80)

    analyzer = ClovaSpeakerAnalyzer()

    json_data = analyzer.fetch_obs_json(filename, token)

    if "error" in json_data:
        raise HTTPException(status_code=500, detail=json_data["error"])

    result = analyzer.process_obs_json(json_data)
    return result


# ======================================================
# 8. 유틸리티 (다운로드)
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