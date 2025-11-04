# -*- coding: utf-8 -*-
"""CLOVA Speech API - FastAPI 서버 (실시간 STT + Object Storage + 비동기 발화자 분석)"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import json
import queue
import uvicorn
import os

# 내부 모듈
from stt.sttStreaming import ClovaSpeechRecognizer
from stt.sttSpeaker import ClovaSpeakerAnalyzer, convert_language_code


# ======================================================
# FastAPI 기본 설정
# ======================================================
app = FastAPI(title="CLOVA Speech API (DialoG)", version="8.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트엔드 연결 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# 기본 정보
# ======================================================
@app.get("/")
async def root():
    """API 정보"""
    return {
        "status": "CLOVA Speech API Server (DialoG)",
        "version": "8.1",
        "description": "실시간 STT + Object Storage + CLOVA ExternalURL 비동기 발화자 구분",
        "endpoints": {
            "websocket": "/ws/realtime",
            "analyze_object": "/api/analyze/object",
            "analyze_local": "/api/analyze",
            "analyze_async": "/api/analyze/async",
            "analyze_result": "/api/analyze/{token}",
            "download_audio": "/api/download/audio",
            "health": "/api/health"
        },
        "workflow": [
            "1️⃣ 실시간 STT → ws://localhost:8000/ws/realtime",
            "2️⃣ Object Storage 업로드 (자동)",
            "3️⃣ 발화자 분석 요청 → POST /api/analyze/object",
            "4️⃣ 비동기 결과 조회 → GET /api/analyze/{token}"
        ]
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "service": "CLOVA Speech API"}


# ======================================================
# WebSocket: 실시간 STT
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
                    # 업로드된 파일 URL 가져오기
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
                                "info": "발화자 분석 시작됨. /api/analyze/{token}으로 결과 조회 가능"
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
# REST API: 로컬 분석
# ======================================================
@app.post("/api/analyze")
async def analyze_speaker_sync(
    language: str = "ko",
    speaker_min: int = -1,
    speaker_max: int = -1
):
    """로컬 저장된 오디오 파일 발화자 구분 분석 (동기)"""
    path = "recordings/session_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="로컬 오디오 파일을 찾을 수 없습니다.")

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


# ======================================================
# REST API: 비동기 로컬 분석
# ======================================================
@app.post("/api/analyze/async")
async def analyze_speaker_async(
    language: str = "ko",
    speaker_min: int = -1,
    speaker_max: int = -1,
    callback_url: str = None
):
    """로컬 오디오 파일 발화자 구분 분석 (비동기)"""
    path = "recordings/session_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="로컬 오디오 파일을 찾을 수 없습니다.")

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
# REST API: Object Storage URL 기반 분석
# ======================================================
@app.post("/api/analyze/object")
async def analyze_from_object_storage(
    file_url: str,
    language: str = "ko",
    speaker_min: int = -1,
    speaker_max: int = -1,
    callback_url: str = None
):
    """
    Object Storage URL을 CLOVA ExternalURL API로 전달하여 비동기 발화자 구분 수행
    """
    try:
        print("\n" + "=" * 80)
        print("🎧 CLOVA ExternalURL 비동기 발화자 분석 요청 시작")
        print(f"🔗 파일 URL: {file_url}")
        print(f"🗣 언어 코드: {language}")
        print(f"👥 화자 범위: {speaker_min} ~ {speaker_max}")
        print("=" * 80)

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

        print(f"✅ CLOVA 비동기 요청 완료 → token: {result.get('token')}")
        print("=" * 80 + "\n")

        return {
            "status": "started",
            "token": result.get("token"),
            "file_url": file_url,
            "message": "CLOVA 비동기 분석 요청 성공"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# REST API: 비동기 결과 조회
# ======================================================
@app.get("/api/analyze/{token}")
async def get_async_result(token: str):
    """CLOVA 비동기 발화자 분석 결과 조회"""
    analyzer = ClovaSpeakerAnalyzer()
    result = analyzer.get_async_result(token)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # 결과가 완료되었을 때 터미널에 예쁘게 출력
    if result.get("success") or result.get("status") == "COMPLETED":
        print("\n" + "=" * 80)
        print("🎉 CLOVA 발화자 분석 완료!")
        print("=" * 80)
        
        # 전체 텍스트
        if "text" in result:
            print(f"\n📝 전체 텍스트:")
            print(f"   {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        
        # 화자 정보
        total_speakers = result.get("totalSpeakers", 0)
        print(f"\n👥 총 화자 수: {total_speakers}명")
        
        # 화자별 통계
        if "speakerStats" in result:
            print(f"\n📊 화자별 통계:")
            for label, info in result["speakerStats"].items():
                name = info.get("name", f"화자{label}")
                time_sec = info.get("time", 0) / 1000  # ms to sec
                ratio = info.get("ratio", 0)
                sentence_count = len(info.get("sentences", []))
                print(f"   • {name}: {time_sec:.1f}초 ({ratio:.1f}%) - {sentence_count}개 문장")
        
        # 총 대화 시간
        if "totalTalkTimeSec" in result:
            total_time = result["totalTalkTimeSec"]
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(f"\n⏱️  총 대화 시간: {minutes}분 {seconds}초")
        
        # 문장 미리보기
        if "segments" in result and len(result["segments"]) > 0:
            print(f"\n💬 발화 미리보기 (처음 3개):")
            for i, seg in enumerate(result["segments"][:3], 1):
                speaker = seg.get("speaker", {}).get("name", "Unknown")
                text = seg.get("text", "")
                start = seg.get("start", 0) / 1000  # ms to sec
                print(f"   {i}. [{start:.1f}초] {speaker}: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        print("=" * 80 + "\n")
    
    return result


# ======================================================
# 오디오 파일 다운로드
# ======================================================
@app.get("/api/download/audio")
async def download_audio():
    """녹음된 오디오 파일 다운로드 (테스트용)"""
    path = "recordings/session_audio.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    return FileResponse(path=path, media_type="audio/wav", filename="session_audio.wav")


# ======================================================
# 서버 실행
# ======================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 CLOVA Speech API 서버 시작! (STT + Object Storage + ExternalURL 분석)")
    print("=" * 80)
    print("📡 주요 엔드포인트:")
    print("   • ws://localhost:8000/ws/realtime   → 실시간 STT")
    print("   • POST /api/analyze/object          → Object Storage 기반 발화자 분석")
    print("   • GET  /api/analyze/{token}         → 비동기 결과 조회")
    print("   • GET  /api/health                  → 서버 상태 확인")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)