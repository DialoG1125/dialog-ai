# -*- coding: utf-8 -*-
"""CLOVA Speech Streaming - 실시간 STT (WebSocket PCM 수신) + Object Storage 업로드"""

import grpc
import json
import queue
import threading
import os
from dotenv import load_dotenv
from stt.nest import nest_pb2, nest_pb2_grpc
import wave
import io
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import time

# .env 로드
load_dotenv()

# ======================== 환경 변수 ========================
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY")
CLOVA_HOST = os.getenv("CLOVA_HOST")
CLOVA_PORT = os.getenv("CLOVA_PORT")

# Object Storage 설정
OBS_ENDPOINT = os.getenv("OBS_ENDPOINT")
OBS_ACCESS_KEY = os.getenv("OBS_ACCESS_KEY")
OBS_SECRET_KEY = os.getenv("OBS_SECRET_KEY")
OBS_BUCKET_NAME = os.getenv("OBS_BUCKET_NAME")
OBS_REGION = os.getenv("OBS_REGION")

# 오디오 설정
RATE = int(os.getenv("AUDIO_RATE", "16000"))
CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
CHUNK = int(os.getenv("AUDIO_CHUNK", "4096"))


class ClovaSpeechRecognizer:
    """CLOVA Speech Streaming - WebSocket PCM 수신 + gRPC 전송"""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.is_processing = False
        self.is_paused = False
        self.channel = None
        self.stub = None
        self.full_text = ""
        self.sentences = []
        self.current_sentence = ""
        self.recorded_frames = []
        self.uploaded_file_url = None

        # PCM
        self.raw_buffer = bytearray()
        self.FRAME_BYTES = 320

        # Object Storage 클라이언트 초기화
        self.s3_client = None
        self._init_s3_client()

        print("CLOVA Speech Streaming")

    # ======================================================
    # Object Storage 초기화
    # ======================================================
    def _init_s3_client(self):
        """Object Storage S3 클라이언트 초기화"""
        try:
            if not all([OBS_ACCESS_KEY, OBS_SECRET_KEY, OBS_BUCKET_NAME]):
                print("Object Storage 설정 누락! .env 확인 필요")
                return

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=OBS_ENDPOINT,
                aws_access_key_id=OBS_ACCESS_KEY,
                aws_secret_access_key=OBS_SECRET_KEY,
                region_name=OBS_REGION
            )

            # 버킷 존재 확인
            print(f"🔍 버킷 확인 중: {OBS_BUCKET_NAME}")
            self.s3_client.head_bucket(Bucket=OBS_BUCKET_NAME)
            print(f"Object Storage 연결 성공!")
            print(f"Bucket: {OBS_BUCKET_NAME}")
            print(f"Endpoint: {OBS_ENDPOINT}")
            print(f"Region: {OBS_REGION}")

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            print(f"Object Storage 연결 실패 ({code})")
            self.s3_client = None
        except Exception as e:
            print(f"Object Storage 초기화 예외: {type(e).__name__}: {e}")
            self.s3_client = None

    # ======================================================
    # Object Storage 업로드 (메모리 버퍼)
    # ======================================================
    def upload_audio_buffer(self, audio_buffer):
        """
        메모리 버퍼에서 Object Storage로 직접 업로드 후 CLOVA ExternalURL 규칙에 맞는 URL 반환
        """
        if not self.s3_client:
            return False, "Object Storage 클라이언트가 초기화되지 않음"

        try:
            # object_key 자동 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_path = os.getenv("OBS_UPLOAD_PATH")
            object_key = f"{upload_path}/{timestamp}_session_audio.wav"

            print(f"Object Storage 업로드 시작...")
            print(f"Object Key: {object_key}")

            extra_args = {
                "ContentType": "audio/wav",
                "Metadata": {"uploaded-at": datetime.now().isoformat()},
                "ACL": "public-read"
            }

            # 업로드 실행 (메모리 버퍼에서 직접)
            self.s3_client.upload_fileobj(
                audio_buffer,
                OBS_BUCKET_NAME,
                object_key,
                ExtraArgs=extra_args
            )

            # CLOVA ExternalURL 규칙에 맞는 URL 생성
            endpoint_domain = OBS_ENDPOINT.replace("https://", "").replace("http://", "")
            file_url = f"https://{OBS_BUCKET_NAME}.{endpoint_domain}/{object_key}"

            print(f"Object Storage 업로드 성공!")
            print(f"CLOVA용 URL: {file_url}")
            print(f"브라우저 접근 URL: {OBS_ENDPOINT}/{OBS_BUCKET_NAME}/{object_key}")

            return True, file_url

        except ClientError as e:
            msg = e.response.get("Error", {}).get("Message", "")
            print(f"ClientError 업로드 실패: {msg}")
            return False, msg
        except Exception as e:
            print(f"업로드 예외: {type(e).__name__}: {e}")
            return False, str(e)

    # ======================================================
    # gRPC 연결
    # ======================================================
    def connect(self):
        """gRPC 채널 연결"""
        try:
            self.channel = grpc.secure_channel(
                f"{CLOVA_HOST}:{CLOVA_PORT}",
                grpc.ssl_channel_credentials()
            )
            self.stub = nest_pb2_grpc.NestServiceStub(self.channel)
            print("gRPC 연결 성공")
        except Exception as e:
            print(f"gRPC 연결 실패: {e}")

    def disconnect(self):
        """gRPC 채널 종료"""
        if self.channel:
            self.channel.close()
            print("gRPC 연결 종료")

    # ======================================================
    # 요청 생성
    # ======================================================
    def create_config_request(self, language="ko"):
        """실시간 STT용 Config 생성"""
        config = {
            "transcription": {"language": language},
            "semanticEpd": {
                "skipEmptyText": True,
                "useWordEpd": True,
                "usePeriodEpd": True,
                "gapThreshold": int(os.getenv("STT_GAP_THRESHOLD", "700")),
                "durationThreshold": int(os.getenv("STT_DURATION_THRESHOLD", "8000")),
                "syllableThreshold": int(os.getenv("STT_SYLLABLE_THRESHOLD", "80"))
            }
        }

        print("\n" + "=" * 60)
        print("실시간 STT Config:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print("=" * 60 + "\n")

        nest_config = nest_pb2.NestConfig(config=json.dumps(config))
        return nest_pb2.NestRequest(type=nest_pb2.CONFIG, config=nest_config)

    def create_data_request(self, audio_chunk, ep_flag=False, seq_id=0):
        """오디오 데이터 요청 생성"""
        extra = {"epFlag": ep_flag, "seqId": seq_id}
        nest_data = nest_pb2.NestData(
            chunk=audio_chunk,
            extra_contents=json.dumps(extra)
        )
        return nest_pb2.NestRequest(type=nest_pb2.DATA, data=nest_data)

    # ======================================================
    # WebSocket에서 받은 PCM 데이터 처리
    # ======================================================
    def add_audio_data(self, pcm_data: bytes):
            """
            WebSocket에서 받은 PCM(Int16) 데이터를
            10ms(320bytes) 단위로 정확히 잘라 gRPC로 전달
            """
            if not self.is_paused and self.is_recording:
                # 1) raw 버퍼에 누적
                self.raw_buffer.extend(pcm_data)

                # 2) 10ms 단위로 자르기
                while len(self.raw_buffer) >= self.FRAME_BYTES:
                    frame = self.raw_buffer[:self.FRAME_BYTES]
                    del self.raw_buffer[:self.FRAME_BYTES]

                    # gRPC로 보낼 큐에 추가 (정확한 10ms 프레임)
                    self.audio_queue.put(bytes(frame))

                    # WAV 저장할 프레임도 동일
                    self.recorded_frames.append(bytes(frame))

    # ======================================================
    # 녹음 제어
    # ======================================================
    def start_recording(self):
        """녹음 시작 (WebSocket 수신 대기)"""
        self.is_recording = True
        self.recorded_frames = []
        print("WebSocket PCM 수신 시작...")

    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        self.is_processing = False
        print("녹음 중지 요청")
        self._upload_audio_to_storage()

    def pause_recording(self):
        """녹음 일시정지"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            print("STT 일시정지")
            return True
        return False

    def resume_recording(self):
        """녹음 재개"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            print("STT 재개")
            return True
        return False

    def _upload_audio_to_storage(self):
        """녹음된 오디오를 메모리에서 직접 Object Storage에 업로드 (오류 시 무시)"""
        # 녹음된 데이터가 없으면 업로드하지 않음
        if not self.recorded_frames or len(self.recorded_frames) == 0:
            print("녹음된 오디오 데이터가 없습니다. 업로드 건너뜀")
            self.uploaded_file_url = None
            return

        try:
            # 메모리에 WAV 파일 생성
            audio_buffer = io.BytesIO()
            
            with wave.open(audio_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # Int16 = 2 bytes
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.recorded_frames))

            audio_buffer.seek(0)
            print(f"오디오 메모리 버퍼 생성 완료")

            # Object Storage 업로드 (오류 발생해도 무시)
            try:
                success, result = self.upload_audio_buffer(audio_buffer)
                if success:
                    self.uploaded_file_url = result
                    self.result_queue.put(("audio_uploaded", result))
                else:
                    print(f"Object Storage 업로드 실패 (무시됨): {result}")
                    self.uploaded_file_url = None
            except Exception as upload_error:
                print(f"Object Storage 업로드 예외 (무시됨): {upload_error}")
                self.uploaded_file_url = None

        except Exception as e:
            # 오디오 버퍼 생성 실패해도 무시
            msg = f"오디오 버퍼 생성 실패 (무시됨): {e}"
            print(f"{msg}")
            self.uploaded_file_url = None

    # ======================================================
    # gRPC 요청/응답 처리
    # ======================================================
    def generate_requests(self, language="ko"):
        """gRPC 요청 생성기 (WebSocket에서 받은 PCM 사용)"""
        yield self.create_config_request(language)
        seq = 0
        while self.is_recording:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                chunk = self.audio_queue.get(timeout=0.1)
                yield self.create_data_request(chunk, False, seq)
                seq += 1
            except queue.Empty:
                continue
        yield self.create_data_request(b"", True, seq)

    def start_recognition(self, language="ko"):
        """STT 인식 시작"""
        self.is_processing = True
        threading.Thread(
            target=self._process_recognition,
            args=(language,),
            daemon=True
        ).start()

    def _process_recognition(self, language="ko"):
        """STT 응답 처리"""
        try:
            metadata = (("authorization", f"Bearer {CLOVA_SECRET_KEY}"),)
            responses = self.stub.recognize(
                self.generate_requests(language),
                metadata=metadata,
                timeout=600
            )
            print("인식 스트림 시작...")

            for response in responses:
                contents = response.contents
                result = json.loads(contents)
                rtype = result.get("responseType", [])

                if "config" in rtype:
                    self.result_queue.put(("config", result.get("config", {})))

                elif "transcription" in rtype:
                    t = result["transcription"]
                    text = t.get("text", "")
                    epd = t.get("epdType", "")
                    conf = t.get("confidence", 0)
                    pos = t.get("position", 0)
                    pp = t.get("periodPositions", [])
                    if not text:
                        continue

                    end_flag = self._is_sentence_end(epd, text, pp)
                    print(f"\nTEXT: {text} / EPD: {epd} / END: {end_flag}\n")

                    if end_flag:
                        self.sentences.append(text)
                        self.full_text += text + " "

                    send_data = {
                        "type": "transcription",
                        "text": text,
                        "isSentenceEnd": end_flag,
                        "confidence": conf,
                        "position": pos,
                        "epdType": epd,
                        "periodPositions": pp
                    }
                    self.result_queue.put(("data", send_data))

        except grpc.RpcError as e:
            self.result_queue.put(("error", {"code": str(e.code()), "message": e.details()}))
        finally:
            print("오디오 업로드 대기 중...")
            time.sleep(0.5)
            self.result_queue.put(("done", None))
            print("인식 종료")

    # ======================================================
    # 문장 종결 판단
    # ======================================================
    # def _is_sentence_end(self, epd_type, text, period_positions):
    #     """문장 종결 여부 판단"""
    #     text = text.strip()
    #     if len(text) < 2:
    #         return False
    #     if epd_type in ["periodEpd", "period"]:
    #         return True
    #     if period_positions:
    #         return True
    #     if text.endswith(('.', '?', '!', '。', '!', '?')):
    #         return True
    #     if epd_type in ["gap", "duration", "syllable", "wordEpd"] and len(text) >= 3:
    #         return True
    #     return False

    def _is_sentence_end(self, epd_type, text, period_positions):
        """문장 종결 여부 판단 - 개선 버전"""
        text = text.strip()
        
        # 너무 짧은 텍스트는 문장으로 인정하지 않음
        if len(text) < 5:  # 3 → 5로 변경
            return False
        
        # 1순위: 명확한 문장 종결 표시
        if epd_type in ["periodEpd", "period"]:
            return True
        
        # 2순위: 마침표 위치 정보가 있는 경우
        if period_positions:
            return True
        
        # 3순위: 문장 부호로 끝나는 경우
        if text.endswith(('.', '?', '!', '。', '!', '?')):
            return True
        
        # 4순위: 충분히 긴 문장 + 명확한 끊김 감지
        # 조건 강화: 최소 10글자 이상 + duration/syllable 끊김만 인정
        if len(text) >= 10 and epd_type in ["duration", "syllable"]:
            return True
        
        # 5순위: 매우 긴 문장은 강제로 끊음 (20글자 이상)
        if len(text) >= 20 and epd_type in ["gap", "wordEpd"]:
            return True
        
        return False

    # ======================================================
    # 결과 URL 반환
    # ======================================================
    def get_uploaded_file_url(self):
        """Object Storage에 업로드된 파일 URL 반환"""
        return self.uploaded_file_url