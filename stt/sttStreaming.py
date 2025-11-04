# -*- coding: utf-8 -*-
"""CLOVA Speech Streaming - 실시간 STT (문장 구분) + Object Storage 업로드"""

import grpc
import json
import pyaudio
import queue
import threading
import os
from dotenv import load_dotenv
from stt.nest import nest_pb2, nest_pb2_grpc
import wave
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
OBS_ENDPOINT = os.getenv("OBS_ENDPOINT", "https://kr.object.ncloudstorage.com")
OBS_ACCESS_KEY = os.getenv("OBS_ACCESS_KEY")
OBS_SECRET_KEY = os.getenv("OBS_SECRET_KEY")
OBS_BUCKET_NAME = os.getenv("OBS_BUCKET_NAME")
OBS_REGION = os.getenv("OBS_REGION", "kr-standard")

# 오디오 설정
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1600


class ClovaSpeechRecognizer:
    """CLOVA Speech Streaming - 실시간 STT (발화자 구분 없음) + Object Storage 업로드"""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.is_processing = False
        self.channel = None
        self.stub = None
        self.full_text = ""
        self.sentences = []
        self.current_sentence = ""
        self.recorded_frames = []
        self.uploaded_file_url = None  # 업로드된 파일 URL 저장

        # Object Storage 클라이언트 초기화
        self.s3_client = None
        self._init_s3_client()

        print("🎙️ CLOVA Speech Streaming - 실시간 STT 활성화")

    # ======================================================
    # Object Storage 초기화
    # ======================================================
    def _init_s3_client(self):
        """Object Storage S3 클라이언트 초기화"""
        try:
            if not all([OBS_ACCESS_KEY, OBS_SECRET_KEY, OBS_BUCKET_NAME]):
                print("⚠️ Object Storage 설정 누락! .env 확인 필요")
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
            print(f"✅ Object Storage 연결 성공!")
            print(f"   📦 Bucket: {OBS_BUCKET_NAME}")
            print(f"   🌏 Endpoint: {OBS_ENDPOINT}")
            print(f"   📍 Region: {OBS_REGION}")

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            print(f"❌ Object Storage 연결 실패 ({code})")
            self.s3_client = None
        except Exception as e:
            print(f"❌ Object Storage 초기화 예외: {type(e).__name__}: {e}")
            self.s3_client = None

    # ======================================================
    # Object Storage 업로드
    # ======================================================
    def upload_to_object_storage(self, local_file_path, object_key=None):
        """
        Object Storage에 파일 업로드 후 CLOVA ExternalURL 규칙에 맞는 URL 반환
        """
        if not self.s3_client:
            return False, "❌ Object Storage 클라이언트가 초기화되지 않음"
        if not os.path.exists(local_file_path):
            return False, f"❌ 파일을 찾을 수 없습니다: {local_file_path}"

        try:
            # object_key 자동 생성
            if not object_key:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.basename(local_file_path)
                object_key = f"stt/input_audio/{timestamp}_{filename}"

            print(f"📤 Object Storage 업로드 시작...")
            print(f"   📁 Local: {local_file_path}")
            print(f"   🔑 Object Key: {object_key}")

            extra_args = {
                "ContentType": "audio/wav",
                "Metadata": {"uploaded-at": datetime.now().isoformat()},
                "ACL": "public-read"
            }

            # 업로드 실행
            self.s3_client.upload_file(
                local_file_path,
                OBS_BUCKET_NAME,
                object_key,
                ExtraArgs=extra_args
            )

            # ✅ CLOVA ExternalURL 규칙에 맞는 URL 생성
            endpoint_domain = OBS_ENDPOINT.replace("https://", "").replace("http://", "")
            file_url = f"https://{OBS_BUCKET_NAME}.{endpoint_domain}/{object_key}"

            print(f"✅ Object Storage 업로드 성공!")
            print(f"   🔗 CLOVA용 URL: {file_url}")
            print(f"   💡 브라우저 접근 URL: {OBS_ENDPOINT}/{OBS_BUCKET_NAME}/{object_key}")

            return True, file_url

        except ClientError as e:
            msg = e.response.get("Error", {}).get("Message", "")
            print(f"❌ ClientError 업로드 실패: {msg}")
            return False, msg
        except Exception as e:
            print(f"❌ 업로드 예외: {type(e).__name__}: {e}")
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
            print("✅ gRPC 연결 성공")
        except Exception as e:
            print(f"❌ gRPC 연결 실패: {e}")

    def disconnect(self):
        """gRPC 채널 종료"""
        if self.channel:
            self.channel.close()
            print("🔌 gRPC 연결 종료")

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
                "gapThreshold": 700,
                "durationThreshold": 8000,
                "syllableThreshold": 80
            }
        }

        print("\n" + "=" * 60)
        print("🔧 실시간 STT Config:")
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
    # 오디오 녹음
    # ======================================================
    def start_recording(self):
        """녹음 시작"""
        self.is_recording = True
        self.recorded_frames = []
        threading.Thread(target=self._record_audio, daemon=True).start()

    def _record_audio(self):
        """오디오 녹음 스레드"""
        print("🎙️ 녹음 시작...")
        audio = pyaudio.PyAudio()

        try:
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            print("✅ 오디오 스트림 열기 성공")

            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
                    self.recorded_frames.append(data)
                except Exception as e:
                    print(f"⚠️ 오디오 읽기 오류: {e}")

        except Exception as e:
            print(f"❌ 오디오 장치 오류: {e}")
        finally:
            if "stream" in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
            print("🎤 녹음 종료")
            self._save_audio_file()

    def _save_audio_file(self):
        """녹음된 오디오 저장 후 Object Storage 업로드"""
        output_path = "recordings/session_audio.wav"
        os.makedirs("recordings", exist_ok=True)

        try:
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.recorded_frames))

            print(f"💾 오디오 저장 완료: {output_path}")

            # Object Storage 업로드
            success, result = self.upload_to_object_storage(output_path)
            if success:
                self.uploaded_file_url = result  # URL 저장
                self.result_queue.put(("audio_uploaded", result))
            else:
                self.result_queue.put(("audio_upload_failed", result))

        except Exception as e:
            msg = f"오디오 저장 실패: {e}"
            print(f"❌ {msg}")
            self.result_queue.put(("audio_upload_failed", msg))

    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        self.is_processing = False
        print("⏹️ 녹음 중지 요청")

    # ======================================================
    # gRPC 요청/응답 처리
    # ======================================================
    def generate_requests(self, language="ko"):
        """gRPC 요청 생성기"""
        yield self.create_config_request(language)
        seq = 0
        while self.is_recording:
            try:
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
            print("🎧 인식 스트림 시작...")

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
                    print(f"\n📝 TEXT: {text} / EPD: {epd} / END: {end_flag}\n")

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
            print("⏳ 오디오 저장 대기 중...")
            time.sleep(0.5)
            self.result_queue.put(("done", None))
            print("🏁 인식 종료")

    # ======================================================
    # 문장 종결 판단
    # ======================================================
    def _is_sentence_end(self, epd_type, text, period_positions):
        """문장 종결 여부 판단"""
        text = text.strip()
        if len(text) < 2:
            return False
        if epd_type in ["periodEpd", "period"]:
            return True
        if period_positions:
            return True
        if text.endswith(('.', '?', '!', '。', '!', '?')):
            return True
        if epd_type in ["gap", "duration", "syllable", "wordEpd"] and len(text) >= 3:
            return True
        return False

    # ======================================================
    # 결과 파일 경로 및 URL 반환
    # ======================================================
    def get_audio_file_path(self):
        """저장된 오디오 파일 경로 반환"""
        return "recordings/session_audio.wav"

    def get_uploaded_file_url(self):
        """Object Storage에 업로드된 파일 URL 반환"""
        return self.uploaded_file_url