# -*- coding: utf-8 -*-
"""CLOVA Speech - 발화자 구분 (External URL + Async + OBS JSON Polling, boto3 기반)"""

import requests
import json
import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

load_dotenv()

# CLOVA API
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY")
CLOVA_INVOKE_URL = os.getenv("CLOVA_INVOKE_URL")

# Object Storage 정보
OBS_BUCKET = os.getenv("OBS_BUCKET_NAME")
OBS_ENDPOINT = os.getenv("OBS_ENDPOINT")
OBS_ACCESS_KEY = os.getenv("OBS_ACCESS_KEY")
OBS_SECRET_KEY = os.getenv("OBS_SECRET_KEY")


class ClovaSpeakerAnalyzer:
    """CLOVA Speech - ExternalURL 비동기 발화자 구분 (OBS JSON Polling + boto3 버전)"""

    def __init__(self):
        self.secret_key = CLOVA_SECRET_KEY
        self.invoke_url = CLOVA_INVOKE_URL
        print("🎤 CLOVA Speech - ExternalURL Async 발화자 분석기 초기화")

    # ------------------------------------------------------------
    # 1) CLOVA로 비동기 분석 요청 (ExternalURL async)
    # ------------------------------------------------------------
    def analyze_audio_url_async(self, file_url, language="ko-KR",
                                speaker_min=2, speaker_max=10,
                                callback_url=None):

        print("\n" + "=" * 70)
        print("🌐 CLOVA ExternalURL Async 호출")
        print(f"🎧 대상 URL: {file_url}")
        print(f"🗣 언어: {language}")
        print("=" * 70 + "\n")

        params = {
            "url": file_url,
            "language": language,
            "completion": "async",
            "wordAlignment": True,
            "fullText": True,
            "noiseFiltering": True,
            "resultToObs": True,
            "diarization": {
                "enable": True,
                "speakerCountMin": speaker_min,
                "speakerCountMax": speaker_max
            },
            "sed": {"enable": True}
        }

        if callback_url:
            params["callback"] = callback_url

        headers = {
            "X-CLOVASPEECH-API-KEY": self.secret_key,
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                f"{self.invoke_url}/recognizer/url",
                headers=headers,
                json=params,
                timeout=30
            )

            print(f"🔍 CLOVA API 응답 상태: {response.status_code}")

            if response.status_code != 200:
                return {"error": f"{response.status_code} {response.text}"}

            data = response.json()

            print("🔍 CLOVA 응답:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            return {
                "status": data.get("result"),
                "token": data.get("token")
            }

        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------
    # 2) OBS JSON 파일 다운로드 (boto3 기반, private도 접근 가능)
    # ------------------------------------------------------------
    def fetch_obs_json(self, original_filename, token):
        """
        resultToObs=True 모드에서 생성된 JSON을
        Object Storage에서 boto3로 안전하게 가져온다.
        (파일이 private이어도 ACCESS KEY로 접근 가능)
        """

        key = f"stt/output_result/{original_filename}_{token}.json"
        print(f"📥 OBS JSON 가져오기 (boto3) → bucket={OBS_BUCKET}, key={key}")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=OBS_ACCESS_KEY,
            aws_secret_access_key=OBS_SECRET_KEY,
            endpoint_url=OBS_ENDPOINT
        )

        try:
            obj = s3.get_object(Bucket=OBS_BUCKET, Key=key)
            data = obj["Body"].read().decode("utf-8")
            print("✅ OBS JSON 다운로드 성공!")
            return json.loads(data)

        except ClientError as e:
            print(f"❌ OBS JSON 다운로드 실패 (boto3): {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------
    # 3) OBS JSON 구조 정리
    # ------------------------------------------------------------
    def process_obs_json(self, result):
        """JSON을 정리하여 텍스트·화자 정보·통계를 반환"""

        text = result.get("text", "")
        segments = result.get("segments", [])
        speakers = result.get("speakers", [])

        print("🔍 OBS JSON 파싱 결과:")
        print(" - text 길이:", len(text))
        print(" - segments:", len(segments))
        print(" - speakers:", len(speakers))

        speaker_stats = {}
        total_talk_time = 0

        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            dur = max(0, end - start)

            spk = seg.get("speaker", {})
            label = spk.get("label", -1)
            name = spk.get("name", f"Speaker{label}")

            if label not in speaker_stats:
                speaker_stats[label] = {
                    "name": name,
                    "time": 0,
                    "sentences": []
                }

            speaker_stats[label]["time"] += dur
            speaker_stats[label]["sentences"].append(seg)
            total_talk_time += dur

        # 비율 계산
        for label, info in speaker_stats.items():
            ratio = (info["time"] / total_talk_time * 100) if total_talk_time else 0
            speaker_stats[label]["ratio"] = round(ratio, 2)

        return {
            "success": True,
            "text": text,
            "speakers": speakers,
            "segments": segments,
            "speakerStats": speaker_stats,
            "totalSpeakers": len(speakers),
            "totalTalkTimeSec": round(total_talk_time / 1000, 2)
        }


# ------------------------------------------------------------
# 언어 코드 변환
# ------------------------------------------------------------
def convert_language_code(short_code):
    mapping = {
        "ko": "ko-KR",
        "en": "en-US",
        "ja": "ja-JP",
        "zh-cn": "zh-CN",
        "zh": "zh-CN"
    }
    return mapping.get(short_code, short_code)
