"""CLOVA Speech - 발화자 구분 (External URL + Async + 화자 통계)"""

import requests
import json
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY")
CLOVA_INVOKE_URL = os.getenv("CLOVA_INVOKE_URL")  # 예: https://clovaspeech-gw.ncloud.com/external/v1/xxxx


class ClovaSpeakerAnalyzer:
    """CLOVA Speech - ExternalURL 비동기 발화자 구분"""

    def __init__(self):
        self.secret_key = CLOVA_SECRET_KEY
        self.invoke_url = CLOVA_INVOKE_URL
        print("🎤 CLOVA Speech - ExternalURL Async 발화자 분석기 초기화")

    # ------------------ 비동기 발화자 구분 ------------------
    def analyze_audio_url_async(self, file_url, language="ko-KR",
                                speaker_min=-1, speaker_max=-1,
                                callback_url=None):
        """
        Object Storage URL을 CLOVA로 비동기 전송
        """
        print(f"\n{'='*70}")
        print(f"🌐 CLOVA ExternalURL Async 호출")
        print(f"🎧 대상 URL: {file_url}")
        print(f"🗣 언어: {language}")
        print(f"{'='*70}\n")

        params = {
            "url": file_url,
            "language": language,
            "completion": "async",
            "wordAlignment": True,
            "fullText": True,
            "noiseFiltering": True,
            "resultToObs": True,  # 비동기 시 필수!
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

            if response.status_code == 200:
                data = response.json()
                token = data.get("token")
                print(f"✅ 비동기 작업 시작됨 | token: {token}")
                return {"token": token, "status": data.get("result", "STARTED")}
            else:
                return {"error": f"API 오류 {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    # ------------------ 비동기 결과 조회 ------------------
    def get_async_result(self, token):
        """비동기 작업 결과 조회"""
        headers = {"X-CLOVASPEECH-API-KEY": self.secret_key}
        try:
            response = requests.get(
                f"{self.invoke_url}/recognizer/{token}",
                headers=headers,
                timeout=30
            )
            if response.status_code != 200:
                return {"error": f"조회 실패: {response.status_code} {response.text}"}

            result = response.json()
            status = result.get("result")

            if status == "COMPLETED":
                print("✅ CLOVA 비동기 분석 완료")
                return self._process_result(result)
            elif status == "FAILED":
                print("❌ CLOVA 분석 실패")
                return {"error": "CLOVA 분석 실패", "message": result.get("message")}
            else:
                # 진행 중일 때도 로그 출력
                progress = result.get("progress", 0)
                print(f"⏳ CLOVA 분석 진행 중... ({progress}%)")
                return {
                    "status": status,
                    "progress": progress,
                    "message": result.get("message", "처리 중...")
                }
        except Exception as e:
            return {"error": f"조회 중 오류: {e}"}

    # ------------------ 결과 정리 ------------------
    def _process_result(self, result):
        """CLOVA Speech 결과 정리 + 화자별 통계 계산"""
        text = result.get("text", "")
        segments = result.get("segments", [])
        speakers = result.get("speakers", [])

        # --- 화자별 통계 계산 ---
        speaker_stats = {}
        total_talk_time = 0

        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            dur = max(0, end - start)  # ms 단위
            spk = seg.get("speaker", {})
            name = spk.get("name", "Unknown")
            label = spk.get("label", -1)

            if label not in speaker_stats:
                speaker_stats[label] = {"name": name, "time": 0, "sentences": []}
            speaker_stats[label]["time"] += dur
            speaker_stats[label]["sentences"].append(seg)
            total_talk_time += dur

        # --- 비율 계산 ---
        for label, info in speaker_stats.items():
            ratio = (info["time"] / total_talk_time * 100) if total_talk_time > 0 else 0
            speaker_stats[label]["ratio"] = round(ratio, 2)

        summary = {
            "success": True,
            "text": text,
            "totalSpeakers": len(speakers),
            "speakers": speakers,
            "segments": segments,
            "speakerStats": speaker_stats,
            "totalTalkTimeSec": round(total_talk_time / 1000, 2)
        }

        return summary

    # ------------------ 특정 발화자 필터링 ------------------
    def filter_by_speaker(self, result, speaker_name):
        """
        특정 화자의 문장만 필터링
        """
        segments = result.get("segments", [])
        filtered = [s for s in segments if s.get("speaker", {}).get("name") == speaker_name]
        return {
            "speaker": speaker_name,
            "count": len(filtered),
            "sentences": filtered
        }


# ------------------ 언어 코드 변환 ------------------
def convert_language_code(short_code):
    mapping = {
        "ko": "ko-KR",
        "en": "en-US",
        "ja": "ja-JP",
        "zh-cn": "zh-CN",
        "zh": "zh-CN"
    }
    return mapping.get(short_code, short_code)