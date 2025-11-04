"""
마이크 장치 확인 스크립트
실행 방법: python check_microphone.py
"""

import pyaudio
import sys

def check_audio_devices():
    """사용 가능한 오디오 장치 목록 출력"""
    print("\n" + "="*80)
    print("🎤 오디오 장치 검색 중...")
    print("="*80 + "\n")
    
    try:
        audio = pyaudio.PyAudio()
        
        # 기본 장치 정보
        try:
            default_input = audio.get_default_input_device_info()
            print("✅ 기본 입력 장치 발견:")
            print(f"   이름: {default_input['name']}")
            print(f"   인덱스: {default_input['index']}")
            print(f"   채널 수: {default_input['maxInputChannels']}")
            print(f"   샘플레이트: {int(default_input['defaultSampleRate'])} Hz")
            print()
        except IOError as e:
            print("❌ 기본 입력 장치를 찾을 수 없습니다!")
            print(f"   에러: {e}")
            print()
        
        # 모든 장치 목록
        device_count = audio.get_device_count()
        print(f"📋 전체 장치 수: {device_count}\n")
        
        input_devices = []
        
        for i in range(device_count):
            try:
                info = audio.get_device_info_by_index(i)
                device_type = "🎤 입력" if info['maxInputChannels'] > 0 else "🔊 출력"
                
                print(f"{device_type} 장치 [{i}]:")
                print(f"   이름: {info['name']}")
                print(f"   입력 채널: {info['maxInputChannels']}")
                print(f"   출력 채널: {info['maxOutputChannels']}")
                print(f"   샘플레이트: {int(info['defaultSampleRate'])} Hz")
                print()
                
                if info['maxInputChannels'] > 0:
                    input_devices.append(i)
                    
            except Exception as e:
                print(f"⚠️ 장치 [{i}] 정보 읽기 실패: {e}\n")
        
        audio.terminate()
        
        # 결과 요약
        print("\n" + "="*80)
        print("📊 검사 결과")
        print("="*80)
        
        if input_devices:
            print(f"✅ 사용 가능한 입력 장치 수: {len(input_devices)}")
            print(f"   장치 인덱스: {input_devices}")
            print("\n💡 해결 방법:")
            print("   1. sttStreaming.py에서 DEVICE_INDEX를 다음 중 하나로 설정:")
            for idx in input_devices:
                print(f"      DEVICE_INDEX = {idx}")
            print("\n   2. 또는 None으로 설정하여 기본 장치 사용:")
            print("      DEVICE_INDEX = None")
        else:
            print("❌ 사용 가능한 입력 장치가 없습니다!")
            print("\n💡 해결 방법:")
            print("   1. 마이크가 컴퓨터에 연결되어 있는지 확인")
            print("   2. Windows 설정 > 시스템 > 사운드에서 마이크 권한 확인")
            print("   3. 장치 관리자에서 오디오 드라이버 확인")
            print("   4. PyAudio 재설치: pip uninstall pyaudio && pip install pyaudio")
        
        print("="*80 + "\n")
        
        return len(input_devices) > 0
        
    except Exception as e:
        print(f"❌ 오디오 시스템 초기화 실패: {e}")
        print("\n💡 PyAudio가 제대로 설치되지 않았을 수 있습니다.")
        print("   해결: pip install --upgrade pyaudio")
        return False


if __name__ == "__main__":
    success = check_audio_devices()
    sys.exit(0 if success else 1)