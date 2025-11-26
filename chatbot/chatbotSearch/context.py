"""
Redis 컨텍스트 관리
"""
import redis
import json
import logging
from .config import REDIS_HOST, REDIS_PORT
import hashlib
import time

logger = logging.getLogger(__name__)

# ============================================================
# Redis 클라이언트 초기화
# ============================================================

redis_client = None

def init_redis_client():
    """Redis 클라이언트 초기화"""
    global redis_client
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        redis_client.ping()
        print(f"[DEBUG] Redis 연결: {REDIS_HOST}:{REDIS_PORT}")
        logger.info(f"Redis 연결 성공")
        return redis_client
    except Exception as e:
        logger.error(f"[ERROR] Redis 연결 실패: {e}")
        redis_client = None
        return None

def get_redis_client():
    """Redis 클라이언트 반환"""
    global redis_client
    
    if redis_client is None:
        return init_redis_client()
    
    try:
        redis_client.ping()
        return redis_client
    except redis.ConnectionError as e:  # 더 구체적인 예외
        logger.warning(f"Redis 연결 끊김, 재연결 시도: {e}")
        return init_redis_client()
    except Exception as e:
        logger.error(f"Redis 예상치 못한 오류: {e}")
        return init_redis_client()

# ============================================================
# 컨텍스트 관리 함수
# ============================================================

def get_context(session_id: str) -> dict:
    """Redis에서 컨텍스트 가져오기"""
    client = get_redis_client()
    if not client:
        return {}
    
    try:
        context_json = client.get(f"context:{session_id}")
        if context_json:
            return json.loads(context_json)
        return {}
    except Exception as e:
        logger.error(f"컨텍스트 조회 실패: {e}")
        return {}

def save_context(session_id: str, context: dict, ttl: int = 600):
    """Redis에 컨텍스트 저장 (기본 TTL: 10분)"""
    client = get_redis_client()
    if not client:
        return False
    
    try:
        # datetime과 bytes를 JSON 직렬화 가능하게 변환
        def convert_to_json_serializable(obj):
            """JSON 직렬화 가능한 형태로 변환"""
            from datetime import datetime
            
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, bytes):
                return obj.decode('utf-8', errors='ignore')
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        # 컨텍스트 전체를 변환
        serializable_context = convert_to_json_serializable(context)
        
        context_json = json.dumps(serializable_context, ensure_ascii=False)
        client.setex(f"context:{session_id}", ttl, context_json)
        logger.info(f"컨텍스트 저장 성공: {session_id}")
        return True
    except Exception as e:
        logger.error(f"컨텍스트 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def delete_context(session_id: str):
    """Redis에서 컨텍스트 삭제"""
    client = get_redis_client()
    if not client:
        return False
    
    try:
        client.delete(f"context:{session_id}")
        logger.info(f"컨텍스트 삭제 성공: {session_id}")
        return True
    except Exception as e:
        logger.error(f"컨텍스트 삭제 실패: {e}")
        return False
    
def generate_session_id(user_id: str = "default") -> str:
    """세션 ID 생성"""
    timestamp = str(time.time())
    return hashlib.md5(f"{user_id}_{timestamp}".encode()).hexdigest()[:16]