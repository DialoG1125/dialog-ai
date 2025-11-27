"""
MySQL 데이터베이스 연결 관리
"""
import pymysql
import logging
from .config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ============================================================
# MySQL 직접 연결 (Connection Pool 없이)
# ============================================================

@contextmanager
def get_db_connection():
    """Context Manager로 자동 연결 해제"""
    connection = None
    try:
        # 매번 새로운 연결 생성
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
        logger.debug("✅ DB 연결 획득")
        yield connection
        
    except pymysql.MySQLError as e:
        logger.error(f"❌ MySQL 연결 실패: {e}")
        print(f"❌ MySQL 연결 실패: {e}")
        yield None

    except Exception as e:
        logger.error(f"❌ DB 연결 예상치 못한 오류: {e}")
        print(f"❌ DB 연결 예상치 못한 오류: {e}")
        yield None
        
    finally:
        if connection:
            connection.close()
            logger.debug("✅ DB 연결 해제")

def init_db_connection():
    """실제 DB 연결 생성 (컨텍스트 저장용)"""
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            autocommit=True
        )
        print(f"[INFO] MySQL 직접 연결 모드: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        return connection
    except Exception as e:
        print(f"[ERROR] DB 연결 실패: {e}")
        return None

def test_db_connection() -> bool:
    """데이터베이스 연결 테스트"""
    try:
        with get_db_connection() as conn:
            if not conn:
                print("[❌] 데이터베이스 연결 실패")
                return False
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            if result:
                print("[✅] 데이터베이스 연결 성공!")
                return True
            else:
                print("[❌] 데이터베이스 쿼리 실패")
                return False
                
    except Exception as e:
        print(f"[❌] 데이터베이스 연결 테스트 실패: {e}")
        return False