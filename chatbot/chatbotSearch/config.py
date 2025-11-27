"""
환경변수 설정
"""
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# ============================================================
# 환경 변수
# ============================================================

# Lambda Function URL
LAMBDA_FUNCTION_URL = os.getenv('LAMBDA_FUNCTION_URL')

# HyperCLOVA X
CLOVA_STUDIO_URL = os.getenv('CLOVA_STUDIO_URL')
CLOVA_API_KEY = os.getenv('CLOVA_API_KEY')

# Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# MySQL (로컬)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'dialog')

# ============================================================
# Phase 2-A: Template 페르소나 설정
# ============================================================

# Template 페르소나 활성화 (HyperCLOVA X 없이도 작동)
ENABLE_PERSONA = os.getenv('ENABLE_PERSONA', 'true').lower() == 'true'

print(f"[CONFIG] Phase 2-A (Template 페르소나): {'✅ 활성화' if ENABLE_PERSONA else '❌ 비활성화'}")

# ============================================================
# MySQL Connector Config
# ============================================================

DB_CONFIG = {
    'host': DB_HOST,
    'port': DB_PORT,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'charset': 'utf8mb4'
}