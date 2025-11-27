"""
Pydantic 데이터 모델
"""
from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    session_id: Optional[str] = None
    user_job: Optional[str] = 'NONE'      # 기본값 NONE
    user_position: Optional[str] = 'NONE' # 기본값 NONE
    user_name: str                        # 필수 (로그인했으므로)

class ChatResponse(BaseModel):
    model_config = ConfigDict(exclude_none=True)  # None 필드 제외
    
    answer: str
    source: str
    history: Optional[List[dict]] = None
    session_id: Optional[str] = None