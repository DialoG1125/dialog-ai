# ================================
# 1) Build Stage
# ================================
FROM python:3.8-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# proto → python 파일 자동 생성 (정확한 경로)
RUN python -m grpc_tools.protoc \
    --proto_path=stt/nest \
    --python_out=stt/nest \
    --grpc_python_out=stt/nest \
    stt/nest/nest.proto

# ================================
# 2) Run Stage (최종 이미지)
# ================================
FROM python:3.8-slim

WORKDIR /app

# 🎯 healthcheck 때문에 curl 추가 필수
RUN apt-get update && apt-get install -y \
    curl \
    portaudio19-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
