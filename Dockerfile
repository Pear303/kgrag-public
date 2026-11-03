# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 环境变量
ENV NEO4J_URI=bolt://host.docker.internal:7690
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 生产建议用 gunicorn + uvicorn worker
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "serve:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--log-level", "info"]