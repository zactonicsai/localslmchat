FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY backend/ /app/
EXPOSE 6976
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6976", "--log-level", "info"]