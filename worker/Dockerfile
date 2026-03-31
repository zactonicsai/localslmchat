FROM python:3.12-slim

# System deps (curl for health checks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for production security
RUN useradd --create-home appuser

WORKDIR /app

# Install Python deps (layer cached separately from code)
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared models and worker code
COPY shared/ /app/shared/
COPY worker/ /app/

# Run as non-root
USER appuser

# Health check: verify the process is alive
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pgrep -f "run_worker.py" || exit 1

CMD ["python", "run_worker.py"]
