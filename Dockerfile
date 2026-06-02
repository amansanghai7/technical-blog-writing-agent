FROM python:3.12-slim

WORKDIR /app

# curl is used by the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying app code so Docker can cache this layer.
# Rebuilds only re-run pip when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (everything not excluded by .dockerignore)
COPY . .

EXPOSE 8501

# Streamlit exposes a health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "bwa_frontend.py"]
