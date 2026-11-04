# ==========================================
# Stage 1: Build React TypeScript Frontend
# ==========================================
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Production Python Backend Runtime
# ==========================================
FROM python:3.12-slim AS runner

# System dependencies for curl and lightgbm OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python CPU dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir greenlet

# Copy preprocessed datasets and trained model artifacts
COPY data/processed /app/data/processed
COPY artifacts /app/artifacts

# Copy source code and pipelines
COPY src /app/src
COPY pipelines /app/pipelines

# Copy static frontend build from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Ensure persistent data directory exists
RUN mkdir -p /app/data_persist

# Set production environment variables
ENV ENVIRONMENT=production \
    FORCE_CPU=true \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn production server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
