# ============================================================
# EV Battery Degradation Engine — Dockerfile
# Multi-stage build: dependencies cached separately from code
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer cache trick)
# If requirements.txt nahi bada to yeh layer rebuild nahi hogi
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user banao (security best practice)
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Installed packages builder se copy karo
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code copy karo
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# FastAPI port
EXPOSE 8000

# Health check — Docker ko pata chale API alive hai
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Production-grade: multiple workers for concurrency
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]