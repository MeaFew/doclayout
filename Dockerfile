FROM python:3.11-slim

LABEL org.opencontainers.image.title="doclayout"
LABEL org.opencontainers.image.description="Document layout analysis + table recognition with PP-StructureV3"

# build-essential: compile fallback for pip deps without wheels.
# libgomp1: OpenMP runtime required by paddlepaddle.
# libgl1 + libglib2.0-0: required by OpenCV (pulled in via paddlex).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching), then the package
# itself — src-layout means `doclayout` is NOT importable without this.
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY . .

# Run as an unprivileged user; streamlit needs a writable HOME.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
