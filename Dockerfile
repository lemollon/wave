# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (safe to keep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (core + pro)
COPY requirements.txt requirements-pro.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-pro.txt

# Copy code
COPY . .

# Expose and run Streamlit on Render
ENV PORT=10000
EXPOSE 10000
CMD ["streamlit", "run", "app.py", "--server.port", "10000", "--server.address", "0.0.0.0"]
