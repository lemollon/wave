# Force Python 3.11
FROM python:3.11-slim

# Avoid pyc files & enable clean logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install basic system packages needed by some Python libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (Docker caching optimization)
COPY requirements.txt /app/
COPY requirements-pro.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
 && pip install --only-binary=:all: -r requirements.txt \
 && pip install --only-binary=:all: -r requirements-pro.txt

# Copy the rest of the app code
COPY . /app

# Streamlit environment vars
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Render will inject $PORT dynamically, fallback to 8080
ENV PORT=8080
EXPOSE 8080

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
