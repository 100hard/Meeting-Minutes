# Railway Optimized Dockerfile - CPU only for reliable deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install PyTorch CPU version (smaller footprint for Railway)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers and other requirements
RUN pip install --no-cache-dir transformers>=4.48.3 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

#PORT environment variable for railway
EXPOSE $PORT

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

# Run the application with Railway's PORT
CMD ["python", "app.py"]