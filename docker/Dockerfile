# Dockerfile - Save as "Dockerfile" (no extension)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=America/New_York

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    netcat-openbsd \
    iputils-ping \
    net-tools \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/reports /app/data /app/backups

# Set permissions
RUN chmod +x *.py

# Health check - ensures your system is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import yfinance as yf; yf.Ticker('AAPL').info" || exit 1

# Default command - you can override this when running
CMD ["python", "phase3_complete_integration.py", "--mode", "production", "--duration", "3600"]