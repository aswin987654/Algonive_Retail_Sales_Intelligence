FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (important)
RUN pip install --upgrade pip

COPY requirements.txt .

# Increase timeout for slower networks
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY data/ data/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]