FROM python:3.9-slim

WORKDIR /app

# System dependencies (minimal, Debian-safe)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch FIRST (important)
RUN pip install \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements (WITHOUT torch / torchvision inside it)
COPY requirements.txt .

# Install remaining Python deps
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy app code
COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
