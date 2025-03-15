FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    git \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Add runpod serverless dependencies
RUN pip3 install --no-cache-dir runpod

# Copy project code
COPY . .

# Create directories for temporary files and voices
RUN mkdir -p /app/temp /app/voices

# Set the entrypoint
ENTRYPOINT ["python3", "-u", "runpod_handler.py"]
