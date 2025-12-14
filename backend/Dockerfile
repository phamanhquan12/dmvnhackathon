FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing and ffmpeg for video
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements-A.txt requirements-A.txt
COPY requirements-docker.txt requirements-docker.txt

# Install PyTorch CPU version first (CUDA versions need special index)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (without torch/torchvision)
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install additional packages for Streamlit and Google GenAI
RUN pip install --no-cache-dir streamlit google-generativeai pydantic-settings python-dotenv pypdf

# Copy application code
COPY . .

# Create directories for uploads and temp files
RUN mkdir -p /app/uploads /app/temp

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
