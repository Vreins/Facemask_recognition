# Base image
FROM python:3.12-slim

# Install system libraries needed for OpenCV / PIL / YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and static files
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI app using Railway's PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
