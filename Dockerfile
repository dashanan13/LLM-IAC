FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libopencv-dev \
    python3-opencv \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories if they don't exist and set permissions
RUN mkdir -p app/static/uploads app/static/processed \
    && chmod 777 app/static/uploads \
    && chmod 777 app/static/processed

EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_ENV=development
ENV FLASK_APP=run.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]