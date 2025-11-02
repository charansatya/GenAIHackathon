# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the healthcare test generator code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_GENAI_USE_VERTEXAI=True

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
