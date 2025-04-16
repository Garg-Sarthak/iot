# FROM python:3.9-slim

# WORKDIR /app

# # Copy requirements first for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code and data
# COPY app.py .
# COPY model/ ./model/
# COPY templates/ ./templates/
# COPY data/ ./data/

# # FastAPI will run on this port
# EXPOSE 8000

# # Command to run the application
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire codebase
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
