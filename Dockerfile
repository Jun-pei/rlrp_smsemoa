# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files to disc
# and to prevent Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies if required (e.g., for building some C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY rlrp_smsemoa/requirements.txt /app/rlrp_smsemoa/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r rlrp_smsemoa/requirements.txt

# Copy the rest of the application code
COPY rlrp_smsemoa /app/rlrp_smsemoa

# Create a default directory for results
RUN mkdir -p /app/results

# Default command: run the demo
CMD ["python", "rlrp_smsemoa/run_demo.py", "--outdir", "/app/results/demo"]