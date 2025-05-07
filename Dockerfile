FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Make sure the models directory exists with proper permissions
RUN mkdir -p models && chmod -R 755 models

# Initialize model files during build
RUN python init_models.py

# Verify model files exist
RUN ls -la models/

# Create a simple start script
RUN echo '#!/bin/bash\nuvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}' > start.sh && \
    chmod +x start.sh

# Expose the port
EXPOSE 8000

# Command to run the API
CMD ["./start.sh"]
