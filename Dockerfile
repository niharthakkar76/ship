FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy initialization script and API first
COPY init_models.py api.py ./

# Copy start script
COPY start.sh ./
RUN chmod +x /app/start.sh

# Copy remaining files
COPY . .

# Make sure the models directory exists with proper permissions
RUN mkdir -p models && chmod -R 755 models

# Initialize model files during build
RUN python init_models.py

# Verify model files exist
RUN ls -la models/

# Expose the port the app runs on
EXPOSE 8000

# Use the start script to handle environment variables properly
CMD ["/bin/bash", "/app/start.sh"]
