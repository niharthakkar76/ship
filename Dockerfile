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

# Initialize model files
RUN python init_models.py

# Verify model files exist
RUN python -c "import os; print('Model files:', os.listdir('./models/'))"

# Make sure start script is executable
RUN chmod +x /app/start.sh

# Expose the port the app runs on
EXPOSE 8000

# Use the start script to handle environment variables properly
CMD ["/app/start.sh"]
