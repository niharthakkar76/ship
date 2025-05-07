FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy initialization script first
COPY init_models.py .

# Copy the application code
COPY . .

# Make sure the models directory exists and has proper permissions
RUN mkdir -p models && chmod -R 755 models

# Initialize model files if they don't exist
RUN python init_models.py

# Create a simple test to verify model files
RUN python -c "import os; print('Model files:', os.listdir('./models/'))"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API with environment variable for port
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
