#!/bin/bash

# Enable debugging
set -x

# Create models directory if it doesn't exist
mkdir -p models
chmod -R 755 models

# Initialize models if needed
python init_models.py

# List model files for debugging
echo "Model files in directory:"
ls -la models/

# Get the PORT from environment or use 8000 as default
PORT=${PORT:-8000}
echo "Starting API server on port: $PORT"

# Run the API with proper logging
exec uvicorn api:app --host 0.0.0.0 --port $PORT --log-level debug
