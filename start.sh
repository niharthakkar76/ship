#!/bin/bash

# For Railway, we know we need to use port 8000
echo "Starting API server on port: 8000"

# Run the API on port 8000 for Railway
exec uvicorn api:app --host 0.0.0.0 --port 8000
