# Berth Utilization Prediction API

This API service provides endpoints for predicting vessel berth arrival times, berth durations, and analyzing berth utilization patterns.

## Model Performance

The underlying Random Forest model shows excellent performance metrics:

- **Training Mean Absolute Error (MAE)**: 0.12 hours (approximately 7 minutes)
- **Test Mean Absolute Error (MAE)**: 0.14 hours (approximately 8-9 minutes)
- **Training R²**: 0.996
- **Test R²**: 0.995

These metrics indicate that the model is highly accurate, with predictions typically within about 8-9 minutes of the actual values. The R² value of 0.995 on test data suggests that the model explains 99.5% of the variance in berth time.

### Prediction Patterns

#### Berth Arrival Time Predictions
- Most vessels have a predicted wait time of approximately 50-60 minutes from port arrival to berth arrival
- Wait times vary slightly based on vessel size (larger vessels have slightly longer wait times)
- Time of day affects the predictions (vessels arriving during busier hours have longer predicted wait times)

#### Berth Duration Predictions
- Vessel-specific durations typically range from 14.6 to 17.1 hours
- Confidence intervals are tight (±0.3-0.4 hours) for forecasts
- Larger vessels (based on LOA and GRT) typically require more time at berth

#### Vessel-Specific Patterns
- Mediterranean Queen has the longest stays (~17.1h)
- Nordic Explorer has the shortest stays (~14.6h)
- Other vessels fall within this range based on their characteristics

## Setup and Installation

1. Ensure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Run the API service:

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Available Endpoints

### 1. Health Check

```
GET /health
```

Checks if the API is running and models are loaded correctly.

### 2. Predict Berth Time

```
POST /predict
```

Predicts berth arrival time, duration, and departure time for a list of vessels.

**Request Body:**
```json
{
  "vessels": [
    {
      "VCN": "string",
      "IMO": "string",
      "Vessel_Name": "string",
      "LOA": 0,
      "Port_Code": "string",
      "Berth_Code": "string",
      "No_of_Teus": 0,
      "GRT": 0,
      "Actual_Arrival": "2023-01-01T00:00:00"
    }
  ]
}
```

**Response:**
```json
{
  "vessel_predictions": [
    {
      "VCN": "string",
      "Vessel_Name": "string",
      "Berth_Code": "string",
      "Actual_Arrival": "2023-01-01 00:00:00",
      "Predicted_Berth_Arrival": "2023-01-01 00:00:00",
      "Predicted_Hours_at_Berth": 0,
      "Lower_Bound": 0,
      "Upper_Bound": 0,
      "Predicted_Departure": "2023-01-01 00:00:00"
    }
  ],
  "berth_statistics": {
    "berth_code": {
      "utilization": 0,
      "occupied_hours": 0,
      "total_hours": 0,
      "num_vessels": 0,
      "avg_gap_hours": 0
    }
  }
}
```

### 3. Upload CSV

```
POST /upload-csv
```

Upload a CSV file with vessel data for batch prediction.

**Request:**
- Form data with a CSV file

**Response:**
Same format as the `/predict` endpoint.

### 4. Forecast

```
POST /forecast
```

Generate future berth occupancy forecasts.

**Request Body:**
```json
{
  "days_ahead": 14
}
```

**Response:**
```json
{
  "forecast": [
    {
      "Berth": "string",
      "Vessel": "string",
      "Start": "2023-01-01 00:00:00",
      "End": "2023-01-01 00:00:00",
      "Duration": 0,
      "Lower_CI": 0,
      "Upper_CI": 0
    }
  ]
}
```

## Example Usage

### Using cURL

1. Health check:
```bash
curl -X GET http://localhost:8000/health
```

2. Predict berth time:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vessels": [
      {
        "VCN": "VCN123",
        "IMO": "IMO456",
        "Vessel_Name": "Mediterranean Queen",
        "LOA": 294.1,
        "Port_Code": "INBOM",
        "Berth_Code": "BT1",
        "No_of_Teus": 8500,
        "GRT": 94000,
        "Actual_Arrival": "2023-01-01T08:00:00",
        "Arrival_at_Berth": "2023-01-01T10:00:00"
      }
    ]
  }'
```

3. Upload CSV:
```bash
curl -X POST http://localhost:8000/upload-csv \
  -F "file=@vessel_data.csv"
```

4. Generate forecast:
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "days_ahead": 14
  }'
```

### Using Python Requests

```python
import requests
import json

# Base URL
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Predict berth time
vessel_data = {
    "vessels": [
        {
            "VCN": "VCN123",
            "IMO": "IMO456",
            "Vessel_Name": "Mediterranean Queen",
            "LOA": 294.1,
            "Port_Code": "INBOM",
            "Berth_Code": "BT1",
            "No_of_Teus": 8500,
            "GRT": 94000,
            "Actual_Arrival": "2023-01-01 00:00:00"
        }
    ]
}

response = requests.post(
    f"{base_url}/predict",
    json=vessel_data
)
print(json.dumps(response.json(), indent=2))

# Upload CSV
with open('vessel_data.csv', 'rb') as f:
    response = requests.post(
        f"{base_url}/upload-csv",
        files={"file": f}
    )
print(json.dumps(response.json(), indent=2))

# Generate forecast
response = requests.post(
    f"{base_url}/forecast",
    json={"days_ahead": 14}
)
print(json.dumps(response.json(), indent=2))
```
