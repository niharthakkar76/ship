# Berth Utilization Prediction System

This system predicts vessel berth arrival times, berth durations, and calculates berth utilization based on historical vessel data. It includes both a Streamlit web application and a FastAPI service for programmatic access.

## Features
- Upload CSV files with vessel data
- Predict berth arrival times, durations, and departure times using machine learning
- Calculate berth utilization and statistics
- Visualize berth occupancy with interactive timelines
- Generate future berth occupancy forecasts (7-14 days ahead)
- Access predictions via web interface or API endpoints

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Alternatively, run the FastAPI service:
```bash
python run_api.py
```
The API will be available at `http://localhost:8000`. See `API_README.md` for detailed API documentation.

## Input Data Format

### For Training
The CSV file should contain the following columns:
- VCN (Vessel Call Number)
- IMO (International Maritime Organization number)
- Vessel_Name
- LOA (Length Overall)
- Port_Code
- Berth_Code
- No_of_Teus
- GRT (Gross Registered Tonnage)
- Actual_Arrival
- Arrival_at_Berth
- Ops_Start_from
- Ops_Completed_On
- DeParture_from_Berth
- Dearture_from_Port

### For Prediction (API)
The minimum required columns are:
- VCN
- IMO
- Vessel_Name
- LOA
- Port_Code
- Berth_Code
- No_of_Teus
- GRT
- Actual_Arrival

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
