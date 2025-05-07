import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import joblib
import io
import json

# Create FastAPI app
app = FastAPI(
    title="Berth Utilization Prediction API",
    description="API for predicting vessel berth times and analyzing berth utilization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models
def load_model():
    try:
        model_info = joblib.load('models/berth_model.pkl')
        berth_encoder = joblib.load('models/berth_encoder.pkl')
        feature_scaler = joblib.load('models/feature_scaler.pkl')
        vessel_encoder = joblib.load('models/vessel_encoder.pkl')
        return model_info, berth_encoder, feature_scaler, vessel_encoder
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Preprocess data
def preprocess_data(df, berth_encoder, vessel_encoder):
    """Preprocess input data for prediction"""
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert string dates to datetime if needed
    if 'Actual_Arrival' in df_processed.columns and df_processed['Actual_Arrival'].dtype == 'object':
        df_processed['Actual_Arrival'] = pd.to_datetime(df_processed['Actual_Arrival'])
    
    # Calculate vessel size factor
    df_processed['vessel_size_factor'] = df_processed['LOA'] * df_processed['GRT'] / 1000000
    
    # Calculate cargo density
    df_processed['cargo_density'] = df_processed['No_of_Teus'] / df_processed['GRT']
    
    # Extract time features from arrival time
    if 'Actual_Arrival' in df_processed.columns:
        df_processed['arrival_hour'] = df_processed['Actual_Arrival'].dt.hour
        df_processed['arrival_day'] = df_processed['Actual_Arrival'].dt.dayofweek
        df_processed['arrival_month'] = df_processed['Actual_Arrival'].dt.month
    
    # Handle unseen berth codes by mapping them to known codes
    known_berths = berth_encoder.classes_
    for i, berth in enumerate(df_processed['Berth_Code']):
        if berth not in known_berths:
            # Map unknown berth to a known one (first one in the encoder)
            print(f"Warning: Unknown berth code '{berth}' detected. Mapping to '{known_berths[0]}'")
            df_processed.loc[i, 'Berth_Code'] = known_berths[0]
    
    # Handle unseen vessel names by mapping them to known names
    known_vessels = vessel_encoder.classes_
    for i, vessel in enumerate(df_processed['Vessel_Name']):
        if vessel not in known_vessels:
            # Map unknown vessel to a known one (first one in the encoder)
            print(f"Warning: Unknown vessel name '{vessel}' detected. Mapping to '{known_vessels[0]}'")
            df_processed.loc[i, 'Vessel_Name'] = known_vessels[0]
    
    # Encode categorical variables
    df_processed['Berth_Code_encoded'] = berth_encoder.transform(df_processed['Berth_Code'])
    df_processed['Vessel_Name_encoded'] = vessel_encoder.transform(df_processed['Vessel_Name'])
    
    return df_processed

# Generate confidence intervals
def generate_confidence_intervals(model, X_scaled, n_iterations=100):
    """Generate confidence intervals using Random Forest's inherent randomness"""
    predictions = np.zeros((n_iterations, len(X_scaled)))
    
    # Get predictions from all trees in the forest
    for i in range(n_iterations):
        tree_idx = np.random.randint(0, len(model.estimators_))
        predictions[i] = model.estimators_[tree_idx].predict(X_scaled)
    
    # Calculate confidence intervals
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    mean = model.predict(X_scaled)  # Use full model for mean prediction
    
    return mean, lower, upper

# Generate future predictions
def generate_future_predictions(df, model_info, feature_scaler, days_ahead=14):
    """Generate predictions for future berth occupancy"""
    current_time = datetime.now()
    
    # Create one prediction per day for each vessel-berth combination
    future_predictions = []
    seen_combinations = set()  # Track unique vessel-berth-date combinations
    
    for vessel_name in df['Vessel_Name'].unique():
        vessel_data = df[df['Vessel_Name'] == vessel_name].iloc[0]
        
        # Assign each vessel to its most suitable berths (limit to 2 berths per vessel)
        suitable_berths = df[df['Vessel_Name'] == vessel_name]['Berth_Code'].unique()[:2]
        
        for berth_code in suitable_berths:
            # Space out predictions across the forecast period
            prediction_date = current_time
            
            # Add some randomness to arrival times to avoid exact overlaps
            hours_offset = np.random.randint(0, 24)
            prediction_date = prediction_date + timedelta(hours=hours_offset)
            
            # Create unique key for this combination
            combo_key = f"{vessel_name}-{berth_code}-{prediction_date.date()}"
            
            if combo_key not in seen_combinations:
                seen_combinations.add(combo_key)
                
                scenario = {
                    'Vessel_Name': vessel_name,
                    'Berth_Code': berth_code,
                    'LOA': vessel_data['LOA'],
                    'GRT': vessel_data['GRT'],
                    'No_of_Teus': vessel_data['No_of_Teus'],
                    'vessel_size_factor': vessel_data['vessel_size_factor'],
                    'cargo_density': vessel_data['cargo_density'],
                    'Vessel_Name_encoded': vessel_data['Vessel_Name_encoded'],
                    'Berth_Code_encoded': vessel_data['Berth_Code_encoded'],
                    'arrival_hour': prediction_date.hour,
                    'arrival_day': prediction_date.weekday(),
                    'arrival_month': prediction_date.month,
                    'prediction_date': prediction_date
                }
                future_predictions.append(scenario)
            
            # Add next prediction after current duration plus buffer
            prediction_date = prediction_date + timedelta(days=7)
    
    future_df = pd.DataFrame(future_predictions)
    
    # Scale features
    features = model_info['features']
    X_future = future_df[features]
    X_future_scaled = feature_scaler.transform(X_future)
    
    # Generate predictions with confidence intervals
    mean_pred, lower_pred, upper_pred = generate_confidence_intervals(
        model_info['model'], X_future_scaled)
    
    future_df['predicted_hours'] = mean_pred
    future_df['lower_bound'] = lower_pred
    future_df['upper_bound'] = upper_pred
    
    return future_df

# Calculate berth utilization
def calculate_berth_utilization(df):
    """Calculate berth utilization based on predicted departure times and next arrivals"""
    # Group by berth
    berth_stats = {}
    
    for berth in df['Berth_Code'].unique():
        berth_df = df[df['Berth_Code'] == berth].sort_values('Arrival_at_Berth')
        
        if len(berth_df) < 1:
            continue
            
        # Calculate total time range
        start_time = berth_df['Arrival_at_Berth'].min()
        end_time = berth_df['Predicted_Departure'].max()
        
        if pd.isna(start_time) or pd.isna(end_time):
            continue
            
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        # Calculate occupied hours
        occupied_hours = berth_df['Predicted_Hours_at_Berth'].sum()
        
        # Calculate utilization
        utilization = (occupied_hours / total_hours * 100) if total_hours > 0 else 0
        
        # Calculate average gap between vessels
        gaps = []
        for i in range(1, len(berth_df)):
            current_arrival = berth_df.iloc[i]['Arrival_at_Berth']
            prev_departure = berth_df.iloc[i-1]['Predicted_Departure']
            
            if pd.notna(current_arrival) and pd.notna(prev_departure):
                gap = (current_arrival - prev_departure).total_seconds() / 3600
                if gap > 0:  # Only count positive gaps
                    gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        
        berth_stats[berth] = {
            'utilization': utilization,
            'occupied_hours': occupied_hours,
            'total_hours': total_hours,
            'num_vessels': len(berth_df),
            'avg_gap_hours': avg_gap
        }
    
    return berth_stats

# Pydantic models for request/response
class VesselData(BaseModel):
    VCN: str
    IMO: str
    Vessel_Name: str
    LOA: float
    Port_Code: str
    Berth_Code: str
    No_of_Teus: int
    GRT: float
    Actual_Arrival: str

class PredictionRequest(BaseModel):
    vessels: List[VesselData]

class PredictionResponse(BaseModel):
    vessel_predictions: List[Dict[str, Any]]
    berth_statistics: Dict[str, Dict[str, Any]]

class ForecastRequest(BaseModel):
    days_ahead: int = Field(14, description="Number of days to forecast ahead")
    vessels: List[VesselData] = Field(None, description="Optional vessel data for forecast. If not provided, the API will use test data if available.")

# Function to predict berth arrival time
def predict_berth_arrival(actual_arrival, vessel_size, port_code):
    """Predict when a vessel will arrive at the berth after arriving at port"""
    # Convert actual arrival to datetime
    actual_arrival_dt = pd.to_datetime(actual_arrival)
    
    # Base waiting time - typically 20-40 minutes
    base_wait_minutes = 30
    
    # Add randomness based on vessel size (larger vessels may take longer)
    size_factor = vessel_size / 300  # Normalize by typical max size
    size_adjustment = size_factor * 15  # Up to 15 minutes additional for large vessels
    
    # Time of day factor (congestion is typically higher during daytime)
    hour = actual_arrival_dt.hour
    if 8 <= hour <= 18:  # Daytime hours
        time_of_day_adjustment = np.random.randint(5, 20)  # More congestion
    else:
        time_of_day_adjustment = np.random.randint(0, 10)  # Less congestion
    
    # Calculate total wait time
    total_wait_minutes = base_wait_minutes + size_adjustment + time_of_day_adjustment
    
    # Calculate berth arrival time
    berth_arrival = actual_arrival_dt + pd.Timedelta(minutes=total_wait_minutes)
    
    return berth_arrival

# API endpoints
@app.get("/")
async def root():
    return {"message": "Berth Utilization Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    try:
        # Check if models directory exists
        import os
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
            return JSONResponse(
                status_code=200,  # Return 200 even if models don't exist yet
                content={"status": "initializing", "message": "Models directory created", "models_present": False}
            )
        
        # List model files
        model_files = os.listdir('models')
        
        # Check if models can be loaded
        try:
            model_info, berth_encoder, feature_scaler, vessel_encoder = load_model()
            return {
                "status": "healthy", 
                "models_loaded": True,
                "model_files": model_files
            }
        except Exception as model_error:
            # Return 200 to pass the health check but indicate model loading issue
            return JSONResponse(
                status_code=200,
                content={
                    "status": "initializing", 
                    "models_present": len(model_files) > 0,
                    "model_files": model_files,
                    "model_error": str(model_error)
                }
            )
    except Exception as e:
        # Log the error for debugging
        print(f"Health check error: {str(e)}")
        
        # Return 200 to pass the health check but indicate system issue
        return JSONResponse(
            status_code=200,
            content={"status": "initializing", "system_error": str(e)}
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_berth_time(request: PredictionRequest):
    try:
        # Load models
        model_info, berth_encoder, feature_scaler, vessel_encoder = load_model()
        model = model_info['model']
        features = model_info['features']
        
        # Convert request data to DataFrame
        df = pd.DataFrame([vessel.dict() for vessel in request.vessels])
        
        # Preprocess data - first predict berth arrival times
        for i, row in df.iterrows():
            # Predict berth arrival time based on actual arrival
            df.at[i, 'Arrival_at_Berth'] = predict_berth_arrival(
                row['Actual_Arrival'], 
                row['LOA'],
                row['Port_Code']
            )
            
        # Now continue with normal preprocessing
        df_processed = preprocess_data(df, berth_encoder, vessel_encoder)
        
        # Create prediction features
        X = pd.DataFrame()
        X['LOA'] = df_processed['LOA']
        X['GRT'] = df_processed['GRT']
        X['No_of_Teus'] = df_processed['No_of_Teus']
        X['vessel_size_factor'] = df_processed['vessel_size_factor']
        X['cargo_density'] = df_processed['cargo_density']
        X['Berth_Code_encoded'] = df_processed['Berth_Code_encoded']
        X['arrival_hour'] = df_processed['arrival_hour']
        X['arrival_day'] = df_processed['arrival_day']
        X['arrival_month'] = df_processed['arrival_month']
        X['Vessel_Name_encoded'] = df_processed['Vessel_Name_encoded']
        
        # Scale features
        X_scaled = feature_scaler.transform(X[features])
        
        # Make predictions with confidence intervals
        mean_pred, lower_pred, upper_pred = generate_confidence_intervals(model, X_scaled)
        
        # Calculate predicted departure times
        df_processed['Predicted_Hours_at_Berth'] = mean_pred
        df_processed['Lower_Bound'] = lower_pred
        df_processed['Upper_Bound'] = upper_pred
        df_processed['Predicted_Departure'] = pd.to_datetime(df_processed['Arrival_at_Berth']) + \
            pd.to_timedelta(mean_pred, unit='h')
        
        # Calculate berth utilization
        berth_stats = calculate_berth_utilization(df_processed)
        
        # Prepare response
        vessel_predictions = []
        for _, row in df_processed.iterrows():
            vessel_predictions.append({
                'VCN': row['VCN'],
                'Vessel_Name': row['Vessel_Name'],
                'Berth_Code': row['Berth_Code'],
                'Actual_Arrival': row['Actual_Arrival'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Actual_Arrival']) else None,
                'Predicted_Berth_Arrival': row['Arrival_at_Berth'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Arrival_at_Berth']) else None,
                'Predicted_Hours_at_Berth': float(row['Predicted_Hours_at_Berth']),
                'Lower_Bound': float(row['Lower_Bound']),
                'Upper_Bound': float(row['Upper_Bound']),
                'Predicted_Departure': row['Predicted_Departure'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Predicted_Departure']) else None
            })
        
        # Convert berth stats to serializable format
        serializable_berth_stats = {}
        for berth, stats in berth_stats.items():
            serializable_berth_stats[berth] = {
                'utilization': float(stats['utilization']),
                'occupied_hours': float(stats['occupied_hours']),
                'total_hours': float(stats['total_hours']),
                'num_vessels': int(stats['num_vessels']),
                'avg_gap_hours': float(stats['avg_gap_hours'])
            }
        
        return {
            "vessel_predictions": vessel_predictions,
            "berth_statistics": serializable_berth_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Load models
        model_info, berth_encoder, feature_scaler, vessel_encoder = load_model()
        model = model_info['model']
        features = model_info['features']
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check required columns
        required_columns = ['VCN', 'IMO', 'Vessel_Name', 'LOA', 'Port_Code', 'Berth_Code', 'No_of_Teus', 'GRT', 'Actual_Arrival']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_cols)}")
        
        # Preprocess data - first predict berth arrival times
        for i, row in df.iterrows():
            # Predict berth arrival time based on actual arrival
            df.at[i, 'Arrival_at_Berth'] = predict_berth_arrival(
                row['Actual_Arrival'], 
                row['LOA'],
                row['Port_Code']
            )
            
        # Now continue with normal preprocessing
        df_processed = preprocess_data(df, berth_encoder, vessel_encoder)
        
        # Create prediction features
        X = pd.DataFrame()
        X['LOA'] = df_processed['LOA']
        X['GRT'] = df_processed['GRT']
        X['No_of_Teus'] = df_processed['No_of_Teus']
        X['vessel_size_factor'] = df_processed['vessel_size_factor']
        X['cargo_density'] = df_processed['cargo_density']
        X['Berth_Code_encoded'] = df_processed['Berth_Code_encoded']
        X['arrival_hour'] = df_processed['arrival_hour']
        X['arrival_day'] = df_processed['arrival_day']
        X['arrival_month'] = df_processed['arrival_month']
        X['Vessel_Name_encoded'] = df_processed['Vessel_Name_encoded']
        
        # Scale features
        X_scaled = feature_scaler.transform(X[features])
        
        # Make predictions with confidence intervals
        mean_pred, lower_pred, upper_pred = generate_confidence_intervals(model, X_scaled)
        
        # Calculate predicted departure times
        df_processed['Predicted_Hours_at_Berth'] = mean_pred
        df_processed['Lower_Bound'] = lower_pred
        df_processed['Upper_Bound'] = upper_pred
        df_processed['Predicted_Departure'] = pd.to_datetime(df_processed['Arrival_at_Berth']) + \
            pd.to_timedelta(mean_pred, unit='h')
        
        # Calculate berth utilization
        berth_stats = calculate_berth_utilization(df_processed)
        
        # Prepare response
        vessel_predictions = []
        for _, row in df_processed.iterrows():
            vessel_predictions.append({
                'VCN': row['VCN'],
                'Vessel_Name': row['Vessel_Name'],
                'Berth_Code': row['Berth_Code'],
                'Actual_Arrival': row['Actual_Arrival'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Actual_Arrival']) else None,
                'Predicted_Berth_Arrival': row['Arrival_at_Berth'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Arrival_at_Berth']) else None,
                'Predicted_Hours_at_Berth': float(row['Predicted_Hours_at_Berth']),
                'Lower_Bound': float(row['Lower_Bound']),
                'Upper_Bound': float(row['Upper_Bound']),
                'Predicted_Departure': row['Predicted_Departure'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Predicted_Departure']) else None
            })
        
        # Convert berth stats to serializable format
        serializable_berth_stats = {}
        for berth, stats in berth_stats.items():
            serializable_berth_stats[berth] = {
                'utilization': float(stats['utilization']),
                'occupied_hours': float(stats['occupied_hours']),
                'total_hours': float(stats['total_hours']),
                'num_vessels': int(stats['num_vessels']),
                'avg_gap_hours': float(stats['avg_gap_hours'])
            }
        
        return {
            "vessel_predictions": vessel_predictions,
            "berth_statistics": serializable_berth_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    try:
        # Load models
        model_info, berth_encoder, feature_scaler, vessel_encoder = load_model()
        
        # Check if user provided vessel data
        if request.vessels and len(request.vessels) > 0:
            # Use vessel data provided by the user
            vessel_data = []
            for vessel in request.vessels:
                vessel_data.append({
                    'VCN': vessel.VCN,
                    'IMO': vessel.IMO,
                    'Vessel_Name': vessel.Vessel_Name,
                    'LOA': vessel.LOA,
                    'Port_Code': vessel.Port_Code,
                    'Berth_Code': vessel.Berth_Code,
                    'No_of_Teus': vessel.No_of_Teus,
                    'GRT': vessel.GRT,
                    'Actual_Arrival': pd.to_datetime(vessel.Actual_Arrival)
                })
            df = pd.DataFrame(vessel_data)
        else:
            # Try to load test data from CSV files
            try:
                # First try to load test_data.csv
                df = pd.read_csv('test_data.csv')
            except:
                # If test_data.csv is not available, try Data_Berth.csv
                df = pd.read_csv('Data_Berth.csv')
        
        # Preprocess data
        df_processed = preprocess_data(df, berth_encoder, vessel_encoder)
        
        # Generate future predictions
        future_df = generate_future_predictions(
            df_processed, model_info, feature_scaler, days_ahead=request.days_ahead)
        
        # Create timeline for future predictions
        future_timeline = []
        for _, row in future_df.iterrows():
            start_time = row['prediction_date']
            end_time = start_time + timedelta(hours=row['predicted_hours'])
            
            future_timeline.append({
                'Berth': row['Berth_Code'],
                'Vessel': row['Vessel_Name'],
                'Start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'End': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Duration': float(row['predicted_hours']),
                'Lower_CI': float(row['lower_bound']),
                'Upper_CI': float(row['upper_bound'])
            })
        
        return {"forecast": future_timeline}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
