# Berth Utilization Prediction Model Documentation

## Table of Contents
1. [Model Overview](#model-overview)
2. [Feature Engineering](#feature-engineering)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Model Performance](#model-performance)
6. [Prediction Pipeline](#prediction-pipeline)

## Model Overview

The Berth Utilization Prediction model is a Random Forest Regressor designed to predict vessel berth times and analyze berth utilization patterns. The model achieves high accuracy with confidence intervals of ±0.3-0.4 hours.

### Key Capabilities:
- Vessel-specific berth time predictions
- Future berth occupancy forecasting (7-14 days)
- Berth utilization analysis
- Confidence interval generation

## Feature Engineering

### Input Features

1. **Vessel Physical Characteristics**
   - `LOA` (Length Overall)
   - `GRT` (Gross Registered Tonnage)
   - `No_of_Teus` (Container Capacity)

2. **Derived Features**
   ```python
   vessel_size_factor = LOA * GRT / 1_000_000  # Combined size impact
   cargo_density = No_of_Teus / GRT            # Cargo efficiency
   ```

3. **Temporal Features**
   - `arrival_hour` (0-23)
   - `arrival_day` (0-6)
   - `arrival_month` (1-12)

4. **Categorical Features**
   - `Vessel_Name` (Label Encoded)
   - `Berth_Code` (Label Encoded)

### Feature Importance
```
Feature               Importance
-----------------------------
vessel_size_factor    0.285
cargo_density         0.198
LOA                   0.156
GRT                   0.142
Vessel_Name_encoded   0.089
Berth_Code_encoded    0.075
arrival_hour          0.025
arrival_month         0.018
arrival_day           0.012
```

## Model Architecture

### Random Forest Configuration
```python
RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum tree depth
    min_samples_split=5, # Minimum samples for split
    min_samples_leaf=2,  # Minimum samples per leaf
    random_state=42      # Seed for reproducibility
)
```

### Data Preprocessing Pipeline
1. **Feature Scaling**
   - StandardScaler for numerical features
   - Mean = 0, Standard Deviation = 1

2. **Categorical Encoding**
   - LabelEncoder for vessel names and berth codes
   - Persistent encoders saved for prediction

## Training Process

### Data Split
- Training Set: 80%
- Test Set: 20%
- 5-fold Cross-validation

### Training Steps
1. Load and preprocess data
2. Engineer features
3. Scale features
4. Train model with cross-validation
5. Evaluate performance
6. Generate feature importance
7. Save model artifacts

## Model Performance

### Accuracy Metrics
- Mean Absolute Error: ±0.35 hours
- Root Mean Squared Error: 0.42 hours
- R² Score: 0.89

### Vessel-Specific Performance
- Mediterranean Queen: ~17.1h (±0.4h)
- Nordic Explorer: ~14.6h (±0.3h)

### Cross-Validation Results
- Mean R² Score: 0.88
- Standard Deviation: 0.02

## Prediction Pipeline

### Input Processing
```python
def preprocess_data(df):
    # Convert timestamps
    df['total_time_at_berth'] = (
        df['DeParture_from_Berth'] - df['Arrival_at_Berth']
    ).dt.total_seconds() / 3600
    
    # Engineer features
    df['vessel_size_factor'] = df['LOA'] * df['GRT'] / 1000000
    df['cargo_density'] = df['No_of_Teus'] / df['GRT']
    
    # Extract temporal features
    df['arrival_hour'] = df['Actual_Arrival'].dt.hour
    df['arrival_day'] = df['Actual_Arrival'].dt.dayofweek
    df['arrival_month'] = df['Actual_Arrival'].dt.month
```

### Confidence Interval Generation
```python
def generate_confidence_intervals(model, X_scaled, n_iterations=100):
    predictions = np.zeros((n_iterations, len(X_scaled)))
    
    # Sample from trees
    for i in range(n_iterations):
        tree_idx = np.random.randint(0, len(model.estimators_))
        predictions[i] = model.estimators_[tree_idx].predict(X_scaled)
    
    # Calculate intervals
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    mean = model.predict(X_scaled)
    
    return mean, lower, upper
```

### Future Predictions
The model can generate predictions up to 14 days ahead by:
1. Creating vessel-berth combinations
2. Generating temporal features for future dates
3. Applying the prediction pipeline
4. Calculating confidence intervals
5. Analyzing berth utilization

## Model Artifacts
- `berth_model.pkl`: Trained Random Forest model
- `feature_scaler.pkl`: StandardScaler for features
- `berth_encoder.pkl`: LabelEncoder for berth codes
- `vessel_encoder.pkl`: LabelEncoder for vessel names
