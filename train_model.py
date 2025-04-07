import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

def evaluate_predictions(y_true, y_pred, title="Model Performance"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{title}:")
    print(f"Mean Absolute Error: {mae:.2f} hours")
    print(f"Root Mean Squared Error: {rmse:.2f} hours")
    print(f"R² Score: {r2:.3f}")
    
    # Print detailed prediction comparison
    comparison = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Difference': np.abs(y_true - y_pred)
    })
    print("\nPrediction Details (showing first 10 samples):")
    print(comparison.head(10))
    
    return mae, rmse, r2

def preprocess_data(df):
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Convert timestamp columns to datetime with microseconds
    timestamp_cols = [
        'Actual_Arrival', 
        'Arrival_at_Berth', 
        'Ops_Start_from', 
        'Ops_Completed_On', 
        'DeParture_from_Berth', 
        'Dearture_from_Port',
        'Month'
    ]
    
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed')
    
    # Calculate target: total time from arrival to departure (in hours)
    df['total_time_at_berth'] = (df['DeParture_from_Berth'] - df['Arrival_at_Berth']).dt.total_seconds() / 3600
    
    # Create features based on initial vessel data only
    df['vessel_size_factor'] = df['LOA'] * df['GRT'] / 1000000  # Combined size factor
    df['cargo_density'] = df['No_of_Teus'] / df['GRT']  # TEUs per vessel weight
    
    # Extract time components that would be known at arrival
    df['arrival_hour'] = df['Actual_Arrival'].dt.hour
    df['arrival_day'] = df['Actual_Arrival'].dt.dayofweek
    df['arrival_month'] = df['Actual_Arrival'].dt.month
    
    # Label encode categorical features
    le_berth = LabelEncoder()
    le_vessel = LabelEncoder()
    
    df['Berth_Code_encoded'] = le_berth.fit_transform(df['Berth_Code'])
    df['Vessel_Name_encoded'] = le_vessel.fit_transform(df['Vessel_Name'])
    
    # Save the encoders
    joblib.dump(le_berth, 'models/berth_encoder.pkl')
    joblib.dump(le_vessel, 'models/vessel_encoder.pkl')
    
    return df

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv('Data_Berth.csv')
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Select features that would be available at arrival time
    features = [
        # Vessel characteristics
        'LOA',                    # Length of vessel
        'GRT',                    # Vessel weight
        'No_of_Teus',            # Cargo volume
        'vessel_size_factor',     # Combined size impact
        'cargo_density',          # Cargo efficiency
        'Vessel_Name_encoded',    # Vessel history
        'Berth_Code_encoded',     # Berth assignment
        
        # Arrival time features
        'arrival_hour',           # Hour of arrival
        'arrival_day',           # Day of week
        'arrival_month',         # Month
    ]
    
    X = df_processed[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Save the scaler
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    # Target: total time vessel will spend at berth
    y = df_processed['total_time_at_berth']
    
    print("\nTarget Variable Statistics:")
    print(f"Mean time at berth: {y.mean():.2f} hours")
    print(f"Std time at berth: {y.std():.2f} hours")
    print(f"Min time at berth: {y.min():.2f} hours")
    print(f"Max time at berth: {y.max():.2f} hours")
    
    print("\nPerforming cross-validation...")
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_model, X_scaled, y, cv=cv, scoring='r2')
    print("\nCross-validation R² scores:", cv_scores)
    print(f"Mean CV R² score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("\nTraining final model...")
    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    final_model.fit(X_train, y_train)
    
    # Evaluate on training data
    y_train_pred = final_model.predict(X_train)
    train_mae, train_rmse, train_r2 = evaluate_predictions(y_train, y_train_pred, "Training Set Performance")
    
    # Evaluate on test data
    y_test_pred = final_model.predict(X_test)
    test_mae, test_rmse, test_r2 = evaluate_predictions(y_test, y_test_pred, "Test Set Performance")
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Check for potential overfitting
    print("\nOverfitting Analysis:")
    print(f"Training R² - Test R² difference: {train_r2 - test_r2:.3f}")
    if train_r2 - test_r2 > 0.1:
        print("Warning: Model might be overfitting (difference > 0.1)")
    
    # Save model and metadata
    model_info = {
        'model': final_model,
        'features': features,
        'metrics': {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_scores': {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
        },
        'feature_importance': feature_importance.to_dict()
    }
    
    # Save the model info
    print("\nSaving model...")
    joblib.dump(model_info, 'models/berth_model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
