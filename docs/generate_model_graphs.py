import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load and preprocess data
df = pd.read_csv('Data_Berth.csv')

# Process timestamps
timestamp_cols = [
    'Actual_Arrival', 'Arrival_at_Berth', 'Ops_Start_from',
    'Ops_Completed_On', 'DeParture_from_Berth', 'Dearture_from_Port', 'Month'
]
for col in timestamp_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='mixed')

# Calculate target
df['total_time_at_berth'] = (df['DeParture_from_Berth'] - df['Arrival_at_Berth']).dt.total_seconds() / 3600

# Feature engineering
df['vessel_size_factor'] = df['LOA'] * df['GRT'] / 1000000
df['cargo_density'] = df['No_of_Teus'] / df['GRT']
df['arrival_hour'] = df['Actual_Arrival'].dt.hour
df['arrival_day'] = df['Actual_Arrival'].dt.dayofweek
df['arrival_month'] = df['Actual_Arrival'].dt.month

# Encode categorical variables
le_vessel = LabelEncoder()
le_berth = LabelEncoder()
df['Vessel_Name_encoded'] = le_vessel.fit_transform(df['Vessel_Name'])
df['Berth_Code_encoded'] = le_berth.fit_transform(df['Berth_Code'])

# Load model and encoders
model_info = joblib.load('models/berth_model.pkl')
feature_importance = pd.DataFrame(model_info['feature_importance'])

# 1. Feature Importance Plot
fig1 = px.bar(
    feature_importance.sort_values('importance', ascending=True),
    x='importance',
    y='feature',
    orientation='h',
    title='Feature Importance in Berth Time Prediction'
)
fig1.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, color='black'),
    xaxis_title='Importance Score',
    yaxis_title='Feature'
)
fig1.write_image("feature_importance.png")

# 2. Prediction vs Actual Plot
# Get features for prediction
X = df[model_info['features']]
X_scaled = StandardScaler().fit_transform(X)
predictions = model_info['model'].predict(X_scaled)
fig2 = px.scatter(
    x=df['total_time_at_berth'],
    y=predictions,
    title='Predicted vs Actual Berth Times'
)
fig2.add_trace(
    go.Scatter(
        x=[df['total_time_at_berth'].min(), df['total_time_at_berth'].max()],
        y=[df['total_time_at_berth'].min(), df['total_time_at_berth'].max()],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Perfect Prediction'
    )
)
fig2.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, color='black'),
    xaxis_title='Actual Berth Time (hours)',
    yaxis_title='Predicted Berth Time (hours)'
)
fig2.write_image("prediction_accuracy.png")

# 3. Cross-validation Performance
cv_scores = model_info['metrics']['cv_scores']['scores']
fig3 = go.Figure()
fig3.add_trace(go.Box(
    y=cv_scores,
    name='Cross-validation R² Scores',
    boxpoints='all',
    jitter=0.3,
    pointpos=-1.8
))
fig3.update_layout(
    title='Model Cross-validation Performance',
    yaxis_title='R² Score',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, color='black'),
    showlegend=False
)
fig3.write_image("cross_validation.png")

# 4. Vessel Size vs Berth Time
fig4 = px.scatter(
    df,
    x='vessel_size_factor',
    y='total_time_at_berth',
    color='Berth_Code',
    title='Vessel Size Factor vs Berth Time'
)
fig4.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, color='black'),
    xaxis_title='Vessel Size Factor',
    yaxis_title='Berth Time (hours)'
)
fig4.write_image("size_vs_time.png")

# 5. Temporal Patterns
hourly_avg = df.groupby('arrival_hour')['total_time_at_berth'].mean().reset_index()
fig5 = px.line(
    hourly_avg,
    x='arrival_hour',
    y='total_time_at_berth',
    title='Average Berth Time by Hour of Day'
)
fig5.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, color='black'),
    xaxis_title='Hour of Day',
    yaxis_title='Average Berth Time (hours)'
)
fig5.write_image("temporal_patterns.png")

print("Generated all model visualization graphs in the current directory.")
