import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from datetime import datetime, timedelta
import joblib

# Configure page settings and theme
st.set_page_config(
    page_title="Berth Utilization Predictor",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS for white theme
st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #333333;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
        }
        .stTextInput>div>div>input {
            color: #333333;
        }
        h1, h2, h3 {
            color: #1E1E1E;
        }
        .stPlotlyChart {
            background-color: #FFFFFF;
        }
        .css-1d391kg {
            background-color: #FFFFFF;
        }
        .st-emotion-cache-1wivap2 {
            background-color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

# Header with logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/fluency/96/cargo-ship.png", width=80)
with col2:
    st.title("Berth Utilization Predictor")

@st.cache_resource
def load_model():
    try:
        model_info = joblib.load('models/berth_model.pkl')
        berth_encoder = joblib.load('models/berth_encoder.pkl')
        feature_scaler = joblib.load('models/feature_scaler.pkl')
        return model_info, berth_encoder, feature_scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_data(df, berth_encoder):
    """Preprocess input data for prediction"""
    # Convert timestamp columns to datetime
    timestamp_cols = ['Actual_Arrival', 'Arrival_at_Berth']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
    
    # Create vessel features
    df['vessel_size_factor'] = df['LOA'] * df['GRT'] / 1000000
    df['cargo_density'] = df['No_of_Teus'] / df['GRT']
    
    # Extract time components
    df['arrival_hour'] = df['Actual_Arrival'].dt.hour
    df['arrival_day'] = df['Actual_Arrival'].dt.dayofweek
    df['arrival_month'] = df['Actual_Arrival'].dt.month
    
    # Encode berth
    df['Berth_Code_encoded'] = berth_encoder.transform(df['Berth_Code'])
    
    return df

def calculate_berth_utilization(df):
    """Calculate berth utilization based on predicted departure times and next arrivals"""
    if len(df) == 0:
        return {}
        
    df = df.sort_values('Arrival_at_Berth')
    berth_stats = {}
    
    for berth in df['Berth_Code'].unique():
        berth_df = df[df['Berth_Code'] == berth].copy()
        if len(berth_df) < 2:
            # For single vessel, calculate utilization for 24-hour period
            duration = berth_df['Predicted_Hours_at_Berth'].iloc[0]
            berth_stats[berth] = {
                'utilization': (duration / 24.0) * 100,
                'total_gap_hours': 24.0 - duration,
                'avg_gap_hours': 24.0 - duration,
                'num_vessels': 1
            }
            continue
            
        # Calculate gaps between vessels
        berth_df['next_arrival'] = berth_df['Arrival_at_Berth'].shift(-1)
        berth_df['gap_hours'] = (berth_df['next_arrival'] - berth_df['Predicted_Departure']).dt.total_seconds() / 3600
        
        # Only consider positive gaps (negative means overlap)
        valid_gaps = berth_df['gap_hours'][berth_df['gap_hours'] > 0]
        
        total_period = (berth_df['Predicted_Departure'].max() - berth_df['Arrival_at_Berth'].min()).total_seconds() / 3600
        if total_period > 0:
            total_gap_hours = valid_gaps.sum() if len(valid_gaps) > 0 else 0
            utilization = ((total_period - total_gap_hours) / total_period) * 100
            
            berth_stats[berth] = {
                'utilization': utilization,
                'total_gap_hours': total_gap_hours,
                'avg_gap_hours': valid_gaps.mean() if len(valid_gaps) > 0 else 0,
                'num_vessels': len(berth_df)
            }
    
    return berth_stats

def main():
    # Load model and components
    try:
        model_info = joblib.load('models/berth_model.pkl')
        berth_encoder = joblib.load('models/berth_encoder.pkl')
        feature_scaler = joblib.load('models/feature_scaler.pkl')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please train the model first using train_model.py")
        st.stop()
    
    model = model_info['model']
    features = model_info['features']
    
    uploaded_file = st.file_uploader("Upload Vessel Data (CSV)", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['VCN', 'IMO', 'Vessel_Name', 'LOA', 'Port_Code', 'Berth_Code', 'No_of_Teus', 'GRT', 'Actual_Arrival', 'Arrival_at_Berth']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Preprocess data
            df_processed = preprocess_data(df, berth_encoder)
            
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
            X['Vessel_Name_encoded'] = 0  # Use default value for new predictions
            
            # Scale features
            X_scaled = feature_scaler.transform(X[features])
            
            # Make predictions
            predicted_hours = model.predict(X_scaled)
            
            # Calculate predicted departure times
            df_processed['Predicted_Hours_at_Berth'] = predicted_hours
            df_processed['Predicted_Departure'] = pd.to_datetime(df_processed['Arrival_at_Berth']) + \
                pd.to_timedelta(predicted_hours, unit='h')
            
            # Calculate berth utilization
            berth_stats = calculate_berth_utilization(df_processed)
            
            if not berth_stats:
                st.warning("No valid berth data to analyze")
                st.stop()
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "🕒 Predictions", "📈 Analysis"])
            
            with tab1:
                # Display overall metrics
                total_utilization = np.mean([stats['utilization'] for stats in berth_stats.values()])
                avg_gap = np.mean([stats['avg_gap_hours'] for stats in berth_stats.values()])
                total_vessels = sum(stats['num_vessels'] for stats in berth_stats.values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Berth Utilization", f"{total_utilization:.1f}%")
                with col2:
                    st.metric("Average Gap Between Vessels", f"{avg_gap:.1f} hours")
                with col3:
                    st.metric("Total Vessels", total_vessels)
                
                # Show timeline
                st.subheader("Berth Timeline")
                timeline_data = []
                for berth in berth_stats.keys():
                    berth_df = df_processed[df_processed['Berth_Code'] == berth]
                    for _, row in berth_df.iterrows():
                        timeline_data.append({
                            'Berth': berth,
                            'Vessel': row['Vessel_Name'],
                            'Start': row['Arrival_at_Berth'],
                            'End': row['Predicted_Departure'],
                            'Duration': row['Predicted_Hours_at_Berth']
                        })
                
                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)
                    fig = px.timeline(timeline_df, x_start='Start', x_end='End', y='Berth', 
                                   color='Duration', hover_data=['Vessel'],
                                   title="Vessel Berth Occupancy Timeline")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timeline data available")
            
            with tab2:
                st.subheader("Berth-wise Utilization")
                stats_df = pd.DataFrame.from_dict(berth_stats, orient='index')
                st.dataframe(stats_df.round(2))
                
                st.subheader("Vessel Predictions")
                predictions_df = pd.DataFrame({
                    'Vessel_Name': df_processed['Vessel_Name'],
                    'Berth_Code': df_processed['Berth_Code'],
                    'Arrival_Time': df_processed['Arrival_at_Berth'],
                    'Predicted_Departure': df_processed['Predicted_Departure'],
                    'Predicted_Hours': df_processed['Predicted_Hours_at_Berth'].round(2)
                }).sort_values('Arrival_Time')
                
                st.dataframe(predictions_df)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.scatter(df_processed, x='vessel_size_factor', y='Predicted_Hours_at_Berth',
                                    color='Berth_Code', title="Vessel Size vs Predicted Berth Time",
                                    labels={'vessel_size_factor': 'Vessel Size Factor',
                                           'Predicted_Hours_at_Berth': 'Predicted Hours at Berth'})
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.scatter(df_processed, x='cargo_density', y='Predicted_Hours_at_Berth',
                                    color='Berth_Code', title="Cargo Density vs Predicted Berth Time",
                                    labels={'cargo_density': 'TEUs per GRT',
                                           'Predicted_Hours_at_Berth': 'Predicted Hours at Berth'})
                    st.plotly_chart(fig2, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
