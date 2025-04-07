import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import numpy as np

# Configure page settings and theme
st.set_page_config(
    page_title="Berth Utilization Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent white theme
st.markdown("""
    <style>
        /* Main app background and text */
        .stApp {
            background-color: #FFFFFF;
            color: #2C3E50;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2C3E50 !important;
            font-weight: 600 !important;
        }
        
        /* Dataframe styling */
        .dataframe {
            background-color: #FFFFFF !important;
            color: #2C3E50 !important;
            border-radius: 5px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        /* Plotly chart background */
        .stPlotlyChart > div {
            background-color: #FFFFFF !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
            padding: 10px !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F8FAFC;
            padding: 10px 10px 0 10px;
            border-radius: 10px 10px 0 0;
            box-shadow: 0 -1px 0 rgba(0,0,0,0.1) inset;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #EDF2F7;
            border-radius: 8px 8px 0 0;
            gap: 8px;
            padding: 8px 16px;
            font-weight: 600;
            color: #000000 !important;
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-bottom: 3px solid #000000 !important;
        }
        
        /* Inputs and widgets */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stDateInput > div > div > input {
            color: #2C3E50 !important;
            background-color: #FFFFFF !important;
            border-radius: 6px !important;
            border: 1px solid #E2E8F0 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #2C3E50 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #34495E !important;
            transform: translateY(-1px);
        }
        
        /* Sidebar */
        .css-1d391kg, .css-12oz5g7 {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        
        /* Tables and DataFrames - General */
        div[data-testid="stTable"] table,
        div[data-testid="stDataFrame"] table,
        .dataframe,
        .streamlit-table {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
            border: 1px solid #E2E8F0 !important;
            width: 100% !important;
        }

        /* Reset any default table styling */
        div[data-testid="stTable"] table *,
        div[data-testid="stDataFrame"] table *,
        .dataframe *,
        .streamlit-table * {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        /* Force white background on all table containers */
        div[data-testid="stTable"],
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stTable"] > div,
        [data-testid="stDataFrame"] div[data-testid="stDataFrameContainer"],
        .element-container [data-testid="stDataFrame"],
        .element-container [data-testid="stTable"] {
            background-color: #FFFFFF !important;
        }
        
        /* All Table Headers */
        div[data-testid="stTable"] th,
        div[data-testid="stDataFrame"] th,
        .dataframe thead th,
        .dataframe tbody tr th,
        [data-testid="stDataFrame"] [role="columnheader"],
        div[class*="Toastify"] div[role="columnheader"],
        div[class*="Table"] div[role="columnheader"],
        .stDataFrame thead tr th {
            background-color: #F8FAFC !important;
            color: #000000 !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 12px !important;
            border-bottom: 2px solid #E2E8F0 !important;
            white-space: nowrap !important;
        }
            background-color: #F8FAFC !important;
            color: #2C3E50 !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 12px !important;
            border-bottom: 2px solid #E2E8F0 !important;
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #E2E8F0 !important;
            padding: 8px !important;
            text-align: left !important;
        }
        
        /* All Table Cells */
        div[data-testid="stTable"] td,
        div[data-testid="stDataFrame"] td,
        .dataframe tbody tr td,
        [data-testid="stDataFrame"] [role="cell"],
        div[class*="Toastify"] div[role="cell"],
        div[class*="Table"] div[role="cell"],
        .streamlit-table td,
        .streamlit-expanderContent td,
        .stDataFrame tbody tr td {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-bottom: 1px solid #E2E8F0 !important;
            padding: 12px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            white-space: nowrap !important;
        }
            background-color: #FFFFFF !important;
            color: #2C3E50 !important;
            border-bottom: 1px solid #E2E8F0 !important;
            padding: 12px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
        }
        
        /* Table Row Hover */
        div[data-testid="stTable"] tr:hover td,
        div[data-testid="stDataFrame"] tr:hover td,
        .dataframe tbody tr:hover td,
        [data-testid="stDataFrame"] [role="row"]:hover [role="cell"],
        .stDataFrame tbody tr:hover td {
            background-color: #F8FAFC !important;
            color: #000000 !important;
            transition: all 0.2s ease !important;
        }
            background-color: #F8FAFC !important;
        }
        
        /* Table Styles Reset */
        .dataframe {
            border: none !important;
            border-collapse: collapse !important;
            width: 100% !important;
        }
        
        /* Remove Default Streamlit Table Styles */
        div[data-testid="StyledFullScreenFrame"] > div > div > div > div > div {
            background-color: #FFFFFF !important;
        }
        
        /* Make sure text is visible in all states */
        div[data-testid="stText"],
        div[data-testid="stMarkdown"] {
            color: #2C3E50 !important;
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
        vessel_encoder = joblib.load('models/vessel_encoder.pkl')
        return model_info, berth_encoder, feature_scaler, vessel_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def preprocess_data(df, berth_encoder, vessel_encoder):
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
    
    # Encode categorical variables
    df['Berth_Code_encoded'] = berth_encoder.transform(df['Berth_Code'])
    
    # Handle vessel encoding - fit if not already fit
    try:
        df['Vessel_Name_encoded'] = vessel_encoder.transform(df['Vessel_Name'])
    except ValueError:
        # If vessel encoder hasn't seen these categories, fit it first
        vessel_encoder.fit(df['Vessel_Name'])
        df['Vessel_Name_encoded'] = vessel_encoder.transform(df['Vessel_Name'])
    
    return df

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
        model_info, berth_encoder, feature_scaler, vessel_encoder = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please train the model first using train_model.py")
        st.stop()
    
    model = model_info['model']
    features = model_info['features']
    
    # Handle compatibility with newer scikit-learn versions
    if hasattr(model, 'estimators_'):
        for estimator in model.estimators_:
            if not hasattr(estimator, 'monotonic_cst'):
                estimator.monotonic_cst = None
    
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
            tab1, tab2, tab3, tab4 = st.tabs([
                "Overview",
                "Analysis",
                "Insights",
                "Future Forecast"
            ])
            
            with tab1:
                # Display overall metrics
                total_utilization = np.mean([stats['utilization'] for stats in berth_stats.values()])
                avg_gap = np.mean([stats['avg_gap_hours'] for stats in berth_stats.values()])
                total_vessels = sum(stats['num_vessels'] for stats in berth_stats.values())
                
                # Custom CSS for black text in metrics
                st.markdown("""
                <style>
                [data-testid="stMetricLabel"] {
                    color: #000000 !important;
                }
                [data-testid="stMetricValue"] {
                    color: #000000 !important;
                }
                </style>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Berth Utilization", f"{total_utilization:.1f}%")
                with col2:
                    st.metric("Average Gap Between Vessels", f"{avg_gap:.1f} hours")
                with col3:
                    st.metric("Total Vessels", total_vessels)
                
                # Show timeline
                st.subheader("Berth Timeline")
                
                # Prepare timeline data
                timeline_data = []
                try:
                    # Convert datetime columns first
                    df_processed['Arrival_at_Berth'] = pd.to_datetime(df_processed['Arrival_at_Berth'])
                    df_processed['Predicted_Departure'] = pd.to_datetime(df_processed['Predicted_Departure'])
                    
                    # Sort by berth and arrival time
                    df_timeline = df_processed.sort_values(['Berth_Code', 'Arrival_at_Berth'])
                    
                    for berth in sorted(df_timeline['Berth_Code'].unique()):
                        berth_df = df_timeline[df_timeline['Berth_Code'] == berth]
                        for _, row in berth_df.iterrows():
                            if pd.notna(row['Arrival_at_Berth']) and pd.notna(row['Predicted_Departure']):
                                timeline_data.append({
                                    'Berth': berth,
                                    'Vessel': row['Vessel_Name'],
                                    'Start': row['Arrival_at_Berth'],
                                    'End': row['Predicted_Departure'],
                                    'Duration': float(row['Predicted_Hours_at_Berth'])
                                })
                    
                    if timeline_data:
                        # Create timeline visualization
                        timeline_df = pd.DataFrame(timeline_data)
                        
                        # Debug info
                        st.write(f"Number of timeline entries: {len(timeline_df)}")
                        st.write("Sample of timeline data:")
                        st.write(timeline_df.head())
                        
                        # Create Gantt chart
                        fig = px.timeline(
                            timeline_df,
                            x_start='Start',
                            x_end='End',
                            y='Berth',
                            color='Duration',
                            hover_data=['Vessel'],
                            title="Vessel Berth Occupancy Timeline",
                            color_continuous_scale='Viridis'
                        )
                        
                        # Update timeline layout
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='black', size=12),
                            title=dict(
                                text="Vessel Berth Occupancy Timeline",
                                font=dict(color='black', size=16)
                            ),
                            xaxis=dict(
                                title='Timeline',
                                showgrid=True,
                                gridcolor='rgba(0,0,0,0.1)',
                                title_font=dict(color='black'),
                                tickfont=dict(color='black'),
                                rangeslider_visible=True
                            ),
                            yaxis=dict(
                                title='Berth',
                                showgrid=True,
                                gridcolor='rgba(0,0,0,0.1)',
                                title_font=dict(color='black'),
                                tickfont=dict(color='black')
                            ),
                            coloraxis_colorbar=dict(
                                title='Duration (hours)',
                                title_font=dict(color='black'),
                                tickfont=dict(color='black')
                            ),
                            height=400  # Fixed height for better visibility
                        )
                        
                        # Show the timeline
                        st.plotly_chart(fig, use_container_width=True, key="berth_timeline")
                    else:
                        st.warning("No valid timeline data available.")
                        
                except Exception as e:
                    st.error(f"Error creating timeline: {str(e)}")
                    st.exception(e)

                    


                else:
                    # Skip if no timeline data
                    pass
            
            with tab4:
                st.subheader("Future Berth Occupancy Forecast (7-14 Days)")
                
                # Generate future predictions
                future_predictions = generate_future_predictions(
                    df_processed, model_info, feature_scaler)
                
                # Create timeline for future predictions
                future_timeline = []
                for _, row in future_predictions.iterrows():
                    start_time = row['prediction_date']
                    end_time = start_time + timedelta(hours=row['predicted_hours'])
                    
                    future_timeline.append({
                        'Berth': row['Berth_Code'],
                        'Vessel': row['Vessel_Name'],
                        'Start': start_time,
                        'End': end_time,
                        'Duration': row['predicted_hours'],
                        'Lower_CI': row['lower_bound'],
                        'Upper_CI': row['upper_bound']
                    })
                
                if future_timeline:
                    timeline_df = pd.DataFrame(future_timeline)
                    
                    # Sort timeline by berth and start time
                    timeline_df = timeline_df.sort_values(['Berth', 'Start'])
                    
                    # Create Gantt chart
                    fig = px.timeline(timeline_df,
                                    x_start="Start",
                                    x_end="End",
                                    y="Berth",
                                    color="Vessel",
                                    hover_name="Vessel",
                                    hover_data={
                                        "Start": True,
                                        "Duration": ":.1f",
                                        "Lower_CI": ":.1f",
                                        "Upper_CI": ":.1f",
                                    },
                                    color_discrete_sequence=px.colors.qualitative.Set3)
                    
                    # Update layout for better readability
                    fig.update_layout(
                        title=dict(
                            text="Future Berth Occupancy Forecast",
                            font=dict(color='black', size=16),
                            x=0.5,
                            xanchor='center'
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=14, color='black'),
                        xaxis=dict(
                            title='Timeline',
                            title_font=dict(color='black', size=14),
                            tickfont=dict(color='black', size=12),
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            type='date'
                        ),
                        yaxis=dict(
                            title='Berth',
                            title_font=dict(color='black', size=14),
                            tickfont=dict(color='black', size=12),
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            zeroline=False
                        ),
                        width=800,
                        height=500,  # Increased height for better spacing
                        showlegend=True,
                        legend=dict(
                            title='Vessels',
                            yanchor='top',
                            y=0.99,
                            xanchor='right',
                            x=0.99,
                            bgcolor='rgba(255,255,255,0.9)'
                        ),
                        hovermode='closest'
                    )
                    
                    # Add confidence interval annotations with better spacing
                    for _, row in timeline_df.iterrows():
                        fig.add_annotation(
                            x=row['Start'] + pd.Timedelta(hours=row['Duration']/2),
                            y=row['Berth'],
                            text=f"CI: [{row['Lower_CI']:.1f}-{row['Upper_CI']:.1f}h",
                            showarrow=False,
                            yshift=20,
                            font=dict(size=11, color='#2C3E50'),
                            bgcolor='rgba(255,255,255,0.95)',
                            bordercolor='rgba(44, 62, 80, 0.2)',
                            borderwidth=1,
                            borderpad=4
                        )
                    
                    # Display the future forecast chart
                    st.plotly_chart(fig, use_container_width=True, key="future_forecast")
                    
                    # Display detailed predictions table
                    st.subheader("Detailed Future Predictions")
                    detailed_df = pd.DataFrame({
                        'Berth': timeline_df['Berth'],
                        'Vessel': timeline_df['Vessel'],
                        'Start Time': timeline_df['Start'],
                        'Predicted Hours': timeline_df['Duration'].round(1),
                        'Lower Bound (hrs)': timeline_df['Lower_CI'].round(1),
                        'Upper Bound (hrs)': timeline_df['Upper_CI'].round(1)
                    }).sort_values('Start Time')
                    
                    st.dataframe(
                        detailed_df.style
                        .set_table_styles([
                            {'selector': 'th', 'props': [('background-color', '#FFFFFF'), ('color', '#000000')]},
                            {'selector': 'td', 'props': [('background-color', '#FFFFFF'), ('color', '#000000')]}
                        ])
                        .format({
                            'Predicted Hours': '{:.2f}',
                            'Start Time': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else '',
                            'Lower Bound (hrs)': '{:.2f}',
                            'Upper Bound (hrs)': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No future predictions available")
            
            with tab2:
                st.subheader("Berth Analysis")
                

                
                # Show berth statistics
                st.subheader("Berth-wise Statistics")
                stats_df = pd.DataFrame.from_dict(berth_stats, orient='index')
                st.dataframe(
                    stats_df.style
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#F8FAFC'), ('color', '#2C3E50')]},
                        {'selector': 'td', 'props': [('background-color', '#FFFFFF'), ('color', '#2C3E50')]}
                    ])
                    .format(precision=2),
                    use_container_width=True
                )
                
                # Show vessel predictions
                st.subheader("Vessel Schedule")
                predictions_df = pd.DataFrame({
                    'VCN': df['VCN'],
                    'IMO': df['IMO'],
                    'Vessel_Name': df_processed['Vessel_Name'],
                    'LOA': df['LOA'],
                    'Port_Code': df['Port_Code'],
                    'Berth_Code': df_processed['Berth_Code'],
                    'No_of_Teus': df['No_of_Teus'],
                    'GRT': df['GRT'],
                    'Actual_Arrival': pd.to_datetime(df['Actual_Arrival']),
                    'Arrival_at_Berth': pd.to_datetime(df['Arrival_at_Berth']),
                    'Predicted_Departure': df_processed['Predicted_Departure'],
                    'Predicted_Hours': df_processed['Predicted_Hours_at_Berth'].round(2)
                }).sort_values('Arrival_at_Berth')
                
                # Add custom CSS for table styling
                st.markdown("""
                <style>
                [data-testid="stDataFrame"] {
                    background-color: white;
                }
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"] {
                    color: black !important;
                    background-color: white !important;
                }
                [data-testid="stDataFrame"] div[data-testid="stDataFrameHeaderCell"] {
                    color: black !important;
                    font-weight: bold !important;
                    background-color: #f8f9fa !important;
                }
                /* Subtle color coding for different data types */
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="VCN"],
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="IMO"] {
                    border-left: 4px solid #007bff;
                }
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="Vessel_Name"],
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="LOA"] {
                    border-left: 4px solid #28a745;
                }
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="No_of_Teus"],
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="GRT"] {
                    border-left: 4px solid #dc3545;
                }
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="Actual_Arrival"],
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="Arrival_at_Berth"] {
                    border-left: 4px solid #6f42c1;
                }
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="Predicted_Departure"],
                [data-testid="stDataFrame"] div[data-testid="stDataFrameDataCell"][data-column="Predicted_Hours"] {
                    border-left: 4px solid #fd7e14;
                }
                </style>
                """, unsafe_allow_html=True)

                st.dataframe(
                    predictions_df.style
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#F8FAFC'), ('color', '#2C3E50')]},
                        {'selector': 'td', 'props': [('background-color', '#FFFFFF'), ('color', '#2C3E50')]}
                    ])
                    .format({
                        'VCN': lambda x: str(x),
                        'IMO': lambda x: str(x),
                        'LOA': '{:.1f}',
                        'GRT': '{:.0f}',
                        'No_of_Teus': '{:.0f}',
                        'Actual_Arrival': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else '',
                        'Arrival_at_Berth': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else '',
                        'Predicted_Departure': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else ''
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'VCN': st.column_config.TextColumn('VCN', help='Vessel Call Number'),
                        'IMO': st.column_config.TextColumn('IMO', help='IMO Number'),
                        'Vessel_Name': st.column_config.TextColumn('Vessel Name', help='Name of the Vessel'),
                        'LOA': st.column_config.NumberColumn('LOA (m)', help='Length Overall in meters', format='%.1f'),
                        'Port_Code': st.column_config.TextColumn('Port Code', help='Port Identifier'),
                        'Berth_Code': st.column_config.TextColumn('Berth Code', help='Berth Identifier'),
                        'No_of_Teus': st.column_config.NumberColumn('TEUs', help='Number of Twenty-foot Equivalent Units'),
                        'GRT': st.column_config.NumberColumn('GRT', help='Gross Registered Tonnage'),
                        'Actual_Arrival': st.column_config.DatetimeColumn('Actual Arrival', help='Actual arrival time at port', format='DD/MM/YYYY HH:mm'),
                        'Arrival_at_Berth': st.column_config.DatetimeColumn('Arrival at Berth', help='Time of arrival at berth', format='DD/MM/YYYY HH:mm'),
                        'Predicted_Departure': st.column_config.DatetimeColumn('Predicted Departure', help='Predicted departure time', format='DD/MM/YYYY HH:mm')
                    }
                )   
            
            with tab3:
                st.subheader("Berth Performance Insights")
                
                # Calculate daily metrics
                df_processed['Date'] = pd.to_datetime(df_processed['Arrival_at_Berth']).dt.date
                
                # Calculate daily utilization for each berth
                daily_stats = []
                unique_dates = sorted(df_processed['Date'].unique())
                
                for date in unique_dates:
                    date_df = df_processed[df_processed['Date'] == date]
                    date_stats = {'Date': date}
                    
                    # Calculate utilization
                    total_gap_hours = 0
                    gap_count = 0
                    vessel_count = 0
                    total_berth_hours = 0
                    num_berths = len(df_processed['Berth_Code'].unique())
                    
                    # Calculate total available hours for all berths on this date
                    total_available_hours = 24 * num_berths
                    
                    for berth in df_processed['Berth_Code'].unique():
                        berth_df = date_df[date_df['Berth_Code'] == berth]
                        if not berth_df.empty:
                            vessel_count += len(berth_df)
                            
                            # Sort by arrival time and ensure datetime format
                            berth_df = berth_df.sort_values('Arrival_at_Berth')
                            berth_df['Arrival_at_Berth'] = pd.to_datetime(berth_df['Arrival_at_Berth'])
                            berth_df['Predicted_Departure'] = pd.to_datetime(berth_df['Predicted_Departure'])
                            
                            # Get start and end of day for this date
                            day_start = pd.to_datetime(date).replace(hour=0, minute=0, second=0)
                            day_end = day_start + pd.Timedelta(days=1)
                            
                            # Calculate actual berth hours within this day only
                            berth_hours = 0
                            
                            for _, vessel in berth_df.iterrows():
                                arrival = max(vessel['Arrival_at_Berth'], day_start)
                                departure = min(vessel['Predicted_Departure'], day_end)
                                
                                # Only count if vessel was present during this day
                                if arrival < day_end and departure > day_start:
                                    hours = (departure - arrival).total_seconds() / 3600
                                    berth_hours += min(hours, 24)  # Cap at 24 hours per vessel per day
                            
                            total_berth_hours += min(berth_hours, 24)  # Cap at 24 hours per berth
                            
                            # Calculate gaps between vessels
                            if len(berth_df) > 1:
                                for i in range(len(berth_df)-1):
                                    current_departure = berth_df['Predicted_Departure'].iloc[i]
                                    next_arrival = berth_df['Arrival_at_Berth'].iloc[i+1]
                                    
                                    # Only consider gaps within the same day
                                    if current_departure >= day_start and next_arrival <= day_end:
                                        gap = (next_arrival - current_departure).total_seconds() / 3600
                                        if gap > 0:
                                            total_gap_hours += gap
                                            gap_count += 1
                
                    # Calculate utilization percentage (capped at 100%)
                    utilization = (total_berth_hours / total_available_hours) * 100
                    date_stats['Utilization'] = min(utilization, 100)  # Cap at 100%
                    
                    # Store detailed stats
                    date_stats.update({
                        'Total_Berth_Hours': round(total_berth_hours, 2),
                        'Available_Hours': total_available_hours,
                        'Utilization': round(utilization, 2),
                        'Num_Berths': num_berths,
                        'Vessel_Count': vessel_count,
                        'Average_Gap': round(total_gap_hours / gap_count if gap_count > 0 else 0, 2)
                    })
                    date_stats['Average_Gap'] = total_gap_hours / gap_count if gap_count > 0 else 0
                    date_stats['Vessel_Count'] = vessel_count
                    daily_stats.append(date_stats)
            
                daily_df = pd.DataFrame(daily_stats)
                
                # Display detailed statistics table
                st.markdown("### Daily Berth Statistics")
                stats_df = daily_df[['Date', 'Total_Berth_Hours', 'Available_Hours', 'Utilization', 'Num_Berths', 'Vessel_Count', 'Average_Gap']]
                stats_df.columns = ['Date', 'Total Berth Hours', 'Available Hours', 'Utilization (%)', 'Number of Berths', 'Number of Vessels', 'Average Gap (hrs)']
                st.dataframe(
                    stats_df.style
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#F8FAFC'), ('color', '#2C3E50')]},
                        {'selector': 'td', 'props': [('background-color', '#FFFFFF'), ('color', '#2C3E50')]}
                    ])
                    .format({
                        'Total Berth Hours': '{:.2f}',
                        'Utilization (%)': '{:.2f}',
                        'Average Gap (hrs)': '{:.2f}',
                        'Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Create visualizations
                st.markdown("### Utilization and Gap Trends")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create utilization trend chart
                    fig1 = go.Figure()
                    
                    fig1.add_trace(go.Scatter(
                        x=daily_df['Date'],
                        y=daily_df['Utilization'],
                        mode='lines+markers',
                        name='Daily Utilization',
                        line=dict(width=2, color='#1f77b4'),
                        marker=dict(
                            size=8,
                            color='#1f77b4',
                            line=dict(width=2, color='black')
                        )
                    ))
                    
                    fig1.update_layout(
                        title=dict(
                            text='Daily Berth Utilization Trend',
                            font=dict(size=18, color='black', family='Arial Black')
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=14, color='black', family='Arial'),
                        showlegend=False,
                        xaxis=dict(
                            title='Date',
                            title_font=dict(size=14, color='black', family='Arial Black'),
                            tickfont=dict(size=12, color='black'),
                            showgrid=False,
                            showline=True,
                            linewidth=1,
                            linecolor='black'
                        ),
                        yaxis=dict(
                            title='Utilization (%)',
                            title_font=dict(size=14, color='black', family='Arial Black'),
                            tickfont=dict(size=12, color='black'),
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            showline=True,
                            linewidth=1,
                            linecolor='black'
                        )
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Create average gap trend chart
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=daily_df['Date'],
                        y=daily_df['Average_Gap'],
                        mode='lines+markers',
                        name='Average Gap',
                        line=dict(width=2, color='#2ca02c'),
                        marker=dict(
                            size=8,
                            color='#2ca02c',
                            line=dict(width=2, color='black')
                        )
                    ))
                    
                    fig2.update_layout(
                        title=dict(
                            text='Daily Average Gap Between Vessels',
                            font=dict(size=18, color='black', family='Arial Black')
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=14, color='black', family='Arial'),
                        showlegend=False,
                        xaxis=dict(
                            title='Date',
                            title_font=dict(size=14, color='black', family='Arial Black'),
                            tickfont=dict(size=12, color='black'),
                            showgrid=False,
                            showline=True,
                            linewidth=1,
                            linecolor='black'
                        ),
                        yaxis=dict(
                            title='Average Gap (hours)',
                            title_font=dict(size=14, color='black', family='Arial Black'),
                            tickfont=dict(size=12, color='black'),
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            showline=True,
                            linewidth=1,
                            linecolor='black'
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                

                

                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
