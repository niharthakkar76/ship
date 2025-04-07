# Berth Utilization Predictor

This Streamlit application predicts vessel departure times and calculates berth utilization based on historical vessel data.

## Features
- Upload CSV files with vessel data
- Predict departure times using machine learning
- Calculate berth utilization
- Visualize actual vs predicted departures
- Display detailed predictions for each vessel

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Input Data Format
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
