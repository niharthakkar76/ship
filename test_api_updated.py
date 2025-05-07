import requests
import json
import time
import sys

def test_api(base_url):
    """Test the API endpoints"""
    print(f"Testing API at: {base_url}")
    
    # Test health endpoint
    try:
        print("\n1. Testing /health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error testing health endpoint: {str(e)}")
    
    # Test predict endpoint with sample data
    try:
        print("\n2. Testing /predict endpoint...")
        sample_data = {
            "vessels": [
                {
                    "VCN": "VCN12345",
                    "IMO": "IMO9876543",
                    "Vessel_Name": "Mediterranean Queen",
                    "LOA": 320.5,
                    "Port_Code": "SGSIN",
                    "Berth_Code": "BRT001",
                    "No_of_Teus": 7500,
                    "GRT": 95000,
                    "Actual_Arrival": "2025-05-07T10:00:00"
                }
            ]
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{base_url}/predict", 
            data=json.dumps(sample_data),
            headers=headers
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing predict endpoint: {str(e)}")
    
    # Test forecast endpoint
    try:
        print("\n3. Testing /forecast endpoint...")
        forecast_data = {"days_ahead": 7}
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{base_url}/forecast", 
            data=json.dumps(forecast_data),
            headers=headers
        )
        print(f"Status code: {response.status_code}")
        print(f"Response preview: {json.dumps(response.json()['forecast_data'][:2], indent=2)}")
    except Exception as e:
        print(f"Error testing forecast endpoint: {str(e)}")

if __name__ == "__main__":
    # Use command line argument for base URL or default to Railway URL
    # Railway handles the routing to the correct internal port
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://ship-production-5954.up.railway.app"
    
    # Remove trailing slash if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Remove any explicit port if present in the URL
    if ':' in base_url.split('/')[-1]:
        base_url = base_url.rsplit(':', 1)[0]
    
    print("Testing with URL:", base_url)
    
    # Add retry logic
    max_retries = 3
    for i in range(max_retries):
        try:
            test_api(base_url)
            break
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"Connection error. Retrying in 5 seconds... (Attempt {i+1}/{max_retries})")
                time.sleep(5)
            else:
                print("Max retries reached. API might be down or unreachable.")
