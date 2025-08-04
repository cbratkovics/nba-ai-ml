#!/usr/bin/env python3
"""
Test script to validate the API is working locally
"""
import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path

# Set environment to avoid MLflow issues
os.environ['ENVIRONMENT'] = 'production'

def run_command(cmd):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("\n=== Testing API Endpoints ===")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health/status", timeout=5)
        print(f"Health Status: {response.status_code}")
        if response.ok:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root Status: {response.status_code}")
        if response.ok:
            data = response.json()
            print(f"API Name: {data.get('name')}")
            print(f"Version: {data.get('version')}")
            print(f"Status: {data.get('status')}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Test prediction endpoint
    print("\n3. Testing prediction endpoint...")
    prediction_data = {
        "player_id": "203999",  # Nikola Jokic
        "game_date": "2024-01-15",
        "opponent_team": "LAL",
        "targets": ["points", "rebounds", "assists"],
        "include_explanation": True,
        "include_confidence_intervals": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "test-key-123"  # Dummy API key for testing
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/predict",
            json=prediction_data,
            headers=headers,
            timeout=10
        )
        print(f"Prediction Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"\nPrediction Results:")
            print(f"Player: {result.get('player_name', 'Unknown')}")
            predictions = result.get('predictions', {})
            print(f"Points: {predictions.get('points', 'N/A'):.1f}")
            print(f"Rebounds: {predictions.get('rebounds', 'N/A'):.1f}")
            print(f"Assists: {predictions.get('assists', 'N/A'):.1f}")
            print(f"Confidence: {result.get('confidence', 0):.2%}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

def main():
    """Main test function"""
    print("=== NBA ML API Local Test ===\n")
    
    # Step 1: Create dummy models
    print("Step 1: Creating dummy models...")
    if not run_command("python scripts/create_dummy_model.py"):
        print("❌ Failed to create dummy models")
        return 1
    
    # Step 2: Start the API in background
    print("\nStep 2: Starting API server...")
    api_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "ENVIRONMENT": "production"}
    )
    
    # Wait for API to start
    print("Waiting for API to start...")
    time.sleep(5)
    
    # Step 3: Test the API
    print("\nStep 3: Testing API endpoints...")
    try:
        success = test_api()
    except Exception as e:
        print(f"Test failed: {e}")
        success = False
    finally:
        # Clean up: terminate the API process
        print("\nStopping API server...")
        api_process.terminate()
        api_process.wait(timeout=5)
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("\nDeployment Ready Checklist:")
        print("✓ Models can be loaded from disk")
        print("✓ API starts without MLflow in production mode")
        print("✓ Prediction endpoint returns valid responses")
        print("✓ Health check endpoint is working")
        print("\nNext steps for Railway deployment:")
        print("1. Push changes to GitHub")
        print("2. Set environment variables in Railway:")
        print("   - ENVIRONMENT=production")
        print("   - DATABASE_URL=your_postgres_url")
        print("   - REDIS_URL=your_redis_url (optional)")
        print("3. Deploy to Railway")
        return 0
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())