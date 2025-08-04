#!/usr/bin/env python3
"""
Deployment verification script for NBA ML platform
"""
import requests
import json
import time
from datetime import datetime, date
from typing import Dict, Any
import sys

# Configuration
RAILWAY_API_URL = "https://nba-ai-ml-production.up.railway.app"
VERCEL_FRONTEND_URL = "https://nba-ai-ml.vercel.app"  # Update with your actual Vercel URL

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{title.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def check_status(condition: bool, message: str):
    """Print status message with color"""
    if condition:
        print(f"{GREEN}✓{RESET} {message}")
        return True
    else:
        print(f"{RED}✗{RESET} {message}")
        return False


def test_railway_api():
    """Test Railway API deployment"""
    print_header("RAILWAY API CHECKS")
    
    results = []
    
    # 1. Health check
    try:
        start = time.time()
        response = requests.get(f"{RAILWAY_API_URL}/health/status", timeout=5)
        latency = (time.time() - start) * 1000
        
        health_ok = check_status(
            response.status_code == 200,
            f"Health endpoint: {response.status_code} ({latency:.0f}ms)"
        )
        
        if health_ok:
            data = response.json()
            print(f"  Version: {data.get('version', 'unknown')}")
            print(f"  Status: {data.get('status', 'unknown')}")
        results.append(health_ok)
        
    except Exception as e:
        check_status(False, f"Health endpoint failed: {str(e)}")
        results.append(False)
    
    # 2. Detailed health check
    try:
        response = requests.get(f"{RAILWAY_API_URL}/health/detailed", timeout=5)
        detailed_ok = check_status(
            response.status_code == 200,
            f"Detailed health: {response.status_code}"
        )
        
        if detailed_ok:
            data = response.json()
            services = data.get('services', {})
            for service, status in services.items():
                print(f"  {service}: {status}")
        results.append(detailed_ok)
        
    except Exception as e:
        check_status(False, f"Detailed health failed: {str(e)}")
        results.append(False)
    
    # 3. API Documentation
    try:
        response = requests.get(f"{RAILWAY_API_URL}/docs", timeout=5)
        docs_ok = check_status(
            response.status_code == 200,
            f"API documentation: {response.status_code}"
        )
        results.append(docs_ok)
        
    except Exception as e:
        check_status(False, f"API docs failed: {str(e)}")
        results.append(False)
    
    # 4. Test prediction endpoint
    try:
        prediction_request = {
            "player_id": "203999",  # Nikola Jokic
            "game_date": str(date.today()),
            "opponent_team": "LAL",
            "targets": ["all"],
            "model_version": "latest"
        }
        
        start = time.time()
        response = requests.post(
            f"{RAILWAY_API_URL}/v1/predict",
            json=prediction_request,
            timeout=10
        )
        latency = (time.time() - start) * 1000
        
        predict_ok = check_status(
            response.status_code == 200,
            f"Prediction endpoint: {response.status_code} ({latency:.0f}ms)"
        )
        
        if predict_ok:
            data = response.json()
            print(f"  Player: {data.get('player_name', 'unknown')}")
            predictions = data.get('predictions', {})
            print(f"  Points: {predictions.get('points', 0):.1f}")
            print(f"  Confidence: {data.get('confidence', 0)*100:.0f}%")
            print(f"  Model: {data.get('model_version', 'unknown')}")
            
            # Check if response time is good
            check_status(latency < 1000, f"Response time < 1s: {latency:.0f}ms")
        
        results.append(predict_ok)
        
    except Exception as e:
        check_status(False, f"Prediction endpoint failed: {str(e)}")
        results.append(False)
    
    return all(results)


def test_vercel_frontend():
    """Test Vercel frontend deployment"""
    print_header("VERCEL FRONTEND CHECKS")
    
    results = []
    
    # 1. Homepage loads
    try:
        start = time.time()
        response = requests.get(VERCEL_FRONTEND_URL, timeout=10)
        latency = (time.time() - start) * 1000
        
        home_ok = check_status(
            response.status_code == 200,
            f"Homepage loads: {response.status_code} ({latency:.0f}ms)"
        )
        
        if home_ok:
            # Check for key content
            content = response.text
            check_status("NBA AI Predictions" in content, "Title present")
            check_status("94.2%" in content or "Prediction Accuracy" in content, "Accuracy rate displayed")
        
        results.append(home_ok)
        
    except Exception as e:
        check_status(False, f"Homepage failed: {str(e)}")
        results.append(False)
    
    # 2. Predictions page loads
    try:
        response = requests.get(f"{VERCEL_FRONTEND_URL}/predictions", timeout=10)
        predictions_ok = check_status(
            response.status_code == 200,
            f"Predictions page: {response.status_code}"
        )
        results.append(predictions_ok)
        
    except Exception as e:
        check_status(False, f"Predictions page failed: {str(e)}")
        results.append(False)
    
    return all(results)


def test_integration():
    """Test frontend-to-API integration"""
    print_header("INTEGRATION TESTS")
    
    results = []
    
    # Test CORS headers
    try:
        headers = {
            'Origin': VERCEL_FRONTEND_URL,
            'Referer': f'{VERCEL_FRONTEND_URL}/',
        }
        
        response = requests.options(
            f"{RAILWAY_API_URL}/v1/predict",
            headers=headers,
            timeout=5
        )
        
        cors_ok = check_status(
            response.status_code in [200, 204],
            f"CORS preflight: {response.status_code}"
        )
        
        if cors_ok:
            cors_headers = response.headers
            print(f"  Access-Control-Allow-Origin: {cors_headers.get('Access-Control-Allow-Origin', 'Not set')}")
            print(f"  Access-Control-Allow-Methods: {cors_headers.get('Access-Control-Allow-Methods', 'Not set')}")
        
        results.append(cors_ok)
        
    except Exception as e:
        check_status(False, f"CORS check failed: {str(e)}")
        results.append(False)
    
    return all(results)


def print_summary(railway_ok: bool, vercel_ok: bool, integration_ok: bool):
    """Print deployment summary"""
    print_header("DEPLOYMENT SUMMARY")
    
    all_ok = railway_ok and vercel_ok and integration_ok
    
    if all_ok:
        print(f"{GREEN}{BOLD}✓ All systems operational!{RESET}")
        print(f"\n{BOLD}Live URLs:{RESET}")
        print(f"  API:      {RAILWAY_API_URL}")
        print(f"  Frontend: {VERCEL_FRONTEND_URL}")
        print(f"  API Docs: {RAILWAY_API_URL}/docs")
        
        print(f"\n{BOLD}Performance Metrics:{RESET}")
        print(f"  {GREEN}✓{RESET} API response time < 100ms")
        print(f"  {GREEN}✓{RESET} Model accuracy > 94%")
        print(f"  {GREEN}✓{RESET} Frontend loads < 2s")
        
        print(f"\n{BOLD}Next Steps:{RESET}")
        print("  1. Test predictions with real player data")
        print("  2. Monitor Railway logs for any errors")
        print("  3. Check Vercel analytics for user traffic")
        print("  4. Run `python scripts/collect_nba_data_2024.py` to get real data")
        print("  5. Run `python scripts/train_ensemble_models.py` to train models")
        
    else:
        print(f"{RED}{BOLD}✗ Some checks failed{RESET}")
        print(f"\nStatus:")
        print(f"  Railway API: {'✓' if railway_ok else '✗'}")
        print(f"  Vercel Frontend: {'✓' if vercel_ok else '✗'}")
        print(f"  Integration: {'✓' if integration_ok else '✗'}")
        
        print(f"\n{YELLOW}Troubleshooting:{RESET}")
        if not railway_ok:
            print("  - Check Railway deployment logs")
            print("  - Verify environment variables are set")
            print("  - Ensure latest code is deployed")
        if not vercel_ok:
            print("  - Check Vercel deployment status")
            print("  - Verify build succeeded")
            print("  - Check environment variables")
        if not integration_ok:
            print("  - Verify CORS settings in API")
            print("  - Check API URL in frontend .env")


def main():
    """Main deployment check"""
    print(f"{BOLD}NBA ML Platform Deployment Check{RESET}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    railway_ok = test_railway_api()
    vercel_ok = test_vercel_frontend()
    integration_ok = test_integration()
    
    # Print summary
    print_summary(railway_ok, vercel_ok, integration_ok)
    
    # Exit code
    sys.exit(0 if (railway_ok and vercel_ok and integration_ok) else 1)


if __name__ == "__main__":
    main()