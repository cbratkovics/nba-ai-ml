#!/usr/bin/env python3
"""Quick health check to verify all imports work before deployment"""
import sys

def check_imports():
    failed_imports = []
    
    critical_imports = [
        "fastapi", "uvicorn", "pandas", "numpy", "sklearn",
        "xgboost", "lightgbm", "redis", "sqlalchemy", "shap"
    ]
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING!")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nERROR: Missing modules: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ All critical imports successful!")
        sys.exit(0)

if __name__ == "__main__":
    check_imports()