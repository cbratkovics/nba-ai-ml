#!/usr/bin/env python3
"""Verify Railway deployment configuration"""
import os
import sys
from pathlib import Path

def check_files():
    """Check required deployment files exist"""
    root = Path.cwd()
    
    required_files = {
        'nixpacks.toml': 'Nixpacks configuration',
        'requirements.txt': 'Python dependencies',
        'railway.json': 'Railway deployment config',
        'runtime.txt': 'Python version specification'
    }
    
    print("🔍 Checking deployment configuration...")
    print("=" * 50)
    
    all_good = True
    for file, description in required_files.items():
        path = root / file
        if path.exists():
            print(f"✅ {file} - {description}")
            # Check file size
            size = path.stat().st_size
            if size == 0:
                print(f"   ⚠️  Warning: {file} is empty")
                all_good = False
        else:
            print(f"❌ {file} missing - {description}")
            all_good = False
    
    print("\n🔍 Checking for conflicting files...")
    print("=" * 50)
    
    # Check for conflicting files
    conflicting_files = ['Dockerfile', 'Procfile', 'docker-compose.yml', 'docker-compose.yaml']
    conflicts_found = False
    for file in conflicting_files:
        if (root / file).exists():
            print(f"⚠️  {file} found - may conflict with Nixpacks")
            conflicts_found = True
    
    if not conflicts_found:
        print("✅ No conflicting files found")
    
    print("\n🔍 Checking Python configuration...")
    print("=" * 50)
    
    # Check Python version in runtime.txt
    runtime_path = root / 'runtime.txt'
    if runtime_path.exists():
        with open(runtime_path, 'r') as f:
            version = f.read().strip()
            print(f"✅ Python version specified: {version}")
    
    # Check nixpacks.toml content
    nixpacks_path = root / 'nixpacks.toml'
    if nixpacks_path.exists():
        with open(nixpacks_path, 'r') as f:
            content = f.read()
            if 'python310' in content:
                print("✅ Nixpacks configured for Python 3.10")
            if 'postgresql' in content:
                print("✅ PostgreSQL dependencies included")
            if 'uvicorn' in content:
                print("✅ Start command configured")
    
    return all_good and not conflicts_found

def check_requirements():
    """Check if requirements.txt has necessary packages"""
    req_path = Path.cwd() / 'requirements.txt'
    if not req_path.exists():
        return False
    
    print("\n🔍 Checking key dependencies...")
    print("=" * 50)
    
    with open(req_path, 'r') as f:
        requirements = f.read().lower()
    
    key_packages = {
        'fastapi': 'Web framework',
        'uvicorn': 'ASGI server',
        'psycopg2': 'PostgreSQL driver',
        'sqlalchemy': 'ORM',
        'redis': 'Cache client'
    }
    
    for package, description in key_packages.items():
        if package in requirements:
            print(f"✅ {package} - {description}")
        else:
            print(f"⚠️  {package} not found - {description}")
    
    return True

if __name__ == "__main__":
    print("\n🚀 Railway Deployment Configuration Verification")
    print("=" * 50)
    
    files_ok = check_files()
    req_ok = check_requirements()
    
    if files_ok and req_ok:
        print("\n✅ Deployment configuration looks good!")
        print("\n📝 Next steps:")
        print("1. git add -A")
        print("2. git commit -m 'Add Nixpacks configuration for Railway Python deployment'")
        print("3. git push")
        print("\n🔗 Railway will automatically redeploy with the new configuration")
    else:
        print("\n❌ Some issues found. Please fix them before deploying.")
        sys.exit(1)