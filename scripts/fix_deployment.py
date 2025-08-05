#!/usr/bin/env python3
"""Emergency deployment fix script"""
import os
import sys
import urllib.parse
from pathlib import Path

def fix_database_url():
    """Fix DATABASE_URL with special characters"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set!")
        return None
    
    # Extract password and encode it
    if '@' in db_url and '://' in db_url:
        try:
            # Parse the URL
            protocol, rest = db_url.split('://', 1)
            if '@' in rest:
                auth, host_part = rest.split('@', 1)
                if ':' in auth:
                    user, password = auth.split(':', 1)
                    # URL encode the password
                    encoded_password = urllib.parse.quote(password, safe='')
                    fixed_url = f"{protocol}://{user}:{encoded_password}@{host_part}"
                    print(f"✅ Fixed DATABASE_URL (password encoded)")
                    return fixed_url
        except Exception as e:
            print(f"❌ Failed to parse DATABASE_URL: {e}")
    
    return db_url

def find_import_issues():
    """Find and report import issues"""
    print("\n🔍 Checking for import issues...")
    
    # Search for problematic imports
    problematic_imports = []
    for py_file in Path('.').rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'from api.models.game_data' in content or 'import api.models.game_data' in content:
                    problematic_imports.append(str(py_file))
        except:
            pass
    
    if problematic_imports:
        print(f"❌ Found {len(problematic_imports)} files with incorrect imports:")
        for file in problematic_imports:
            print(f"   - {file}")
    else:
        print("✅ No problematic imports found")
    
    # Check if api/models.py exists
    if Path('api/models.py').exists():
        print("✅ api/models.py exists")
    else:
        print("❌ api/models.py missing!")

def main():
    print("🚀 NBA AI/ML Deployment Fix Script\n")
    
    # Fix DATABASE_URL
    fixed_url = fix_database_url()
    if fixed_url and fixed_url != os.getenv("DATABASE_URL"):
        print("\n📝 Add this to your Railway environment variables:")
        print(f"DATABASE_URL={fixed_url}")
    
    # Check imports
    find_import_issues()
    
    print("\n✅ Deployment fix analysis complete!")

if __name__ == "__main__":
    main()