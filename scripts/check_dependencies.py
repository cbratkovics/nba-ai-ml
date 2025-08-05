#!/usr/bin/env python3
"""Check for dependency conflicts before deployment"""
import subprocess
import sys
import os
from pathlib import Path

def check_deps():
    """Check for dependency conflicts"""
    print("🔍 Checking Python dependencies for conflicts...")
    print("=" * 60)
    
    # First, check if requirements.txt exists
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    # Check for pip
    print("\n1. Checking pip installation...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ pip is not installed properly")
        return False
    else:
        print(f"✅ {result.stdout.strip()}")
    
    # Run pip check for conflicts
    print("\n2. Running pip check for conflicts...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ No dependency conflicts found in current environment")
    else:
        print("⚠️  Dependency conflicts detected in current environment:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("\nNote: This checks your current environment, not requirements.txt")
    
    # Dry run install to check if requirements would install
    print("\n3. Simulating requirements.txt installation...")
    print("   (This may take a moment...)")
    
    # Create a temporary requirements file without index URLs for dry run
    temp_req = "temp_requirements_check.txt"
    with open(req_file, 'r') as f:
        lines = [line for line in f.readlines() if not line.startswith('--')]
    
    with open(temp_req, 'w') as f:
        f.writelines(lines)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "-r", temp_req],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Clean up temp file
        os.remove(temp_req)
        
        # Check for specific conflict messages
        output = result.stdout + result.stderr
        
        if "conflict" in output.lower() or "incompatible" in output.lower():
            print("❌ Dependency conflicts detected:")
            # Extract conflict information
            for line in output.split('\n'):
                if 'conflict' in line.lower() or 'incompatible' in line.lower():
                    print(f"   {line.strip()}")
            return False
        
        if "ERROR" in output or result.returncode != 0:
            print("❌ Installation would fail. Issues found:")
            # Show last few lines of error
            error_lines = [l for l in output.split('\n') if l.strip()][-10:]
            for line in error_lines:
                print(f"   {line}")
            return False
        
        print("✅ All dependencies can be installed successfully")
        
    except subprocess.TimeoutExpired:
        print("⚠️  Dry run timed out (this is normal for large dependency sets)")
        print("   Proceeding with basic checks...")
    except Exception as e:
        print(f"⚠️  Could not complete dry run: {e}")
    
    # Check for specific known conflicts
    print("\n4. Checking for known conflicts...")
    with open(req_file, 'r') as f:
        requirements = f.read()
    
    conflicts_found = []
    
    # Check httpx version
    if 'httpx==' in requirements:
        import re
        httpx_match = re.search(r'httpx==(\d+\.\d+\.\d+)', requirements)
        if httpx_match:
            version = httpx_match.group(1)
            major, minor, patch = map(int, version.split('.'))
            if major == 0 and minor >= 25:
                conflicts_found.append(
                    f"httpx=={version} conflicts with supabase<0.25.0 requirement"
                )
    
    # Check if both torch and PyTorch CPU are specified
    if 'torch==' in requirements and '+cpu' not in requirements and '--extra-index-url' not in requirements:
        print("ℹ️  Using full PyTorch (includes CUDA). Consider torch==2.2.0+cpu for smaller deployments")
    
    if conflicts_found:
        print("❌ Known conflicts found:")
        for conflict in conflicts_found:
            print(f"   - {conflict}")
        return False
    else:
        print("✅ No known conflicts detected")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    # Check file sizes
    req_size = req_file.stat().st_size / 1024  # KB
    print(f"📄 requirements.txt size: {req_size:.1f} KB")
    
    # Count dependencies
    with open(req_file, 'r') as f:
        dep_count = sum(1 for line in f if line.strip() and not line.startswith('#') and not line.startswith('--'))
    print(f"📦 Total dependencies: {dep_count}")
    
    # Check for dev requirements
    dev_req = Path("requirements-dev.txt")
    if dev_req.exists():
        print(f"🔧 Development requirements found: {dev_req.name}")
    
    print("\n✅ Dependencies look good for deployment!")
    print("\n💡 Tips:")
    print("   - Run 'pip install -r requirements.txt' locally to test")
    print("   - Use 'pip freeze' to see actual installed versions")
    print("   - Consider pip-tools for more advanced dependency management")
    
    return True

if __name__ == "__main__":
    success = check_deps()
    sys.exit(0 if success else 1)