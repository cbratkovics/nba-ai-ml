#!/usr/bin/env python3
"""
Initial setup script for NBA AI/ML prediction system
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 10):
        logger.error("Python 3.10 or higher is required")
        sys.exit(1)
    logger.info(f"Python version check passed: {sys.version}")

def check_dependencies():
    """Check if required system dependencies are installed"""
    dependencies = ['docker', 'docker-compose', 'redis-server']
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '--version'], capture_output=True, check=True)
            logger.info(f"✓ {dep} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"⚠ {dep} is not installed - some features may not work")

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'data/raw',
        'data/processed',
        'data/cache',
        'logs',
        'reports',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_environment():
    """Setup environment configuration"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        # Copy example env file
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            content = src.read()
            # Generate random API key salt
            import secrets
            salt = secrets.token_urlsafe(32)
            content = content.replace('your-secret-salt-here', salt)
            dst.write(content)
        
        logger.info("Created .env file from template")
        logger.warning("Please update .env file with your actual API keys and database URLs")
    else:
        logger.info(".env file already exists")

def install_python_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        logger.info("Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Python dependencies: {e}")
        sys.exit(1)

def setup_git_hooks():
    """Setup git pre-commit hooks"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pre-commit'], check=True)
        subprocess.run(['pre-commit', 'install'], check=True)
        logger.info("Git pre-commit hooks installed")
    except subprocess.CalledProcessError:
        logger.warning("Failed to setup git hooks - continuing without them")

def setup_mlflow():
    """Initialize MLflow tracking"""
    try:
        import mlflow
        
        # Create MLflow directory
        mlflow_dir = Path('mlruns')
        mlflow_dir.mkdir(exist_ok=True)
        
        # Set tracking URI
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        logger.info("MLflow tracking initialized")
    except ImportError:
        logger.warning("MLflow not available - install requirements.txt first")

def validate_setup():
    """Validate that setup was successful"""
    checks = []
    
    # Check directories
    required_dirs = ['models', 'data', 'logs']
    for directory in required_dirs:
        if Path(directory).exists():
            checks.append(f"✓ Directory {directory} exists")
        else:
            checks.append(f"✗ Directory {directory} missing")
    
    # Check .env file
    if Path('.env').exists():
        checks.append("✓ Environment file configured")
    else:
        checks.append("✗ Environment file missing")
    
    # Check Python imports
    try:
        import fastapi, pandas, sklearn, xgboost
        checks.append("✓ Core dependencies importable")
    except ImportError as e:
        checks.append(f"✗ Import error: {e}")
    
    logger.info("Setup validation:")
    for check in checks:
        logger.info(f"  {check}")
    
    # Overall status
    if all("✓" in check for check in checks):
        logger.info("🎉 Setup completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Update .env file with your API keys")
        logger.info("  2. Run: python scripts/collect_historical_data.py")
        logger.info("  3. Run: make train")
        logger.info("  4. Run: make serve")
    else:
        logger.warning("⚠ Setup completed with some issues - check the logs above")

def main():
    """Main setup function"""
    logger.info("🚀 Starting NBA AI/ML system setup...")
    
    try:
        check_python_version()
        check_dependencies()
        create_directories()
        setup_environment()
        install_python_dependencies()
        setup_git_hooks()
        setup_mlflow()
        validate_setup()
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()