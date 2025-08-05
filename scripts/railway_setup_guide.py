#!/usr/bin/env python3
"""Railway deployment setup guide"""

import os
import sys

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def main():
    print("""
🚀 RAILWAY DEPLOYMENT SETUP GUIDE
=================================

This guide will help you configure your Railway deployment properly.
""")

    print_section("1. REQUIRED Environment Variables in Railway")
    
    print("""
DATABASE_URL (REQUIRED):
  - Your Supabase PostgreSQL connection string
  - Find in Supabase: Settings → Database → Connection string (URI)
  - Format: postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
  - ⚠️  Without this, the app will start but all database operations will fail!
""")

    print_section("2. OPTIONAL but Recommended Variables")
    
    print("""
SUPABASE_URL:
  - Format: https://[PROJECT-REF].supabase.co
  - Used for Supabase client operations

SUPABASE_ANON_KEY:
  - Your anon/public key from Supabase
  - Required if using Supabase client features

REDIS_URL:
  - Redis connection string for caching
  - Improves performance significantly
  - Can use Railway's Redis add-on

ENVIRONMENT:
  - Set to 'production' for Railway deployments
  - Default: 'development'

ENABLE_MONITORING:
  - Set to 'true' to enable performance monitoring
  - Default: 'true'

PORT:
  - Railway sets this automatically, don't override
""")

    print_section("3. How to Set Variables in Railway")
    
    print("""
1. Go to your Railway project dashboard
2. Click on your service (nba-ai-ml)
3. Click the "Variables" tab
4. Click "Add Variable" for each one:
   - Variable name: DATABASE_URL
   - Value: [paste your Supabase connection string]
5. Railway will automatically redeploy when you save
""")

    print_section("4. Verify Your Deployment")
    
    print("""
After setting DATABASE_URL and deploying:

1. Check the deployment logs in Railway:
   - Should see "✅ All required environment variables are set"
   - Should see "✅ Database connection successful"

2. Test the health endpoints:
   - https://your-app.railway.app/health/status
   - https://your-app.railway.app/health/detailed

3. The detailed health check will show:
   - Database status (should be "connected")
   - Redis status (if configured)
   - Model status
   - Environment variables status
""")

    print_section("5. Common Issues and Solutions")
    
    print("""
❌ DATABASE_URL not set:
   - App starts but shows "not configured (DATABASE_URL missing)"
   - All database operations fail
   - Solution: Set DATABASE_URL in Railway Variables

❌ Invalid DATABASE_URL format:
   - Check if it starts with postgresql:// (not postgres://)
   - Ensure password is URL-encoded if it contains special characters
   - Test connection string locally first

❌ Connection timeouts:
   - Supabase requires SSL connections
   - Add ?sslmode=require to your DATABASE_URL if needed

❌ Models not found:
   - Run the setup script first: python scripts/setup_ml_models.py
   - Or let the app create dummy models on first run
""")

    print_section("6. Setting Up Monitoring (Optional)")
    
    print("""
For the monitoring service to work:

1. Deploy monitoring as a separate Railway service:
   - Create new service in Railway
   - Point to same GitHub repo
   - Set start command: cd monitoring && uvicorn main:app --host 0.0.0.0 --port $PORT
   - Share the same environment variables

2. Required variables for monitoring:
   - DATABASE_URL (same as main app)
   - REDIS_URL (if using Redis)
   - ALERT_WEBHOOK_URL (for Slack/Discord alerts)
""")

    print_section("7. Quick Checklist")
    
    print("""
□ DATABASE_URL is set in Railway Variables
□ Deployment logs show successful database connection
□ /health/detailed endpoint shows all services
□ Can make predictions via API
□ (Optional) Redis configured for caching
□ (Optional) Monitoring service deployed
""")

    # Check current environment
    print_section("Current Environment Check")
    
    if os.getenv("DATABASE_URL"):
        print("✅ DATABASE_URL is set locally")
    else:
        print("❌ DATABASE_URL is not set locally")
    
    if os.getenv("RAILWAY_ENVIRONMENT"):
        print("✅ Running on Railway")
        print(f"   Region: {os.getenv('RAILWAY_REGION', 'unknown')}")
        print(f"   Environment: {os.getenv('RAILWAY_ENVIRONMENT')}")
    else:
        print("ℹ️  Not running on Railway (local environment)")
    
    print("\n" + "="*60)
    print("For more help, check the Railway docs: https://docs.railway.app")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()