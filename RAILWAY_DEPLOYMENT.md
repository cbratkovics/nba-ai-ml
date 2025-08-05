# Railway Deployment Guide

## 🚨 Critical: DATABASE_URL Configuration

Your Railway deployment will fail without setting `DATABASE_URL`. The app will start but all database operations will fail.

## Quick Setup

1. **In Railway Dashboard:**
   - Go to your service → Variables tab
   - Add `DATABASE_URL` with your Supabase PostgreSQL connection string
   - Format: `postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`

2. **Find Your Supabase Connection String:**
   - Supabase Dashboard → Settings → Database
   - Connection string → URI (copy this)
   - Make sure it starts with `postgresql://` not `postgres://`

3. **Verify Deployment:**
   - Check logs for: "✅ All required environment variables are set"
   - Visit: `https://your-app.railway.app/health/detailed`
   - Database should show "connected"

## Environment Variables

### Required:
- `DATABASE_URL` - PostgreSQL connection string

### Optional but Recommended:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anon key
- `REDIS_URL` - Redis for caching (use Railway Redis add-on)
- `ENVIRONMENT` - Set to `production`

## Troubleshooting

Run the setup guide for detailed help:
```bash
python scripts/railway_setup_guide.py
```

### App Crashes on Start?
- Check Railway logs for "DATABASE_URL not found"
- Verify DATABASE_URL is set in Variables tab
- Ensure format is correct (postgresql:// not postgres://)

### Database Connection Failed?
- Check if password contains special characters (need URL encoding)
- Try adding `?sslmode=require` to the connection string
- Test connection string locally first

### Health Check Shows "Degraded"?
- Visit `/health/detailed` to see which service is failing
- Database "not configured" = missing DATABASE_URL
- Database "error" = invalid connection string

## Monitoring

The app includes comprehensive monitoring. Check:
- `/health/status` - Basic health check
- `/health/detailed` - Detailed service status
- `/dashboard/metrics` - Performance metrics (if monitoring enabled)

## Support

The app is designed to start even without DATABASE_URL to allow health checks to work. This helps with debugging. Once DATABASE_URL is set, full functionality will be available.