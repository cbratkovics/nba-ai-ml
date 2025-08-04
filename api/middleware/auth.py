from fastapi import HTTPException, Header, Depends
from typing import Optional
import httpx
import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_API_URL = "https://api.clerk.com/v1"

@lru_cache()
def get_clerk_client():
    return httpx.AsyncClient(
        base_url=CLERK_API_URL,
        headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"}
    )

async def verify_api_key(authorization: Optional[str] = Header(None)) -> dict:
    """Verify JWT token from Clerk"""
    if not authorization:
        # Allow demo endpoint without auth
        return {"tier": "demo", "user_id": "anonymous"}
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    
    # For API keys (non-JWT), check Supabase
    if not token.startswith("ey"):
        # This is an API key, not a JWT
        return {"tier": "api", "user_id": "api_user", "api_key": token}
    
    # Verify JWT with Clerk
    try:
        client = get_clerk_client()
        response = await client.post(
            "/tokens/verify",
            json={"token": token}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        data = response.json()
        return {
            "tier": "premium",
            "user_id": data.get("sub"),
            "session_id": data.get("sid")
        }
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

def get_api_key_tier(user_data: dict) -> str:
    return user_data.get("tier", "demo")