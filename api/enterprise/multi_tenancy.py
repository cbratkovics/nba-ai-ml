"""
Multi-tenancy System for NBA ML Platform
Implements organization-based isolation and resource management
"""
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import redis
from fastapi import HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel, EmailStr, validator
import logging

Base = declarative_base()
logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class TenantTier(Enum):
    """Tenant subscription tiers"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Resource types for quota tracking"""
    API_CALLS = "api_calls"
    PREDICTIONS = "predictions"
    STORAGE = "storage"
    COMPUTE = "compute"
    USERS = "users"
    MODELS = "models"


@dataclass
class TierLimits:
    """Resource limits for each tier"""
    api_calls_per_month: int
    predictions_per_month: int
    storage_gb: float
    max_users: int
    max_models: int
    rate_limit_per_minute: int
    batch_size: int
    data_retention_days: int
    custom_models: bool
    priority_support: bool
    sla_uptime: float
    features: Set[str] = field(default_factory=set)


# Tier configurations
TIER_CONFIGS = {
    TenantTier.FREE: TierLimits(
        api_calls_per_month=1000,
        predictions_per_month=1000,
        storage_gb=1,
        max_users=1,
        max_models=0,
        rate_limit_per_minute=10,
        batch_size=10,
        data_retention_days=7,
        custom_models=False,
        priority_support=False,
        sla_uptime=0,
        features={"basic_predictions", "public_models"}
    ),
    TenantTier.STARTER: TierLimits(
        api_calls_per_month=10000,
        predictions_per_month=10000,
        storage_gb=10,
        max_users=5,
        max_models=3,
        rate_limit_per_minute=60,
        batch_size=50,
        data_retention_days=30,
        custom_models=False,
        priority_support=False,
        sla_uptime=99.0,
        features={"basic_predictions", "public_models", "api_access", "webhooks"}
    ),
    TenantTier.PROFESSIONAL: TierLimits(
        api_calls_per_month=100000,
        predictions_per_month=100000,
        storage_gb=100,
        max_users=20,
        max_models=10,
        rate_limit_per_minute=300,
        batch_size=200,
        data_retention_days=90,
        custom_models=True,
        priority_support=True,
        sla_uptime=99.9,
        features={"all_predictions", "custom_models", "api_access", "webhooks", 
                 "advanced_analytics", "export_data", "team_collaboration"}
    ),
    TenantTier.ENTERPRISE: TierLimits(
        api_calls_per_month=1000000,
        predictions_per_month=1000000,
        storage_gb=1000,
        max_users=100,
        max_models=50,
        rate_limit_per_minute=1000,
        batch_size=1000,
        data_retention_days=365,
        custom_models=True,
        priority_support=True,
        sla_uptime=99.99,
        features={"all_predictions", "custom_models", "api_access", "webhooks",
                 "advanced_analytics", "export_data", "team_collaboration",
                 "white_label", "dedicated_support", "custom_integration", "audit_logs"}
    ),
    TenantTier.CUSTOM: TierLimits(
        api_calls_per_month=-1,  # Unlimited
        predictions_per_month=-1,
        storage_gb=-1,
        max_users=-1,
        max_models=-1,
        rate_limit_per_minute=2000,
        batch_size=5000,
        data_retention_days=-1,
        custom_models=True,
        priority_support=True,
        sla_uptime=99.99,
        features=set()  # All features
    )
}


# Database Models
class Organization(Base):
    """Organization/Tenant model"""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=False)
    tier = Column(String(50), default=TenantTier.FREE.value)
    api_key = Column(String(255), unique=True, nullable=False)
    secret_key = Column(String(255), nullable=False)
    
    # Contact info
    email = Column(String(255), nullable=False)
    phone = Column(String(50))
    website = Column(String(255))
    
    # Billing
    stripe_customer_id = Column(String(255))
    stripe_subscription_id = Column(String(255))
    billing_email = Column(String(255))
    
    # Settings
    settings = Column(JSON, default={})
    features = Column(JSON, default=[])
    custom_limits = Column(JSON, default={})
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    trial_ends_at = Column(DateTime)
    suspended_at = Column(DateTime)
    suspension_reason = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="organization", cascade="all, delete-orphan")
    
    def get_limits(self) -> TierLimits:
        """Get organization resource limits"""
        base_limits = TIER_CONFIGS.get(TenantTier(self.tier), TIER_CONFIGS[TenantTier.FREE])
        
        # Apply custom limits if any
        if self.custom_limits:
            for key, value in self.custom_limits.items():
                if hasattr(base_limits, key):
                    setattr(base_limits, key, value)
        
        return base_limits
    
    def has_feature(self, feature: str) -> bool:
        """Check if organization has access to feature"""
        limits = self.get_limits()
        return feature in limits.features or feature in self.features


class User(Base):
    """User model with organization association"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    first_name = Column(String(255))
    last_name = Column(String(255))
    
    # Role and permissions
    role = Column(String(50), default="member")  # owner, admin, member, viewer
    permissions = Column(JSON, default=[])
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    
    def can(self, permission: str) -> bool:
        """Check if user has permission"""
        if self.role == "owner":
            return True
        if self.role == "admin" and permission != "delete_organization":
            return True
        return permission in self.permissions


class APIKey(Base):
    """API Key model for programmatic access"""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    key = Column(String(255), unique=True, nullable=False)
    secret = Column(String(255), nullable=False)
    
    # Scopes and permissions
    scopes = Column(JSON, default=["read:predictions"])
    rate_limit_override = Column(Integer)
    
    # Usage tracking
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="api_keys")


class UsageRecord(Base):
    """Usage tracking for billing and quotas"""
    __tablename__ = "usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    resource_type = Column(String(50), nullable=False)
    quantity = Column(Float, nullable=False)
    
    # Billing period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="usage_records")


# Pydantic Models
class OrganizationCreate(BaseModel):
    """Organization creation model"""
    name: str
    email: EmailStr
    phone: Optional[str] = None
    website: Optional[str] = None
    tier: TenantTier = TenantTier.FREE
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError('Organization name must be at least 3 characters')
        return v


class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    username: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str = "member"


class TenantContext(BaseModel):
    """Tenant context for request processing"""
    organization_id: str
    organization_slug: str
    tier: TenantTier
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    scopes: List[str] = []
    rate_limit: int = 60
    
    @property
    def is_authenticated(self) -> bool:
        return bool(self.user_id or self.api_key_id)


# Multi-tenancy Manager
class MultiTenancyManager:
    """Manages multi-tenant operations"""
    
    def __init__(self, db: Session, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
        self.security = HTTPBearer()
    
    async def create_organization(self, org_data: OrganizationCreate) -> Organization:
        """Create new organization"""
        # Generate unique slug
        slug = self._generate_slug(org_data.name)
        
        # Generate API credentials
        api_key = self._generate_api_key()
        secret_key = self._generate_secret_key()
        
        # Create organization
        org = Organization(
            name=org_data.name,
            slug=slug,
            email=org_data.email,
            phone=org_data.phone,
            website=org_data.website,
            tier=org_data.tier.value,
            api_key=api_key,
            secret_key=secret_key
        )
        
        # Set trial period for paid tiers
        if org_data.tier != TenantTier.FREE:
            org.trial_ends_at = datetime.utcnow() + timedelta(days=14)
        
        self.db.add(org)
        self.db.commit()
        self.db.refresh(org)
        
        logger.info(f"Created organization: {org.slug} (tier: {org.tier})")
        
        return org
    
    async def create_user(self, org_id: str, user_data: UserCreate) -> User:
        """Create user within organization"""
        # Check organization exists and has capacity
        org = self.db.query(Organization).filter_by(id=org_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        limits = org.get_limits()
        current_users = self.db.query(User).filter_by(organization_id=org_id).count()
        
        if limits.max_users != -1 and current_users >= limits.max_users:
            raise HTTPException(
                status_code=403,
                detail=f"User limit reached ({limits.max_users} users)"
            )
        
        # Hash password
        password_hash = self._hash_password(user_data.password)
        
        # Create user
        user = User(
            organization_id=org_id,
            email=user_data.email,
            username=user_data.username,
            password_hash=password_hash,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role
        )
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(f"Created user {user.username} for organization {org.slug}")
        
        return user
    
    async def authenticate(self, credentials: HTTPAuthorizationCredentials) -> TenantContext:
        """Authenticate request and return tenant context"""
        token = credentials.credentials
        
        try:
            # Decode JWT token
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Extract tenant context
            org_id = payload.get("org_id")
            user_id = payload.get("user_id")
            api_key_id = payload.get("api_key_id")
            
            if not org_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Get organization
            org = self.db.query(Organization).filter_by(id=org_id).first()
            if not org or not org.is_active:
                raise HTTPException(status_code=401, detail="Organization not found or inactive")
            
            # Check if suspended
            if org.suspended_at:
                raise HTTPException(
                    status_code=403,
                    detail=f"Organization suspended: {org.suspension_reason}"
                )
            
            # Get limits
            limits = org.get_limits()
            
            # Build context
            context = TenantContext(
                organization_id=str(org.id),
                organization_slug=org.slug,
                tier=TenantTier(org.tier),
                user_id=user_id,
                api_key_id=api_key_id,
                scopes=payload.get("scopes", []),
                rate_limit=limits.rate_limit_per_minute
            )
            
            return context
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def check_quota(self, org_id: str, resource: ResourceType, quantity: float = 1) -> bool:
        """Check if organization has quota for resource"""
        # Get organization
        org = self.db.query(Organization).filter_by(id=org_id).first()
        if not org:
            return False
        
        limits = org.get_limits()
        
        # Get current period (monthly)
        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, 1)
        period_end = (period_start + timedelta(days=32)).replace(day=1)
        
        # Get usage for current period
        usage = self.db.query(UsageRecord).filter(
            UsageRecord.organization_id == org_id,
            UsageRecord.resource_type == resource.value,
            UsageRecord.period_start >= period_start,
            UsageRecord.period_end <= period_end
        ).first()
        
        current_usage = usage.quantity if usage else 0
        
        # Check limits
        if resource == ResourceType.API_CALLS:
            limit = limits.api_calls_per_month
        elif resource == ResourceType.PREDICTIONS:
            limit = limits.predictions_per_month
        elif resource == ResourceType.STORAGE:
            limit = limits.storage_gb
        elif resource == ResourceType.USERS:
            limit = limits.max_users
        elif resource == ResourceType.MODELS:
            limit = limits.max_models
        else:
            return True
        
        # Unlimited (-1) or within limit
        return limit == -1 or (current_usage + quantity) <= limit
    
    async def track_usage(self, org_id: str, resource: ResourceType, quantity: float = 1):
        """Track resource usage for billing"""
        # Get current period
        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, 1)
        period_end = (period_start + timedelta(days=32)).replace(day=1)
        
        # Get or create usage record
        usage = self.db.query(UsageRecord).filter(
            UsageRecord.organization_id == org_id,
            UsageRecord.resource_type == resource.value,
            UsageRecord.period_start >= period_start,
            UsageRecord.period_end <= period_end
        ).first()
        
        if usage:
            usage.quantity += quantity
        else:
            usage = UsageRecord(
                organization_id=org_id,
                resource_type=resource.value,
                quantity=quantity,
                period_start=period_start,
                period_end=period_end
            )
            self.db.add(usage)
        
        self.db.commit()
        
        # Update Redis cache for fast lookups
        cache_key = f"usage:{org_id}:{resource.value}:{period_start.isoformat()}"
        self.redis.setex(cache_key, 3600, str(usage.quantity))
    
    async def get_organization_stats(self, org_id: str) -> Dict[str, Any]:
        """Get organization usage statistics"""
        org = self.db.query(Organization).filter_by(id=org_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        limits = org.get_limits()
        
        # Get current period usage
        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, 1)
        
        usage_records = self.db.query(UsageRecord).filter(
            UsageRecord.organization_id == org_id,
            UsageRecord.period_start >= period_start
        ).all()
        
        usage = {}
        for record in usage_records:
            usage[record.resource_type] = record.quantity
        
        # Get user count
        user_count = self.db.query(User).filter_by(organization_id=org_id).count()
        
        # Get API key count
        api_key_count = self.db.query(APIKey).filter_by(organization_id=org_id).count()
        
        return {
            "organization": {
                "id": str(org.id),
                "name": org.name,
                "slug": org.slug,
                "tier": org.tier,
                "created_at": org.created_at.isoformat()
            },
            "limits": {
                "api_calls": limits.api_calls_per_month,
                "predictions": limits.predictions_per_month,
                "storage_gb": limits.storage_gb,
                "max_users": limits.max_users,
                "max_models": limits.max_models,
                "rate_limit": limits.rate_limit_per_minute
            },
            "usage": {
                "api_calls": usage.get(ResourceType.API_CALLS.value, 0),
                "predictions": usage.get(ResourceType.PREDICTIONS.value, 0),
                "storage_gb": usage.get(ResourceType.STORAGE.value, 0),
                "users": user_count,
                "api_keys": api_key_count
            },
            "features": list(limits.features)
        }
    
    def _generate_slug(self, name: str) -> str:
        """Generate unique slug from name"""
        base_slug = name.lower().replace(" ", "-").replace(".", "")
        slug = base_slug
        counter = 1
        
        while self.db.query(Organization).filter_by(slug=slug).first():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug
    
    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"nba_{''.join(secrets.token_hex(16))}"
    
    def _generate_secret_key(self) -> str:
        """Generate secret key"""
        return secrets.token_urlsafe(32)
    
    def _hash_password(self, password: str) -> str:
        """Hash password"""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000).hex()
    
    def generate_token(self, org_id: str, user_id: Optional[str] = None, 
                      api_key_id: Optional[str] = None, scopes: List[str] = []) -> str:
        """Generate JWT token"""
        payload = {
            "org_id": str(org_id),
            "user_id": str(user_id) if user_id else None,
            "api_key_id": str(api_key_id) if api_key_id else None,
            "scopes": scopes,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# FastAPI Dependencies
async def get_tenant_context(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
    db: Session = Depends(get_database),
    redis_client: redis.Redis = Depends(get_redis)
) -> TenantContext:
    """Dependency to get tenant context from request"""
    manager = MultiTenancyManager(db, redis_client)
    return await manager.authenticate(credentials)


async def require_feature(feature: str):
    """Dependency to require specific feature"""
    async def check_feature(context: TenantContext = Depends(get_tenant_context),
                           db: Session = Depends(get_database)):
        org = db.query(Organization).filter_by(id=context.organization_id).first()
        if not org.has_feature(feature):
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature}' not available in {context.tier.value} tier"
            )
        return context
    return check_feature


async def check_quota(resource: ResourceType):
    """Dependency to check resource quota"""
    async def check(context: TenantContext = Depends(get_tenant_context),
                   db: Session = Depends(get_database),
                   redis_client: redis.Redis = Depends(get_redis)):
        manager = MultiTenancyManager(db, redis_client)
        if not await manager.check_quota(context.organization_id, resource):
            raise HTTPException(
                status_code=429,
                detail=f"Quota exceeded for {resource.value}"
            )
        return context
    return check