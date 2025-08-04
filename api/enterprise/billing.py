"""
Usage-based Billing System for NBA ML Platform
Integrates with Stripe for payment processing and usage tracking
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum
import stripe
import logging
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from pydantic import BaseModel, EmailStr
from fastapi import HTTPException, BackgroundTasks
import redis

Base = declarative_base()
logger = logging.getLogger(__name__)

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_your_key")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_your_secret")


class BillingPlan(Enum):
    """Billing plan types"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    USAGE_BASED = "usage_based"
    CUSTOM = "custom"


class PricingModel(Enum):
    """Pricing model types"""
    FLAT_RATE = "flat_rate"
    PER_USER = "per_user"
    USAGE_BASED = "usage_based"
    TIERED = "tiered"
    VOLUME = "volume"


# Pricing configuration
PRICING_CONFIG = {
    BillingPlan.FREE: {
        "name": "Free",
        "model": PricingModel.FLAT_RATE,
        "base_price": 0,
        "included": {
            "predictions": 1000,
            "api_calls": 1000,
            "storage_gb": 1,
            "users": 1
        },
        "overage_rates": {}  # No overages on free plan
    },
    BillingPlan.STARTER: {
        "name": "Starter",
        "model": PricingModel.FLAT_RATE,
        "base_price": 49.00,
        "included": {
            "predictions": 10000,
            "api_calls": 10000,
            "storage_gb": 10,
            "users": 5
        },
        "overage_rates": {
            "predictions": 0.005,  # $0.005 per prediction
            "api_calls": 0.001,    # $0.001 per API call
            "storage_gb": 0.10,    # $0.10 per GB
            "users": 10.00         # $10 per additional user
        }
    },
    BillingPlan.PROFESSIONAL: {
        "name": "Professional",
        "model": PricingModel.FLAT_RATE,
        "base_price": 299.00,
        "included": {
            "predictions": 100000,
            "api_calls": 100000,
            "storage_gb": 100,
            "users": 20
        },
        "overage_rates": {
            "predictions": 0.003,
            "api_calls": 0.0005,
            "storage_gb": 0.08,
            "users": 8.00
        }
    },
    BillingPlan.ENTERPRISE: {
        "name": "Enterprise",
        "model": PricingModel.FLAT_RATE,
        "base_price": 999.00,
        "included": {
            "predictions": 1000000,
            "api_calls": 1000000,
            "storage_gb": 1000,
            "users": 100
        },
        "overage_rates": {
            "predictions": 0.001,
            "api_calls": 0.0001,
            "storage_gb": 0.05,
            "users": 5.00
        }
    },
    BillingPlan.USAGE_BASED: {
        "name": "Pay As You Go",
        "model": PricingModel.USAGE_BASED,
        "base_price": 0,
        "included": {},
        "usage_rates": {
            "predictions": {
                "tiers": [
                    {"up_to": 1000, "rate": 0.01},
                    {"up_to": 10000, "rate": 0.008},
                    {"up_to": 100000, "rate": 0.005},
                    {"up_to": None, "rate": 0.003}
                ]
            },
            "api_calls": {
                "rate": 0.001
            },
            "storage_gb": {
                "rate": 0.10
            }
        }
    }
}


# Database Models
class Subscription(Base):
    """Subscription model"""
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Stripe IDs
    stripe_subscription_id = Column(String(255), unique=True)
    stripe_customer_id = Column(String(255))
    stripe_price_id = Column(String(255))
    stripe_payment_method_id = Column(String(255))
    
    # Plan details
    plan = Column(String(50), nullable=False)
    pricing_model = Column(String(50), nullable=False)
    base_price = Column(Float, default=0)
    
    # Billing cycle
    billing_period = Column(String(20), default="monthly")  # monthly, yearly
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    
    # Status
    status = Column(String(50), default="active")  # active, canceled, past_due, trialing
    trial_end = Column(DateTime)
    cancel_at_period_end = Column(Boolean, default=False)
    canceled_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    invoices = relationship("Invoice", back_populates="subscription")
    usage_records = relationship("BillingUsageRecord", back_populates="subscription")


class Invoice(Base):
    """Invoice model"""
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Stripe IDs
    stripe_invoice_id = Column(String(255), unique=True)
    stripe_payment_intent_id = Column(String(255))
    
    # Invoice details
    invoice_number = Column(String(50), unique=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Amounts (in cents)
    subtotal = Column(Integer, default=0)
    tax = Column(Integer, default=0)
    total = Column(Integer, default=0)
    amount_paid = Column(Integer, default=0)
    amount_due = Column(Integer, default=0)
    
    # Line items
    line_items = Column(JSON, default=[])
    
    # Status
    status = Column(String(50), default="draft")  # draft, open, paid, void, uncollectible
    paid_at = Column(DateTime)
    due_date = Column(DateTime)
    
    # PDF
    invoice_pdf_url = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="invoices")


class BillingUsageRecord(Base):
    """Detailed usage records for billing"""
    __tablename__ = "billing_usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Usage details
    metric = Column(String(50), nullable=False)  # predictions, api_calls, storage, etc.
    quantity = Column(Float, nullable=False)
    unit_price = Column(Float)  # Price per unit
    total_price = Column(Float)  # Total for this usage
    
    # Period
    timestamp = Column(DateTime, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Status
    billed = Column(Boolean, default=False)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="usage_records")


class PaymentMethod(Base):
    """Payment method model"""
    __tablename__ = "payment_methods"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Stripe IDs
    stripe_payment_method_id = Column(String(255), unique=True)
    
    # Card details (last 4 digits only)
    type = Column(String(50))  # card, bank_account
    brand = Column(String(50))  # visa, mastercard, etc.
    last4 = Column(String(4))
    exp_month = Column(Integer)
    exp_year = Column(Integer)
    
    # Status
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic Models
class CreateSubscriptionRequest(BaseModel):
    """Request model for creating subscription"""
    plan: BillingPlan
    payment_method_id: str
    billing_period: str = "monthly"
    trial_days: Optional[int] = None


class UpdateSubscriptionRequest(BaseModel):
    """Request model for updating subscription"""
    plan: Optional[BillingPlan] = None
    billing_period: Optional[str] = None
    payment_method_id: Optional[str] = None


class UsageReportRequest(BaseModel):
    """Request model for reporting usage"""
    metric: str
    quantity: float
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict] = {}


# Billing Manager
class BillingManager:
    """Manages billing operations"""
    
    def __init__(self, db: Session, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
    
    async def create_subscription(
        self,
        organization_id: str,
        request: CreateSubscriptionRequest
    ) -> Subscription:
        """Create new subscription"""
        
        # Get organization
        from api.enterprise.multi_tenancy import Organization
        org = self.db.query(Organization).filter_by(id=organization_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Create or get Stripe customer
        if not org.stripe_customer_id:
            customer = stripe.Customer.create(
                email=org.email,
                name=org.name,
                metadata={"organization_id": str(org.id)}
            )
            org.stripe_customer_id = customer.id
            self.db.commit()
        
        # Attach payment method
        stripe.PaymentMethod.attach(
            request.payment_method_id,
            customer=org.stripe_customer_id
        )
        
        # Set as default payment method
        stripe.Customer.modify(
            org.stripe_customer_id,
            invoice_settings={"default_payment_method": request.payment_method_id}
        )
        
        # Get pricing
        pricing = PRICING_CONFIG[request.plan]
        
        # Create Stripe subscription
        stripe_sub = stripe.Subscription.create(
            customer=org.stripe_customer_id,
            items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": pricing["name"],
                        "metadata": {"plan": request.plan.value}
                    },
                    "unit_amount": int(pricing["base_price"] * 100),  # Convert to cents
                    "recurring": {"interval": request.billing_period}
                }
            }],
            trial_period_days=request.trial_days,
            metadata={
                "organization_id": str(org.id),
                "plan": request.plan.value
            }
        )
        
        # Create subscription record
        subscription = Subscription(
            organization_id=org.id,
            stripe_subscription_id=stripe_sub.id,
            stripe_customer_id=org.stripe_customer_id,
            plan=request.plan.value,
            pricing_model=pricing["model"].value,
            base_price=pricing["base_price"],
            billing_period=request.billing_period,
            current_period_start=datetime.fromtimestamp(stripe_sub.current_period_start),
            current_period_end=datetime.fromtimestamp(stripe_sub.current_period_end),
            status=stripe_sub.status,
            trial_end=datetime.fromtimestamp(stripe_sub.trial_end) if stripe_sub.trial_end else None
        )
        
        self.db.add(subscription)
        self.db.commit()
        self.db.refresh(subscription)
        
        # Update organization tier
        org.tier = request.plan.value
        org.stripe_subscription_id = stripe_sub.id
        self.db.commit()
        
        logger.info(f"Created subscription for organization {org.slug}: {request.plan.value}")
        
        return subscription
    
    async def update_subscription(
        self,
        subscription_id: str,
        request: UpdateSubscriptionRequest
    ) -> Subscription:
        """Update existing subscription"""
        
        subscription = self.db.query(Subscription).filter_by(id=subscription_id).first()
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        # Update Stripe subscription if plan changed
        if request.plan and request.plan.value != subscription.plan:
            pricing = PRICING_CONFIG[request.plan]
            
            stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
            stripe.Subscription.modify(
                subscription.stripe_subscription_id,
                items=[{
                    "id": stripe_sub["items"]["data"][0].id,
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": pricing["name"],
                            "metadata": {"plan": request.plan.value}
                        },
                        "unit_amount": int(pricing["base_price"] * 100),
                        "recurring": {"interval": subscription.billing_period}
                    }
                }]
            )
            
            subscription.plan = request.plan.value
            subscription.base_price = pricing["base_price"]
        
        # Update payment method if provided
        if request.payment_method_id:
            stripe.PaymentMethod.attach(
                request.payment_method_id,
                customer=subscription.stripe_customer_id
            )
            stripe.Customer.modify(
                subscription.stripe_customer_id,
                invoice_settings={"default_payment_method": request.payment_method_id}
            )
            subscription.stripe_payment_method_id = request.payment_method_id
        
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediately: bool = False
    ) -> Subscription:
        """Cancel subscription"""
        
        subscription = self.db.query(Subscription).filter_by(id=subscription_id).first()
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        # Cancel in Stripe
        if immediately:
            stripe.Subscription.delete(subscription.stripe_subscription_id)
            subscription.status = "canceled"
            subscription.canceled_at = datetime.utcnow()
        else:
            stripe.Subscription.modify(
                subscription.stripe_subscription_id,
                cancel_at_period_end=True
            )
            subscription.cancel_at_period_end = True
        
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
    
    async def record_usage(
        self,
        organization_id: str,
        request: UsageReportRequest
    ):
        """Record usage for billing"""
        
        # Get active subscription
        subscription = self.db.query(Subscription).filter_by(
            organization_id=organization_id,
            status="active"
        ).first()
        
        if not subscription:
            logger.warning(f"No active subscription for organization {organization_id}")
            return
        
        # Get pricing
        pricing = PRICING_CONFIG.get(BillingPlan(subscription.plan), {})
        
        # Calculate price based on plan
        unit_price = 0
        total_price = 0
        
        if subscription.pricing_model == PricingModel.USAGE_BASED.value:
            # Tiered pricing
            if request.metric in pricing.get("usage_rates", {}):
                rates = pricing["usage_rates"][request.metric]
                if "tiers" in rates:
                    remaining = request.quantity
                    for tier in rates["tiers"]:
                        if remaining <= 0:
                            break
                        tier_quantity = min(remaining, (tier["up_to"] or float('inf')) - (request.quantity - remaining))
                        total_price += tier_quantity * tier["rate"]
                        remaining -= tier_quantity
                    unit_price = total_price / request.quantity if request.quantity > 0 else 0
                else:
                    unit_price = rates.get("rate", 0)
                    total_price = request.quantity * unit_price
        else:
            # Check if over included amount
            included = pricing.get("included", {}).get(request.metric, 0)
            
            # Get current period usage
            period_usage = self.db.query(BillingUsageRecord).filter(
                BillingUsageRecord.subscription_id == subscription.id,
                BillingUsageRecord.metric == request.metric,
                BillingUsageRecord.period_start >= subscription.current_period_start,
                BillingUsageRecord.period_end <= subscription.current_period_end
            ).all()
            
            current_total = sum(u.quantity for u in period_usage)
            
            # Calculate overage
            if current_total + request.quantity > included:
                overage = max(0, current_total + request.quantity - included)
                unit_price = pricing.get("overage_rates", {}).get(request.metric, 0)
                total_price = overage * unit_price
        
        # Create usage record
        usage_record = BillingUsageRecord(
            subscription_id=subscription.id,
            organization_id=organization_id,
            metric=request.metric,
            quantity=request.quantity,
            unit_price=unit_price,
            total_price=total_price,
            timestamp=request.timestamp or datetime.utcnow(),
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
            metadata=request.metadata
        )
        
        self.db.add(usage_record)
        self.db.commit()
        
        # Update Redis cache for real-time tracking
        cache_key = f"usage:{organization_id}:{request.metric}:{subscription.current_period_start.isoformat()}"
        self.redis.incrbyfloat(cache_key, request.quantity)
        self.redis.expire(cache_key, 86400)  # 24 hour expiry
        
        # Report to Stripe for usage-based billing
        if subscription.pricing_model == PricingModel.USAGE_BASED.value:
            stripe.SubscriptionItem.create_usage_record(
                subscription_item=subscription.stripe_subscription_id,
                quantity=int(request.quantity),
                timestamp=int(request.timestamp.timestamp() if request.timestamp else datetime.utcnow().timestamp()),
                action="increment"
            )
    
    async def generate_invoice(self, subscription_id: str) -> Invoice:
        """Generate invoice for subscription"""
        
        subscription = self.db.query(Subscription).filter_by(id=subscription_id).first()
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        # Get unbilled usage records
        usage_records = self.db.query(BillingUsageRecord).filter(
            BillingUsageRecord.subscription_id == subscription.id,
            BillingUsageRecord.billed == False,
            BillingUsageRecord.period_start >= subscription.current_period_start,
            BillingUsageRecord.period_end <= subscription.current_period_end
        ).all()
        
        # Calculate line items
        line_items = []
        subtotal = int(subscription.base_price * 100)  # Base price in cents
        
        # Add base subscription
        line_items.append({
            "description": f"{subscription.plan} Plan - {subscription.billing_period}",
            "quantity": 1,
            "unit_price": subscription.base_price,
            "total": subscription.base_price
        })
        
        # Add usage charges
        usage_by_metric = {}
        for record in usage_records:
            if record.metric not in usage_by_metric:
                usage_by_metric[record.metric] = {
                    "quantity": 0,
                    "total_price": 0
                }
            usage_by_metric[record.metric]["quantity"] += record.quantity
            usage_by_metric[record.metric]["total_price"] += record.total_price
        
        for metric, data in usage_by_metric.items():
            if data["total_price"] > 0:
                line_items.append({
                    "description": f"{metric.title()} Usage",
                    "quantity": data["quantity"],
                    "unit_price": data["total_price"] / data["quantity"] if data["quantity"] > 0 else 0,
                    "total": data["total_price"]
                })
                subtotal += int(data["total_price"] * 100)
        
        # Create invoice
        invoice = Invoice(
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
            invoice_number=f"INV-{datetime.utcnow().strftime('%Y%m')}-{str(uuid.uuid4())[:8].upper()}",
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
            subtotal=subtotal,
            tax=0,  # Tax calculation would go here
            total=subtotal,
            amount_due=subtotal,
            line_items=line_items,
            status="draft",
            due_date=subscription.current_period_end + timedelta(days=7)
        )
        
        self.db.add(invoice)
        
        # Mark usage records as billed
        for record in usage_records:
            record.billed = True
            record.invoice_id = invoice.id
        
        self.db.commit()
        self.db.refresh(invoice)
        
        return invoice
    
    async def get_billing_summary(self, organization_id: str) -> Dict[str, Any]:
        """Get billing summary for organization"""
        
        # Get subscription
        subscription = self.db.query(Subscription).filter_by(
            organization_id=organization_id
        ).order_by(Subscription.created_at.desc()).first()
        
        if not subscription:
            return {
                "has_subscription": False,
                "plan": "free",
                "status": "no_subscription"
            }
        
        # Get current period usage
        usage_records = self.db.query(BillingUsageRecord).filter(
            BillingUsageRecord.subscription_id == subscription.id,
            BillingUsageRecord.period_start >= subscription.current_period_start,
            BillingUsageRecord.period_end <= subscription.current_period_end
        ).all()
        
        # Calculate usage by metric
        usage_summary = {}
        total_usage_cost = 0
        
        for record in usage_records:
            if record.metric not in usage_summary:
                usage_summary[record.metric] = {
                    "quantity": 0,
                    "cost": 0
                }
            usage_summary[record.metric]["quantity"] += record.quantity
            usage_summary[record.metric]["cost"] += record.total_price
            total_usage_cost += record.total_price
        
        # Get recent invoices
        recent_invoices = self.db.query(Invoice).filter_by(
            subscription_id=subscription.id
        ).order_by(Invoice.created_at.desc()).limit(5).all()
        
        # Get payment methods
        payment_methods = self.db.query(PaymentMethod).filter_by(
            organization_id=organization_id,
            is_active=True
        ).all()
        
        return {
            "has_subscription": True,
            "subscription": {
                "id": str(subscription.id),
                "plan": subscription.plan,
                "status": subscription.status,
                "base_price": subscription.base_price,
                "billing_period": subscription.billing_period,
                "current_period_start": subscription.current_period_start.isoformat(),
                "current_period_end": subscription.current_period_end.isoformat(),
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "trial_end": subscription.trial_end.isoformat() if subscription.trial_end else None
            },
            "current_usage": usage_summary,
            "estimated_total": subscription.base_price + total_usage_cost,
            "recent_invoices": [
                {
                    "id": str(inv.id),
                    "invoice_number": inv.invoice_number,
                    "total": inv.total / 100,  # Convert from cents
                    "status": inv.status,
                    "due_date": inv.due_date.isoformat() if inv.due_date else None,
                    "paid_at": inv.paid_at.isoformat() if inv.paid_at else None
                }
                for inv in recent_invoices
            ],
            "payment_methods": [
                {
                    "id": str(pm.id),
                    "type": pm.type,
                    "brand": pm.brand,
                    "last4": pm.last4,
                    "exp_month": pm.exp_month,
                    "exp_year": pm.exp_year,
                    "is_default": pm.is_default
                }
                for pm in payment_methods
            ]
        }
    
    async def handle_stripe_webhook(self, payload: str, signature: str) -> Dict:
        """Handle Stripe webhook events"""
        
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Handle different event types
        if event["type"] == "invoice.payment_succeeded":
            invoice = event["data"]["object"]
            await self._handle_payment_succeeded(invoice)
            
        elif event["type"] == "invoice.payment_failed":
            invoice = event["data"]["object"]
            await self._handle_payment_failed(invoice)
            
        elif event["type"] == "customer.subscription.updated":
            subscription = event["data"]["object"]
            await self._handle_subscription_updated(subscription)
            
        elif event["type"] == "customer.subscription.deleted":
            subscription = event["data"]["object"]
            await self._handle_subscription_deleted(subscription)
        
        return {"status": "success"}
    
    async def _handle_payment_succeeded(self, stripe_invoice):
        """Handle successful payment"""
        
        invoice = self.db.query(Invoice).filter_by(
            stripe_invoice_id=stripe_invoice["id"]
        ).first()
        
        if invoice:
            invoice.status = "paid"
            invoice.paid_at = datetime.fromtimestamp(stripe_invoice["status_transitions"]["paid_at"])
            invoice.amount_paid = stripe_invoice["amount_paid"]
            invoice.amount_due = 0
            self.db.commit()
            
            logger.info(f"Invoice {invoice.invoice_number} marked as paid")
    
    async def _handle_payment_failed(self, stripe_invoice):
        """Handle failed payment"""
        
        invoice = self.db.query(Invoice).filter_by(
            stripe_invoice_id=stripe_invoice["id"]
        ).first()
        
        if invoice:
            invoice.status = "past_due"
            self.db.commit()
            
            # Send notification
            logger.warning(f"Payment failed for invoice {invoice.invoice_number}")
    
    async def _handle_subscription_updated(self, stripe_subscription):
        """Handle subscription update"""
        
        subscription = self.db.query(Subscription).filter_by(
            stripe_subscription_id=stripe_subscription["id"]
        ).first()
        
        if subscription:
            subscription.status = stripe_subscription["status"]
            subscription.current_period_start = datetime.fromtimestamp(stripe_subscription["current_period_start"])
            subscription.current_period_end = datetime.fromtimestamp(stripe_subscription["current_period_end"])
            self.db.commit()
    
    async def _handle_subscription_deleted(self, stripe_subscription):
        """Handle subscription deletion"""
        
        subscription = self.db.query(Subscription).filter_by(
            stripe_subscription_id=stripe_subscription["id"]
        ).first()
        
        if subscription:
            subscription.status = "canceled"
            subscription.canceled_at = datetime.utcnow()
            self.db.commit()
            
            # Update organization tier
            from api.enterprise.multi_tenancy import Organization
            org = self.db.query(Organization).filter_by(id=subscription.organization_id).first()
            if org:
                org.tier = "free"
                self.db.commit()