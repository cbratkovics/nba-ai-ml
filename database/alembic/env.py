"""
Alembic environment configuration
Handles database migrations with support for partitioning
"""
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

# Import models
from database.models import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate
target_metadata = Base.metadata


def get_database_url():
    """Get database URL from environment or config"""
    # Priority: Environment variable > alembic.ini
    url = os.getenv('DATABASE_URL')
    if url:
        # Handle Heroku-style postgres:// URLs
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        return url
    
    # Fallback to alembic.ini
    return config.get_main_option("sqlalchemy.url")


def include_object(object, name, type_, reflected, compare_to):
    """
    Filter objects for migration generation
    Skip partitioned table partitions
    """
    # Skip auto-generated constraint names
    if type_ == "index" and name and name.startswith("ix_"):
        return False
    
    # Skip partition tables
    if type_ == "table" and name and "_partition_" in name:
        return False
    
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
            # PostgreSQL-specific options
            transaction_per_migration=True,
            include_schemas=True,
        )

        with context.begin_transaction():
            # Create custom SQL for partitioning if needed
            connection.execute("SET lock_timeout = '10s'")
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()