#!/usr/bin/env python3
"""
Database CLI for NBA ML system
Commands for database initialization, migration, and maintenance
"""
import os
import sys
import click
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alembic import command
from alembic.config import Config
from database.connection import DatabaseManager, init_db
from database.models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """NBA ML Database Management CLI"""
    pass


@cli.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
def init(url):
    """Initialize database with tables and partitions"""
    if not url:
        click.echo("Error: DATABASE_URL not provided", err=True)
        sys.exit(1)
    
    click.echo(f"Initializing database...")
    
    try:
        # Initialize database manager
        db_manager = init_db(url)
        
        # Create tables
        db_manager.create_tables()
        
        # Create additional partitions
        create_future_partitions(db_manager)
        
        click.echo("✓ Database initialized successfully")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--message', '-m', required=True, help='Migration message')
def migrate(message):
    """Create a new migration"""
    alembic_cfg = Config("alembic.ini")
    
    try:
        command.revision(alembic_cfg, message=message, autogenerate=True)
        click.echo(f"✓ Migration created: {message}")
    except Exception as e:
        click.echo(f"Error creating migration: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def upgrade():
    """Apply pending migrations"""
    alembic_cfg = Config("alembic.ini")
    
    try:
        click.echo("Applying migrations...")
        command.upgrade(alembic_cfg, "head")
        click.echo("✓ Migrations applied successfully")
    except Exception as e:
        click.echo(f"Error applying migrations: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--steps', default=1, help='Number of migrations to rollback')
def rollback(steps):
    """Rollback migrations"""
    alembic_cfg = Config("alembic.ini")
    
    try:
        click.echo(f"Rolling back {steps} migration(s)...")
        command.downgrade(alembic_cfg, f"-{steps}")
        click.echo("✓ Rollback completed")
    except Exception as e:
        click.echo(f"Error during rollback: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show migration status"""
    alembic_cfg = Config("alembic.ini")
    
    try:
        command.current(alembic_cfg, verbose=True)
        click.echo("\nPending migrations:")
        command.heads(alembic_cfg, verbose=True)
    except Exception as e:
        click.echo(f"Error checking status: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
def seed(url):
    """Seed database with initial data"""
    if not url:
        click.echo("Error: DATABASE_URL not provided", err=True)
        sys.exit(1)
    
    click.echo("Seeding database...")
    
    try:
        db_manager = init_db(url)
        
        with db_manager.transaction() as session:
            # Seed data sources
            data_sources = [
                DataSource(
                    name="nba_api",
                    source_type="api",
                    base_url="https://stats.nba.com",
                    requires_auth=False,
                    rate_limit_requests=300,
                    rate_limit_window_seconds=60,
                    cost_per_request=0,
                    monthly_cost=0,
                    is_active=True,
                    health_status="healthy"
                ),
                DataSource(
                    name="api_nba",
                    source_type="api",
                    base_url="https://api-nba-v1.p.rapidapi.com",
                    requires_auth=True,
                    rate_limit_requests=None,  # Unlimited on paid plan
                    rate_limit_window_seconds=None,
                    cost_per_request=0,
                    monthly_cost=39.00,
                    is_active=True,
                    health_status="healthy"
                ),
                DataSource(
                    name="balldontlie",
                    source_type="api",
                    base_url="https://www.balldontlie.io/api/v1",
                    requires_auth=False,
                    rate_limit_requests=60,
                    rate_limit_window_seconds=60,
                    cost_per_request=0,
                    monthly_cost=0,
                    is_active=True,
                    health_status="healthy"
                )
            ]
            
            for ds in data_sources:
                existing = session.query(DataSource).filter_by(name=ds.name).first()
                if not existing:
                    session.add(ds)
                    click.echo(f"  Added data source: {ds.name}")
            
            # Seed sample teams
            sample_teams = [
                {"nba_team_id": "1610612738", "abbreviation": "BOS", "name": "Celtics", "full_name": "Boston Celtics", "conference": "East"},
                {"nba_team_id": "1610612747", "abbreviation": "LAL", "name": "Lakers", "full_name": "Los Angeles Lakers", "conference": "West"},
                {"nba_team_id": "1610612744", "abbreviation": "GSW", "name": "Warriors", "full_name": "Golden State Warriors", "conference": "West"},
                {"nba_team_id": "1610612743", "abbreviation": "DEN", "name": "Nuggets", "full_name": "Denver Nuggets", "conference": "West"},
            ]
            
            for team_data in sample_teams:
                existing = session.query(Team).filter_by(nba_team_id=team_data["nba_team_id"]).first()
                if not existing:
                    team = Team(**team_data)
                    session.add(team)
                    click.echo(f"  Added team: {team_data['full_name']}")
            
            # Seed sample players
            sample_players = [
                {"nba_player_id": "203999", "first_name": "Nikola", "last_name": "Jokic", "full_name": "Nikola Jokic", "position": "C"},
                {"nba_player_id": "2544", "first_name": "LeBron", "last_name": "James", "full_name": "LeBron James", "position": "F"},
                {"nba_player_id": "201939", "first_name": "Stephen", "last_name": "Curry", "full_name": "Stephen Curry", "position": "G"},
                {"nba_player_id": "1628369", "first_name": "Jayson", "last_name": "Tatum", "full_name": "Jayson Tatum", "position": "F"},
            ]
            
            for player_data in sample_players:
                existing = session.query(Player).filter_by(nba_player_id=player_data["nba_player_id"]).first()
                if not existing:
                    player = Player(**player_data)
                    session.add(player)
                    click.echo(f"  Added player: {player_data['full_name']}")
        
        click.echo("✓ Database seeded successfully")
        
    except Exception as e:
        click.echo(f"Error seeding database: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
def validate(url):
    """Validate database schema and data integrity"""
    if not url:
        click.echo("Error: DATABASE_URL not provided", err=True)
        sys.exit(1)
    
    click.echo("Validating database...")
    
    try:
        db_manager = init_db(url)
        
        # Check health
        health = db_manager.health_check()
        click.echo(f"Database health: {health}")
        
        # Check table counts
        with db_manager.get_db(read_only=True) as session:
            checks = {
                "Players": session.query(Player).count(),
                "Teams": session.query(Team).count(),
                "Games": session.query(Game).count(),
                "Models": session.query(Model).count(),
                "API Keys": session.query(APIKey).count(),
                "Data Sources": session.query(DataSource).count(),
            }
            
            click.echo("\nTable counts:")
            for table, count in checks.items():
                click.echo(f"  {table}: {count}")
        
        # Check partitions
        check_partitions(db_manager)
        
        click.echo("\n✓ Database validation completed")
        
    except Exception as e:
        click.echo(f"Error validating database: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
@click.option('--months', default=3, help='Number of months to create partitions for')
def create_partitions(url, months):
    """Create future partitions for partitioned tables"""
    if not url:
        click.echo("Error: DATABASE_URL not provided", err=True)
        sys.exit(1)
    
    click.echo(f"Creating partitions for next {months} months...")
    
    try:
        db_manager = init_db(url)
        create_future_partitions(db_manager, months)
        click.echo("✓ Partitions created successfully")
        
    except Exception as e:
        click.echo(f"Error creating partitions: {str(e)}", err=True)
        sys.exit(1)


def create_future_partitions(db_manager: DatabaseManager, months: int = 3):
    """Helper to create future partitions"""
    partitioned_tables = ['player_game_logs', 'player_features', 'predictions']
    current_date = datetime.now()
    
    with db_manager.engine.connect() as conn:
        for table_name in partitioned_tables:
            for i in range(months):
                start_date = current_date.replace(day=1) + timedelta(days=30*i)
                end_date = start_date + timedelta(days=31)
                partition_name = f"{table_name}_{start_date.strftime('%Y_%m')}"
                
                sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name} 
                PARTITION OF {table_name}
                FOR VALUES FROM ('{start_date.strftime('%Y-%m-01')}') 
                TO ('{end_date.strftime('%Y-%m-01')}')
                """
                
                try:
                    conn.execute(sql)
                    logger.info(f"Created partition: {partition_name}")
                except Exception as e:
                    if "already exists" not in str(e):
                        logger.error(f"Error creating partition {partition_name}: {e}")


def check_partitions(db_manager: DatabaseManager):
    """Check existing partitions"""
    sql = """
    SELECT 
        parent.relname AS parent_table,
        child.relname AS partition_name
    FROM pg_inherits
    JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
    JOIN pg_class child ON pg_inherits.inhrelid = child.oid
    WHERE parent.relname IN ('player_game_logs', 'player_features', 'predictions')
    ORDER BY parent.relname, child.relname;
    """
    
    results = db_manager.execute_sql(sql, read_only=True)
    
    click.echo("\nPartitions:")
    current_table = None
    for row in results:
        if row[0] != current_table:
            current_table = row[0]
            click.echo(f"\n  {current_table}:")
        click.echo(f"    - {row[1]}")


@cli.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
@click.option('--older-than-days', default=90, help='Delete data older than N days')
@click.confirmation_option(prompt='Are you sure you want to clean old data?')
def clean(url, older_than_days):
    """Clean old data from database"""
    if not url:
        click.echo("Error: DATABASE_URL not provided", err=True)
        sys.exit(1)
    
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    click.echo(f"Cleaning data older than {cutoff_date.date()}...")
    
    try:
        db_manager = init_db(url)
        
        with db_manager.transaction() as session:
            # Clean audit logs
            deleted = session.query(AuditLog).filter(
                AuditLog.created_at < cutoff_date
            ).delete()
            click.echo(f"  Deleted {deleted} audit logs")
            
            # Clean old predictions
            sql = f"""
            DELETE FROM predictions 
            WHERE game_date < '{cutoff_date.date()}'
            """
            session.execute(sql)
            click.echo(f"  Cleaned old predictions")
        
        click.echo("✓ Database cleaned successfully")
        
    except Exception as e:
        click.echo(f"Error cleaning database: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
def health(url):
    """Check database health and connection pool status"""
    if not url:
        click.echo("Error: DATABASE_URL not provided", err=True)
        sys.exit(1)
    
    try:
        db_manager = init_db(url)
        health_status = db_manager.health_check()
        
        click.echo("Database Health Check:")
        click.echo(f"  Primary: {health_status['primary']}")
        click.echo(f"  Read Replica: {health_status.get('read_replica', 'Not configured')}")
        click.echo(f"\nConnection Pool:")
        click.echo(f"  Size: {health_status['pool_size']}")
        click.echo(f"  Checked out: {health_status['pool_checked_out']}")
        click.echo(f"  Overflow: {health_status['pool_overflow']}")
        click.echo(f"  Total: {health_status['pool_total']}")
        
    except Exception as e:
        click.echo(f"Error checking health: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()