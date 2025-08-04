"""Initial database schema with partitioning

Revision ID: 001
Revises: 
Create Date: 2025-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def create_partitioned_table(table_name: str, partition_column: str, create_sql: str):
    """Helper to create partitioned tables with initial partitions"""
    # Create main partitioned table
    op.execute(create_sql)
    
    # Create partitions for current and next 3 months
    from datetime import datetime, timedelta
    current_date = datetime.now()
    
    for i in range(4):  # Current month + 3 future months
        start_date = current_date.replace(day=1) + timedelta(days=30*i)
        end_date = start_date + timedelta(days=31)
        
        partition_name = f"{table_name}_{start_date.strftime('%Y_%m')}"
        
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {partition_name} 
            PARTITION OF {table_name}
            FOR VALUES FROM ('{start_date.strftime('%Y-%m-01')}') 
            TO ('{end_date.strftime('%Y-%m-01')}')
        """)


def upgrade() -> None:
    # Create UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # ============ Core Tables ============
    
    # Players table
    op.create_table('players',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('nba_player_id', sa.String(20), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('full_name', sa.String(200), nullable=False),
        sa.Column('position', sa.String(10)),
        sa.Column('height_inches', sa.SmallInteger()),
        sa.Column('weight_pounds', sa.SmallInteger()),
        sa.Column('birth_date', sa.Date()),
        sa.Column('country', sa.String(100)),
        sa.Column('draft_year', sa.SmallInteger()),
        sa.Column('draft_round', sa.SmallInteger()),
        sa.Column('draft_number', sa.SmallInteger()),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('nba_player_id')
    )
    op.create_index('idx_player_full_name', 'players', ['full_name'])
    op.create_index('idx_player_active', 'players', ['is_active'])
    op.create_index('idx_player_nba_id', 'players', ['nba_player_id'])
    
    # Teams table
    op.create_table('teams',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('nba_team_id', sa.String(20), nullable=False),
        sa.Column('abbreviation', sa.String(5), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('full_name', sa.String(200), nullable=False),
        sa.Column('city', sa.String(100)),
        sa.Column('state', sa.String(50)),
        sa.Column('conference', sa.String(20)),
        sa.Column('division', sa.String(50)),
        sa.Column('venue', sa.String(200)),
        sa.Column('year_founded', sa.SmallInteger()),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('nba_team_id'),
        sa.UniqueConstraint('abbreviation')
    )
    op.create_index('idx_team_abbreviation', 'teams', ['abbreviation'])
    op.create_index('idx_team_conference_division', 'teams', ['conference', 'division'])
    
    # Games table
    op.create_table('games',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('nba_game_id', sa.String(30), nullable=False),
        sa.Column('game_date', sa.Date(), nullable=False),
        sa.Column('game_time', sa.TIMESTAMP(timezone=True)),
        sa.Column('season', sa.String(10), nullable=False),
        sa.Column('season_type', sa.String(20), default='Regular Season'),
        sa.Column('home_team_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('away_team_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('home_team_score', sa.SmallInteger()),
        sa.Column('away_team_score', sa.SmallInteger()),
        sa.Column('game_status', sa.String(20)),
        sa.Column('attendance', sa.Integer()),
        sa.Column('game_duration_minutes', sa.SmallInteger()),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['home_team_id'], ['teams.id']),
        sa.ForeignKeyConstraint(['away_team_id'], ['teams.id']),
        sa.UniqueConstraint('nba_game_id'),
        sa.CheckConstraint('home_team_id != away_team_id', name='different_teams')
    )
    op.create_index('idx_game_date', 'games', ['game_date'])
    op.create_index('idx_game_season', 'games', ['season'])
    op.create_index('idx_game_teams', 'games', ['home_team_id', 'away_team_id'])
    
    # Player Game Logs - PARTITIONED BY DATE
    create_partitioned_table(
        'player_game_logs',
        'game_date',
        """
        CREATE TABLE player_game_logs (
            id UUID DEFAULT gen_random_uuid(),
            player_id UUID NOT NULL,
            game_id UUID NOT NULL,
            team_id UUID NOT NULL,
            game_date DATE NOT NULL,
            minutes_played DECIMAL(5,2),
            started BOOLEAN DEFAULT FALSE,
            points SMALLINT DEFAULT 0,
            rebounds SMALLINT DEFAULT 0,
            offensive_rebounds SMALLINT DEFAULT 0,
            defensive_rebounds SMALLINT DEFAULT 0,
            assists SMALLINT DEFAULT 0,
            steals SMALLINT DEFAULT 0,
            blocks SMALLINT DEFAULT 0,
            turnovers SMALLINT DEFAULT 0,
            personal_fouls SMALLINT DEFAULT 0,
            field_goals_made SMALLINT DEFAULT 0,
            field_goals_attempted SMALLINT DEFAULT 0,
            field_goal_percentage DECIMAL(4,3),
            three_pointers_made SMALLINT DEFAULT 0,
            three_pointers_attempted SMALLINT DEFAULT 0,
            three_point_percentage DECIMAL(4,3),
            free_throws_made SMALLINT DEFAULT 0,
            free_throws_attempted SMALLINT DEFAULT 0,
            free_throw_percentage DECIMAL(4,3),
            plus_minus SMALLINT,
            true_shooting_percentage DECIMAL(4,3),
            effective_field_goal_percentage DECIMAL(4,3),
            usage_rate DECIMAL(4,3),
            game_score DECIMAL(5,2),
            home_game BOOLEAN,
            rest_days SMALLINT,
            back_to_back BOOLEAN DEFAULT FALSE,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (id, game_date),
            FOREIGN KEY (player_id) REFERENCES players(id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            FOREIGN KEY (team_id) REFERENCES teams(id),
            UNIQUE (player_id, game_id, game_date)
        ) PARTITION BY RANGE (game_date)
        """
    )
    
    # ============ Feature Store ============
    
    # Player Features - PARTITIONED BY DATE
    create_partitioned_table(
        'player_features',
        'game_date',
        """
        CREATE TABLE player_features (
            id UUID DEFAULT gen_random_uuid(),
            player_id UUID NOT NULL,
            game_id UUID NOT NULL,
            game_date DATE NOT NULL,
            feature_version VARCHAR(20) NOT NULL DEFAULT 'v1.0',
            points_ma5 DECIMAL(5,2),
            points_ma10 DECIMAL(5,2),
            points_ma20 DECIMAL(5,2),
            rebounds_ma5 DECIMAL(5,2),
            rebounds_ma10 DECIMAL(5,2),
            assists_ma5 DECIMAL(5,2),
            assists_ma10 DECIMAL(5,2),
            points_trend DECIMAL(5,2),
            rebounds_trend DECIMAL(5,2),
            assists_trend DECIMAL(5,2),
            hot_streak BOOLEAN,
            opponent_defensive_rating DECIMAL(5,2),
            matchup_difficulty DECIMAL(3,2),
            pace_factor DECIMAL(5,2),
            features JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            ttl TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id, game_date),
            FOREIGN KEY (player_id) REFERENCES players(id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            UNIQUE (player_id, game_id, feature_version, game_date)
        ) PARTITION BY RANGE (game_date)
        """
    )
    
    # ============ Model Registry ============
    
    op.create_table('models',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('target', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(100)),
        sa.Column('framework', sa.String(50)),
        sa.Column('artifact_uri', sa.Text(), nullable=False),
        sa.Column('model_size_mb', sa.DECIMAL(10, 2)),
        sa.Column('train_r2_score', sa.DECIMAL(5, 4)),
        sa.Column('val_r2_score', sa.DECIMAL(5, 4)),
        sa.Column('test_r2_score', sa.DECIMAL(5, 4)),
        sa.Column('train_mae', sa.DECIMAL(10, 4)),
        sa.Column('val_mae', sa.DECIMAL(10, 4)),
        sa.Column('test_mae', sa.DECIMAL(10, 4)),
        sa.Column('train_rmse', sa.DECIMAL(10, 4)),
        sa.Column('val_rmse', sa.DECIMAL(10, 4)),
        sa.Column('test_rmse', sa.DECIMAL(10, 4)),
        sa.Column('training_duration_seconds', sa.Integer()),
        sa.Column('training_samples', sa.Integer()),
        sa.Column('feature_count', sa.Integer()),
        sa.Column('hyperparameters', postgresql.JSONB()),
        sa.Column('feature_importance', postgresql.JSONB()),
        sa.Column('status', sa.String(20), default='training'),
        sa.Column('deployed_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('retired_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('created_by', sa.String(100)),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'version'),
        sa.CheckConstraint('val_r2_score >= 0 AND val_r2_score <= 1', name='valid_r2_score')
    )
    op.create_index('idx_model_status', 'models', ['status'])
    op.create_index('idx_model_target', 'models', ['target'])
    
    # ============ User Management ============
    
    op.create_table('api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('key_hash', sa.String(256), nullable=False),
        sa.Column('key_prefix', sa.String(20), nullable=False),
        sa.Column('user_email', sa.String(255), nullable=False),
        sa.Column('organization', sa.String(200)),
        sa.Column('tier', sa.String(20), nullable=False, default='free'),
        sa.Column('requests_per_hour', sa.Integer(), default=1000),
        sa.Column('requests_per_day', sa.Integer(), default=10000),
        sa.Column('total_requests', sa.BigInteger(), default=0),
        sa.Column('total_predictions', sa.BigInteger(), default=0),
        sa.Column('last_used_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash')
    )
    op.create_index('idx_api_key_hash', 'api_keys', ['key_hash'])
    op.create_index('idx_api_key_email', 'api_keys', ['user_email'])
    op.create_index('idx_api_key_tier', 'api_keys', ['tier'])
    op.create_index('idx_api_key_active', 'api_keys', ['is_active'])
    
    # Predictions - PARTITIONED BY DATE
    create_partitioned_table(
        'predictions',
        'game_date',
        """
        CREATE TABLE predictions (
            id UUID DEFAULT gen_random_uuid(),
            prediction_id VARCHAR(100) NOT NULL,
            player_id UUID NOT NULL,
            game_id UUID,
            model_id UUID NOT NULL,
            game_date DATE NOT NULL,
            predicted_points DECIMAL(5,2),
            predicted_rebounds DECIMAL(5,2),
            predicted_assists DECIMAL(5,2),
            predicted_steals DECIMAL(5,2),
            predicted_blocks DECIMAL(5,2),
            points_lower_bound DECIMAL(5,2),
            points_upper_bound DECIMAL(5,2),
            rebounds_lower_bound DECIMAL(5,2),
            rebounds_upper_bound DECIMAL(5,2),
            assists_lower_bound DECIMAL(5,2),
            assists_upper_bound DECIMAL(5,2),
            confidence_score DECIMAL(3,2),
            prediction_metadata JSONB DEFAULT '{}',
            api_key_id UUID,
            request_id VARCHAR(100),
            response_time_ms INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (id, game_date),
            FOREIGN KEY (player_id) REFERENCES players(id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            FOREIGN KEY (model_id) REFERENCES models(id),
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id),
            UNIQUE (prediction_id, game_date)
        ) PARTITION BY RANGE (game_date)
        """
    )
    
    # Prediction Feedback
    op.create_table('prediction_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('actual_points', sa.SmallInteger()),
        sa.Column('actual_rebounds', sa.SmallInteger()),
        sa.Column('actual_assists', sa.SmallInteger()),
        sa.Column('actual_steals', sa.SmallInteger()),
        sa.Column('actual_blocks', sa.SmallInteger()),
        sa.Column('points_error', sa.DECIMAL(5, 2)),
        sa.Column('rebounds_error', sa.DECIMAL(5, 2)),
        sa.Column('assists_error', sa.DECIMAL(5, 2)),
        sa.Column('points_absolute_error', sa.DECIMAL(5, 2)),
        sa.Column('rebounds_absolute_error', sa.DECIMAL(5, 2)),
        sa.Column('assists_absolute_error', sa.DECIMAL(5, 2)),
        sa.Column('within_confidence_interval', sa.Boolean()),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        # Note: Foreign key to predictions table can't reference partitioned table directly
        sa.UniqueConstraint('prediction_id')
    )
    op.create_index('idx_feedback_created', 'prediction_feedback', ['created_at'])
    
    # ============ A/B Testing ============
    
    op.create_table('experiments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('status', sa.String(20), default='draft'),
        sa.Column('traffic_percentage', sa.DECIMAL(5, 2), default=50.0),
        sa.Column('start_date', sa.TIMESTAMP(timezone=True)),
        sa.Column('end_date', sa.TIMESTAMP(timezone=True)),
        sa.Column('primary_metric', sa.String(100)),
        sa.Column('success_criteria', postgresql.JSONB()),
        sa.Column('control_metrics', postgresql.JSONB()),
        sa.Column('treatment_metrics', postgresql.JSONB()),
        sa.Column('statistical_significance', sa.DECIMAL(5, 4)),
        sa.Column('winner', sa.String(20)),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('created_by', sa.String(100)),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.CheckConstraint('traffic_percentage >= 0 AND traffic_percentage <= 100', name='valid_traffic_percentage')
    )
    op.create_index('idx_experiment_status', 'experiments', ['status'])
    op.create_index('idx_experiment_dates', 'experiments', ['start_date', 'end_date'])
    
    op.create_table('experiment_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('variant', sa.String(20), nullable=False),
        sa.Column('request_count', sa.BigInteger(), default=0),
        sa.Column('total_predictions', sa.BigInteger(), default=0),
        sa.Column('cumulative_mae', sa.DECIMAL(10, 4)),
        sa.Column('cumulative_rmse', sa.DECIMAL(10, 4)),
        sa.Column('cumulative_r2', sa.DECIMAL(5, 4)),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id']),
        sa.ForeignKeyConstraint(['model_id'], ['models.id']),
        sa.UniqueConstraint('experiment_id', 'variant')
    )
    op.create_index('idx_experiment_model_variant', 'experiment_models', ['variant'])
    
    # ============ Monitoring ============
    
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('entity_id', sa.String(100)),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('user_id', sa.String(100)),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True)),
        sa.Column('old_values', postgresql.JSONB()),
        sa.Column('new_values', postgresql.JSONB()),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.Text()),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_entity', 'audit_logs', ['entity_type', 'entity_id'])
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])
    
    op.create_table('data_sources',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('source_type', sa.String(50)),
        sa.Column('base_url', sa.Text()),
        sa.Column('requires_auth', sa.Boolean(), default=False),
        sa.Column('rate_limit_requests', sa.Integer()),
        sa.Column('rate_limit_window_seconds', sa.Integer()),
        sa.Column('cost_per_request', sa.DECIMAL(10, 6)),
        sa.Column('monthly_cost', sa.DECIMAL(10, 2)),
        sa.Column('total_requests', sa.BigInteger(), default=0),
        sa.Column('failed_requests', sa.BigInteger(), default=0),
        sa.Column('last_accessed_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('health_status', sa.String(20)),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('idx_data_source_active', 'data_sources', ['is_active'])
    op.create_index('idx_data_source_type', 'data_sources', ['source_type'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('data_sources')
    op.drop_table('audit_logs')
    op.drop_table('experiment_models')
    op.drop_table('experiments')
    op.drop_table('prediction_feedback')
    
    # Drop partitioned tables
    op.execute('DROP TABLE IF EXISTS predictions CASCADE')
    op.execute('DROP TABLE IF EXISTS player_features CASCADE')
    op.execute('DROP TABLE IF EXISTS player_game_logs CASCADE')
    
    op.drop_table('api_keys')
    op.drop_table('models')
    op.drop_table('games')
    op.drop_table('teams')
    op.drop_table('players')
    
    # Drop UUID extension
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')