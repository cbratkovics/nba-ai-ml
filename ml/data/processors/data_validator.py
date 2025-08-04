"""
Production-grade data validation for NBA ML system
Uses Great Expectations for comprehensive data quality checks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, date
import logging
from enum import Enum

try:
    import great_expectations as ge
    from great_expectations.dataset import PandasDataset
except ImportError:
    ge = None
    PandasDataset = None

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of data validation"""
    status: ValidationStatus
    passed_checks: int
    failed_checks: int
    warning_checks: int
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.FAILED
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


class NBADataValidator:
    """
    Comprehensive data validation for NBA ML pipeline
    Ensures data quality for production models
    """
    
    # Validation ranges for player statistics
    STAT_RANGES = {
        "points": (0, 100, 60),  # min, max, warning_threshold
        "rebounds": (0, 55, 30),
        "assists": (0, 30, 20),
        "steals": (0, 10, 8),
        "blocks": (0, 10, 8),
        "turnovers": (0, 15, 10),
        "minutes_played": (0, 48, 45),
        "field_goals_made": (0, 40, 25),
        "field_goals_attempted": (0, 50, 35),
        "three_pointers_made": (0, 15, 10),
        "three_pointers_attempted": (0, 25, 15),
        "free_throws_made": (0, 30, 20),
        "free_throws_attempted": (0, 35, 25),
        "personal_fouls": (0, 6, 6),
        "plus_minus": (-50, 50, 40),
        "field_goal_percentage": (0, 1, None),
        "three_point_percentage": (0, 1, None),
        "free_throw_percentage": (0, 1, None),
        "usage_rate": (0, 0.5, 0.4),
        "true_shooting_percentage": (0, 1, None),
    }
    
    # Required columns for different data types
    REQUIRED_COLUMNS = {
        "player_game_log": [
            "player_id", "game_id", "game_date", "minutes_played",
            "points", "rebounds", "assists", "field_goals_made",
            "field_goals_attempted", "free_throws_made", "free_throws_attempted"
        ],
        "features": [
            "player_id", "game_id", "game_date", "features"
        ],
        "prediction_input": [
            "player_id", "game_date", "opponent_team"
        ]
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
        self.ge_context = None
        
        if ge:
            try:
                self.ge_context = ge.data_context.DataContext()
            except:
                logger.warning("Could not initialize Great Expectations context")
    
    def validate_player_game_log(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate raw player game data
        
        Args:
            df: DataFrame with player game logs
            
        Returns:
            ValidationResult with detailed findings
        """
        errors = []
        warnings = []
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            failed_checks += 1
            return ValidationResult(
                status=ValidationStatus.FAILED,
                passed_checks=0,
                failed_checks=1,
                warning_checks=0,
                errors=errors,
                warnings=warnings,
                metadata={"row_count": 0}
            )
        
        # 1. Check required columns
        missing_columns = set(self.REQUIRED_COLUMNS["player_game_log"]) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            failed_checks += 1
        else:
            passed_checks += 1
        
        # 2. Check data types
        type_checks = self._check_data_types(df)
        if type_checks["errors"]:
            errors.extend(type_checks["errors"])
            failed_checks += len(type_checks["errors"])
        else:
            passed_checks += 1
        
        # 3. Check value ranges for each stat
        for col, (min_val, max_val, warn_threshold) in self.STAT_RANGES.items():
            if col in df.columns:
                # Check for values outside valid range
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    errors.append(f"{col}: {invalid_count} values outside range [{min_val}, {max_val}]")
                    failed_checks += 1
                else:
                    passed_checks += 1
                
                # Check for warning threshold
                if warn_threshold is not None:
                    warn_mask = df[col] > warn_threshold
                    if warn_mask.any():
                        warn_count = warn_mask.sum()
                        warnings.append(f"{col}: {warn_count} values exceed warning threshold {warn_threshold}")
                        warning_checks += 1
        
        # 4. Check for anomalies
        anomalies = self._detect_anomalies(df)
        if anomalies["errors"]:
            errors.extend(anomalies["errors"])
            failed_checks += len(anomalies["errors"])
        if anomalies["warnings"]:
            warnings.extend(anomalies["warnings"])
            warning_checks += len(anomalies["warnings"])
        
        # 5. Check temporal consistency
        temporal_checks = self._check_temporal_consistency(df)
        if temporal_checks["errors"]:
            errors.extend(temporal_checks["errors"])
            failed_checks += len(temporal_checks["errors"])
        else:
            passed_checks += 1
        
        # 6. Check for duplicates
        if "player_id" in df.columns and "game_id" in df.columns:
            duplicates = df.duplicated(subset=["player_id", "game_id"])
            if duplicates.any():
                errors.append(f"Found {duplicates.sum()} duplicate player-game combinations")
                failed_checks += 1
            else:
                passed_checks += 1
        
        # 7. Check data completeness
        completeness = self._check_completeness(df)
        if completeness["errors"]:
            errors.extend(completeness["errors"])
            failed_checks += len(completeness["errors"])
        if completeness["warnings"]:
            warnings.extend(completeness["warnings"])
            warning_checks += len(completeness["warnings"])
        
        # 8. Check logical consistency
        logical_checks = self._check_logical_consistency(df)
        if logical_checks["errors"]:
            errors.extend(logical_checks["errors"])
            failed_checks += len(logical_checks["errors"])
        else:
            passed_checks += 1
        
        # Determine overall status
        if failed_checks > 0:
            status = ValidationStatus.FAILED
        elif warning_checks > 0 and self.strict_mode:
            status = ValidationStatus.FAILED
            errors.extend(warnings)
            failed_checks += warning_checks
        elif warning_checks > 0:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        return ValidationResult(
            status=status,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            errors=errors,
            warnings=warnings,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "date_range": (df["game_date"].min() if "game_date" in df.columns else None,
                             df["game_date"].max() if "game_date" in df.columns else None)
            }
        )
    
    def validate_features(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate engineered features
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            ValidationResult with detailed findings
        """
        errors = []
        warnings = []
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0
        
        # Check for required columns
        missing_columns = set(self.REQUIRED_COLUMNS["features"]) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            failed_checks += 1
        else:
            passed_checks += 1
        
        # Check for NaN values
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            for col in nan_columns:
                nan_count = df[col].isna().sum()
                nan_pct = (nan_count / len(df)) * 100
                if nan_pct > 10:
                    errors.append(f"{col}: {nan_pct:.1f}% NaN values (threshold: 10%)")
                    failed_checks += 1
                else:
                    warnings.append(f"{col}: {nan_pct:.1f}% NaN values")
                    warning_checks += 1
        else:
            passed_checks += 1
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_mask = np.isinf(df[col].values)
            if inf_mask.any():
                errors.append(f"{col}: Contains {inf_mask.sum()} infinite values")
                failed_checks += 1
        
        if not errors:
            passed_checks += 1
        
        # Check feature ranges
        feature_range_checks = self._check_feature_ranges(df)
        if feature_range_checks["errors"]:
            errors.extend(feature_range_checks["errors"])
            failed_checks += len(feature_range_checks["errors"])
        if feature_range_checks["warnings"]:
            warnings.extend(feature_range_checks["warnings"])
            warning_checks += len(feature_range_checks["warnings"])
        
        # Check temporal consistency for rolling features
        if "game_date" in df.columns:
            temporal_checks = self._check_feature_temporal_consistency(df)
            if temporal_checks["errors"]:
                errors.extend(temporal_checks["errors"])
                failed_checks += len(temporal_checks["errors"])
            else:
                passed_checks += 1
        
        # Check feature correlations
        correlation_checks = self._check_feature_correlations(df)
        if correlation_checks["warnings"]:
            warnings.extend(correlation_checks["warnings"])
            warning_checks += len(correlation_checks["warnings"])
        
        # Determine overall status
        if failed_checks > 0:
            status = ValidationStatus.FAILED
        elif warning_checks > 0 and self.strict_mode:
            status = ValidationStatus.FAILED
        elif warning_checks > 0:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        return ValidationResult(
            status=status,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            errors=errors,
            warnings=warnings,
            metadata={
                "row_count": len(df),
                "feature_count": len(numeric_cols),
                "nan_columns": len(nan_columns)
            }
        )
    
    def validate_prediction_input(self, data: Dict) -> ValidationResult:
        """
        Validate API prediction requests
        
        Args:
            data: Dictionary with prediction request data
            
        Returns:
            ValidationResult with detailed findings
        """
        errors = []
        warnings = []
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0
        
        # Check required fields
        missing_fields = set(self.REQUIRED_COLUMNS["prediction_input"]) - set(data.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            failed_checks += 1
        else:
            passed_checks += 1
        
        # Validate player_id
        if "player_id" in data:
            if not data["player_id"]:
                errors.append("player_id cannot be empty")
                failed_checks += 1
            else:
                passed_checks += 1
        
        # Validate game_date
        if "game_date" in data:
            try:
                game_date = pd.to_datetime(data["game_date"]).date()
                
                # Check if date is in the future
                if game_date <= date.today():
                    errors.append(f"game_date must be in the future (got: {game_date})")
                    failed_checks += 1
                else:
                    passed_checks += 1
                
                # Check if date is not too far in the future
                max_future_days = 365
                if (game_date - date.today()).days > max_future_days:
                    warnings.append(f"game_date is more than {max_future_days} days in the future")
                    warning_checks += 1
                    
            except Exception as e:
                errors.append(f"Invalid game_date format: {str(e)}")
                failed_checks += 1
        
        # Validate opponent_team
        if "opponent_team" in data:
            valid_teams = [
                "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
                "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
                "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
                "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
            ]
            
            if data["opponent_team"] not in valid_teams:
                errors.append(f"Invalid opponent_team: {data['opponent_team']}")
                failed_checks += 1
            else:
                passed_checks += 1
        
        # Check if we have enough historical data
        if "historical_games" in data:
            if data["historical_games"] < 10:
                warnings.append(f"Only {data['historical_games']} historical games available (recommended: 10+)")
                warning_checks += 1
            else:
                passed_checks += 1
        
        # Determine overall status
        if failed_checks > 0:
            status = ValidationStatus.FAILED
        elif warning_checks > 0:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        return ValidationResult(
            status=status,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            errors=errors,
            warnings=warnings,
            metadata=data
        )
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check data types are appropriate"""
        errors = []
        warnings = []
        
        # Expected types for common columns
        expected_types = {
            "game_date": ["datetime64", "object"],  # Allow string dates
            "points": ["int", "float"],
            "rebounds": ["int", "float"],
            "assists": ["int", "float"],
            "minutes_played": ["float", "int"],
            "field_goal_percentage": ["float"],
        }
        
        for col, expected in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not any(t in actual_type for t in expected):
                    warnings.append(f"{col} has type {actual_type}, expected {expected}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect statistical anomalies in the data"""
        errors = []
        warnings = []
        
        # Check for extreme outliers (>4 std dev)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and len(df[col]) > 10:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    extreme_outliers = (z_scores > 4).sum()
                    outliers = (z_scores > 3).sum()
                    
                    if extreme_outliers > 0:
                        errors.append(f"{col}: {extreme_outliers} extreme outliers (>4 std dev)")
                    elif outliers > len(df) * 0.05:  # More than 5% outliers
                        warnings.append(f"{col}: {outliers} outliers (>3 std dev)")
        
        # Check for impossible combinations
        if "field_goals_made" in df.columns and "field_goals_attempted" in df.columns:
            impossible = df["field_goals_made"] > df["field_goals_attempted"]
            if impossible.any():
                errors.append(f"Found {impossible.sum()} games with FGM > FGA")
        
        if "three_pointers_made" in df.columns and "field_goals_made" in df.columns:
            impossible = df["three_pointers_made"] > df["field_goals_made"]
            if impossible.any():
                errors.append(f"Found {impossible.sum()} games with 3PM > FGM")
        
        # Check for negative values
        for col in ["points", "rebounds", "assists", "minutes_played"]:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    errors.append(f"{col}: {negative} negative values")
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check temporal consistency of the data"""
        errors = []
        warnings = []
        
        if "game_date" not in df.columns:
            return {"errors": errors, "warnings": warnings}
        
        # Convert to datetime if needed
        try:
            df["game_date"] = pd.to_datetime(df["game_date"])
        except:
            errors.append("Cannot parse game_date as datetime")
            return {"errors": errors, "warnings": warnings}
        
        # Check for future dates in historical data
        future_games = df["game_date"] > pd.Timestamp.now()
        if future_games.any():
            errors.append(f"Found {future_games.sum()} games with future dates in historical data")
        
        # Check date range is reasonable
        min_date = df["game_date"].min()
        max_date = df["game_date"].max()
        
        if min_date < pd.Timestamp("2000-01-01"):
            warnings.append(f"Data contains very old games (earliest: {min_date})")
        
        # Check for gaps in data
        if "player_id" in df.columns:
            for player_id in df["player_id"].unique()[:10]:  # Check first 10 players
                player_df = df[df["player_id"] == player_id].sort_values("game_date")
                if len(player_df) > 1:
                    date_diff = player_df["game_date"].diff()
                    max_gap = date_diff.max()
                    if max_gap > pd.Timedelta(days=180):
                        warnings.append(f"Player {player_id} has {max_gap.days} day gap in games")
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check data completeness"""
        errors = []
        warnings = []
        
        # Check for missing values in critical columns
        critical_columns = ["player_id", "game_id", "game_date", "points"]
        
        for col in critical_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                
                if missing_pct > 5:
                    errors.append(f"{col}: {missing_pct:.1f}% missing values")
                elif missing_pct > 0:
                    warnings.append(f"{col}: {missing_pct:.1f}% missing values")
        
        # Check if players have enough games
        if "player_id" in df.columns:
            player_game_counts = df.groupby("player_id").size()
            low_game_players = (player_game_counts < 10).sum()
            
            if low_game_players > 0:
                warnings.append(f"{low_game_players} players have fewer than 10 games")
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_logical_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check logical consistency of basketball statistics"""
        errors = []
        
        # Players can't play more than 48 minutes (overtime possible, but check > 60)
        if "minutes_played" in df.columns:
            excessive_minutes = (df["minutes_played"] > 60).sum()
            if excessive_minutes > 0:
                errors.append(f"{excessive_minutes} games with >60 minutes played")
        
        # Field goal percentage calculation check
        if all(col in df.columns for col in ["field_goals_made", "field_goals_attempted", "field_goal_percentage"]):
            mask = df["field_goals_attempted"] > 0
            calculated_fg_pct = df.loc[mask, "field_goals_made"] / df.loc[mask, "field_goals_attempted"]
            actual_fg_pct = df.loc[mask, "field_goal_percentage"]
            
            # Allow small difference due to rounding
            diff = np.abs(calculated_fg_pct - actual_fg_pct)
            inconsistent = (diff > 0.01).sum()
            
            if inconsistent > 0:
                errors.append(f"{inconsistent} games with inconsistent FG%")
        
        # Personal fouls can't exceed 6 (player fouls out)
        if "personal_fouls" in df.columns:
            excessive_fouls = (df["personal_fouls"] > 6).sum()
            if excessive_fouls > 0:
                errors.append(f"{excessive_fouls} games with >6 personal fouls")
        
        return {"errors": errors, "warnings": []}
    
    def _check_feature_ranges(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check if feature values are in expected ranges"""
        errors = []
        warnings = []
        
        # Check rolling averages are reasonable
        for col in df.columns:
            if "_ma" in col or "_MA" in col:  # Moving average columns
                if col in df.columns:
                    # Moving averages shouldn't be negative for counting stats
                    if any(keyword in col.lower() for keyword in ["points", "rebounds", "assists"]):
                        negative_values = (df[col] < 0).sum()
                        if negative_values > 0:
                            errors.append(f"{col}: {negative_values} negative values in moving average")
                    
                    # Check if values are reasonable
                    max_val = df[col].max()
                    if "points" in col.lower() and max_val > 50:
                        warnings.append(f"{col}: Maximum value {max_val:.1f} seems high for points average")
        
        # Check normalized features are in [0, 1] or [-1, 1]
        for col in df.columns:
            if "normalized" in col.lower() or "scaled" in col.lower():
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val < -1.1 or max_val > 1.1:
                    warnings.append(f"{col}: Values outside [-1, 1] range ({min_val:.2f}, {max_val:.2f})")
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_feature_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check temporal consistency of features"""
        errors = []
        
        # Sort by date
        df_sorted = df.sort_values("game_date")
        
        # Check that rolling features make sense temporally
        # For example, MA10 should be relatively smooth
        for col in df_sorted.columns:
            if "_MA10" in col or "_ma10" in col:
                if len(df_sorted) > 20:
                    # Calculate change rate
                    values = df_sorted[col].values
                    diffs = np.diff(values)
                    
                    # Large jumps in 10-game average are suspicious
                    large_jumps = np.abs(diffs) > (values[:-1] * 0.5)  # 50% change
                    if large_jumps.sum() > len(diffs) * 0.1:  # More than 10% large jumps
                        errors.append(f"{col}: Excessive volatility in 10-game moving average")
        
        return {"errors": errors, "warnings": []}
    
    def _check_feature_correlations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for suspicious feature correlations"""
        warnings = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 2:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Find perfect correlations (excluding diagonal)
            perfect_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.99:
                        perfect_corr.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if perfect_corr:
                for col1, col2 in perfect_corr[:5]:  # Limit to first 5
                    warnings.append(f"Near-perfect correlation between {col1} and {col2}")
        
        return {"errors": [], "warnings": warnings}
    
    def generate_validation_report(self, 
                                 validation_results: List[ValidationResult],
                                 output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            validation_results: List of validation results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("NBA DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        for i, result in enumerate(validation_results, 1):
            report.append(f"Validation {i}:")
            report.append(f"  Status: {result.status.value.upper()}")
            report.append(f"  Passed Checks: {result.passed_checks}")
            report.append(f"  Failed Checks: {result.failed_checks}")
            report.append(f"  Warnings: {result.warning_checks}")
            
            if result.errors:
                report.append("  Errors:")
                for error in result.errors:
                    report.append(f"    - {error}")
            
            if result.warnings:
                report.append("  Warnings:")
                for warning in result.warnings:
                    report.append(f"    - {warning}")
            
            report.append("")
        
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_str)
        
        return report_str