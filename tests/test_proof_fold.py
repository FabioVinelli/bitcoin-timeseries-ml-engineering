"""
Proof Fold Smoke Tests
======================
Basic tests to ensure the proof fold pipeline runs without errors.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFeatureEngineering:
    """Test feature engineering module imports and constants."""
    
    def test_imports(self):
        """Verify all feature engineering modules import correctly."""
        from src.features.ark_layers import LAYER_1_FEATURES, LAYER_2_FEATURES, LAYER_3_LABELS
        from src.features.halving_features import HALVING_DATES, add_halving_features
        from src.features.feature_engineering import (
            create_percentile_regime_labels,
            PERCENTILE_WINDOW,
            PERCENTILE_BUY,
            PERCENTILE_SELL
        )
        
        assert len(LAYER_1_FEATURES) > 0
        assert len(LAYER_2_FEATURES) > 0
        assert len(LAYER_3_LABELS) > 0
        assert len(HALVING_DATES) == 4
    
    def test_percentile_constants(self):
        """Verify percentile thresholds are within valid range."""
        from src.features.feature_engineering import (
            PERCENTILE_WINDOW,
            PERCENTILE_BUY,
            PERCENTILE_SELL
        )
        
        assert 0 < PERCENTILE_BUY < PERCENTILE_SELL < 1
        assert PERCENTILE_WINDOW >= 365  # At least 1 year


class TestValidation:
    """Test validation module imports and functions."""
    
    def test_imports(self):
        """Verify validation module imports correctly."""
        from src.validation.split_validator import (
            validate_temporal_splits,
            validate_no_leakage,
            validate_label_distribution,
            compute_always_hold_baseline
        )
    
    def test_leakage_detection(self):
        """Verify leakage detection catches forbidden columns."""
        from src.validation.split_validator import validate_no_leakage
        
        # Should pass with clean features
        passed, msg = validate_no_leakage(['hash_rate', 'active_addresses'], raise_on_error=False)
        assert passed
        
        # Should fail with forbidden columns
        passed, msg = validate_no_leakage(['hash_rate', 'Close'], raise_on_error=False)
        assert not passed


class TestHalvingFeatures:
    """Test halving features computation."""
    
    def test_halving_dates(self):
        """Verify halving dates are correct."""
        from src.features.halving_features import HALVING_DATES
        from datetime import datetime
        
        assert HALVING_DATES[2] == datetime(2016, 7, 9)
        assert HALVING_DATES[3] == datetime(2020, 5, 11)
        assert HALVING_DATES[4] == datetime(2024, 4, 20)
    
    def test_get_halving_cycle(self):
        """Verify cycle detection logic."""
        from src.features.halving_features import get_halving_cycle
        from datetime import datetime
        
        assert get_halving_cycle(datetime(2017, 1, 1)) == 2
        assert get_halving_cycle(datetime(2021, 1, 1)) == 3
        assert get_halving_cycle(datetime(2025, 1, 1)) == 4


class TestReportsExist:
    """Test that reports were generated."""
    
    def test_reports_directory(self):
        """Verify reports directory exists and contains expected files."""
        reports_dir = project_root / "reports"
        
        if reports_dir.exists():
            assert (reports_dir / "split_summary.json").exists(), "split_summary.json missing"
            assert (reports_dir / "metrics.json").exists(), "metrics.json missing"
            assert (reports_dir / "label_distribution.csv").exists(), "label_distribution.csv missing"
        else:
            pytest.skip("Reports directory not found - run proof_fold.py first")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
