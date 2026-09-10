from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.models import PredictionResponse
from ml.serving.predictor_v2 import ModelRegistry, ModelUnavailableError, PredictionCache
from scripts.create_dummy_model import FEATURE_NAMES, create_synthetic_artifacts


class FakeRedis:
    def __init__(self):
        self.values = {}

    def setex(self, key, ttl, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


def response(**changes):
    values = dict(player_id='1', player_name='Player', game_date=date(2025, 1, 1),
                  opponent_team='BOS', predictions={'points': 0.0}, confidence=0.0,
                  model_version='v1', model_accuracy={},
                  provenance={'source_kind': 'model_inference', 'model_version': 'v1', 'observed_at': '2025-01-01T00:00:00Z'},
                  prediction_id='p1')
    values.update(changes)
    return PredictionResponse(**values)


def test_zero_is_preserved_and_non_finite_is_rejected():
    assert response().predictions['points'] == 0.0
    with pytest.raises(ValidationError):
        response(predictions={'points': float('nan')})


def test_missing_model_is_an_error_not_a_synthetic_prediction(tmp_path):
    with pytest.raises(ModelUnavailableError):
        ModelRegistry(str(tmp_path)).load_model('v1', 'points')


def test_cache_keys_partition_source_and_model_version():
    cache = PredictionCache()
    base = ('1', date(2025, 1, 1), 'BOS')
    assert cache.get_cache_key(*base, 'v1', 'model_inference') != cache.get_cache_key(*base, 'v2', 'model_inference')
    assert cache.get_cache_key(*base, 'v1', 'model_inference') != cache.get_cache_key(*base, 'v1', 'synthetic_fixture')

    cache.redis_client = FakeRedis()
    cached = {'predictions': {'points': 0.0}, 'provenance': {
        'source_kind': 'model_inference', 'model_version': 'v1',
        'observed_at': '2025-01-01T00:00:00Z'}}
    cache.set(*base, cached, 'v1', 'model_inference')
    assert cache.get(*base, 'v1', 'model_inference') == cached
    assert cache.get(*base, 'v2', 'model_inference') is None


def test_synthetic_generator_is_isolated_non_overwriting_and_ordered(tmp_path):
    output = tmp_path / 'synthetic-demo'
    metadata = create_synthetic_artifacts(output, n_samples=30)
    assert metadata.exists()
    assert FEATURE_NAMES[0] == 'PTS_MA5'
    with pytest.raises(FileExistsError):
        create_synthetic_artifacts(output, n_samples=30)
    with pytest.raises(ValueError):
        create_synthetic_artifacts(tmp_path / 'models', n_samples=30)


def test_maintained_ui_labels_fixtures_and_omits_invented_kpis():
    maintained = [Path('frontend/app/page.tsx'), Path('frontend/app/dashboard/page.tsx')]
    text = '\n'.join(path.read_text() for path in maintained)
    assert 'Portfolio demonstration' in text
    assert 'Source: synthetic fixture' in Path(
        'frontend/components/dashboard/PredictionInterface.tsx').read_text()
    for unsupported in ('94.2%', '1.2M+', '99.99%', 'All systems normal'):
        assert unsupported not in text


def test_deploy_check_requires_provenance_notice_not_accuracy_claim():
    check = Path('scripts/deploy_check.py').read_text()
    assert 'Portfolio demonstration' in check
    assert 'Accuracy rate displayed' not in check
