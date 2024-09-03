import pandas as pd
import pytest
from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model

@pytest.fixture
def sample_summary_data():
    return pd.DataFrame({
        'frequency': [1, 2, 3, 4, 5],
        'recency': [10, 20, 30, 40, 50],
        'T': [100, 100, 100, 100, 100],
        'monetary': [100, 200, 300, 400, 500]
    })

def test_fit_bg_nbd_model(sample_summary_data):
    model = fit_bg_nbd_model(sample_summary_data)
    
    assert hasattr(model, 'predict')
    assert hasattr(model, 'params_')
    assert len(model.params_) == 4  # r, alpha, a, b

def test_fit_gamma_gamma_model(sample_summary_data):
    model = fit_gamma_gamma_model(sample_summary_data)
    
    assert hasattr(model, 'conditional_expected_average_profit')
    assert hasattr(model, 'params_')
    assert len(model.params_) == 3  # p, q, v