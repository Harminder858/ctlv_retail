import pandas as pd
import pytest
from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model
from src.cltv_calculation import calculate_cltv

@pytest.fixture
def sample_summary_data():
    return pd.DataFrame({
        'frequency': [1, 2, 3, 4, 5],
        'recency': [10, 20, 30, 40, 50],
        'T': [100, 100, 100, 100, 100],
        'monetary': [100, 200, 300, 400, 500]
    })

@pytest.fixture
def fitted_models(sample_summary_data):
    bg_nbd_model = fit_bg_nbd_model(sample_summary_data)
    gamma_gamma_model = fit_gamma_gamma_model(sample_summary_data)
    return bg_nbd_model, gamma_gamma_model

def test_calculate_cltv(sample_summary_data, fitted_models):
    bg_nbd_model, gamma_gamma_model = fitted_models
    cltv_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, sample_summary_data)
    
    assert 'clv' in cltv_df.columns
    assert len(cltv_df) == len(sample_summary_data)
    assert (cltv_df['clv'] >= 0).all()