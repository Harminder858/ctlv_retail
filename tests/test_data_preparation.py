import pandas as pd
import pytest
from src.data_preparation import clean_data, prepare_data_for_modeling

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'CustomerID': [1, 1, 2, 2, 3],
        'InvoiceDate': ['2021-01-01', '2021-01-15', '2021-02-01', '2021-02-15', '2021-03-01'],
        'InvoiceNo': ['A001', 'A002', 'B001', 'B002', 'C001'],
        'Quantity': [1, 2, 3, 4, 5],
        'UnitPrice': [10, 20, 30, 40, 50]
    })

def test_clean_data(sample_df):
    cleaned_df = clean_data(sample_df)
    
    assert cleaned_df['InvoiceDate'].dtype == 'datetime64[ns]'
    assert 'TotalAmount' in cleaned_df.columns
    assert (cleaned_df['Quantity'] > 0).all()
    assert (cleaned_df['UnitPrice'] > 0).all()
    assert cleaned_df['CustomerID'].dtype == 'int64'

def test_prepare_data_for_modeling(sample_df):
    cleaned_df = clean_data(sample_df)
    summary_data = prepare_data_for_modeling(cleaned_df)
    
    assert set(summary_data.columns) == {'recency', 'frequency', 'monetary', 'T'}
    assert len(summary_data) == len(sample_df['CustomerID'].unique())
    assert (summary_data['frequency'] > 0).all()
    assert (summary_data['monetary'] > 0).all()
    assert (summary_data['T'] >= summary_data['recency']).all()