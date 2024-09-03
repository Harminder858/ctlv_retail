import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the online retail dataset from an Excel file.
    """
    return pd.read_excel(file_path, engine='openpyxl')

def clean_data(df):
    """
    Clean and preprocess the data.
    """
    # Convert InvoiceDate to datetime if it's not already
    if df['InvoiceDate'].dtype != 'datetime64[ns]':
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate total amount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Remove rows with negative quantities or prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Remove rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to int
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Remove outliers
    for col in ['Quantity', 'UnitPrice', 'TotalAmount']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def prepare_data_for_modeling(df):
    """
    Prepare the data for CLTV modeling.
    """
    # Set the last date of the dataset
    last_date = df['InvoiceDate'].max()
    
    # Group by CustomerID and calculate recency, frequency, and monetary value
    summary = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (last_date - x.max()).days,  # recency
        'InvoiceNo': 'count',  # frequency
        'TotalAmount': 'sum'  # monetary value
    })
    
    summary.columns = ['recency', 'frequency', 'monetary']
    
    # Calculate T (age of the customer in days)
    summary['T'] = (last_date - df.groupby('CustomerID')['InvoiceDate'].min()).dt.days
    
    # Remove customers with frequency = 1 as they can cause issues in the model
    summary = summary[summary['frequency'] > 1]
    
    # Log transform monetary values
    summary['monetary'] = np.log1p(summary['monetary'])
    
    return summary

def calculate_rfm_scores(summary):
    """
    Calculate RFM scores as a fallback method.
    """
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    r_quartiles = pd.qcut(summary['recency'], q=4, labels=r_labels)
    f_quartiles = pd.qcut(summary['frequency'], q=4, labels=f_labels)
    m_quartiles = pd.qcut(summary['monetary'], q=4, labels=m_labels)
    
    summary['R'] = r_quartiles
    summary['F'] = f_quartiles
    summary['M'] = m_quartiles
    
    summary['RFM_Score'] = summary['R'].astype(str) + summary['F'].astype(str) + summary['M'].astype(str)
    summary['RFM_Score'] = summary['RFM_Score'].astype(int)
    
    return summary

def main():
    # Load data
    df = load_data('data/Online Retail.xlsx')
    
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare data for modeling
    summary_data = prepare_data_for_modeling(df_clean)
    
    # Calculate RFM scores as a fallback
    summary_data_with_rfm = calculate_rfm_scores(summary_data)
    
    print("Data preparation completed.")
    print(summary_data_with_rfm.head())

if __name__ == "__main__":
    main()
