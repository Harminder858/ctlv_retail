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
    
    return df

def prepare_data_for_modeling(df):
    """
    Prepare the data for CLTV modeling.
    """
    # Group by CustomerID and calculate recency, frequency, and monetary value
    summary = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,  # recency
        'InvoiceNo': 'count',  # frequency
        'TotalAmount': 'sum'  # monetary value
    })
    
    summary.columns = ['recency', 'frequency', 'monetary']
    
    # Calculate T (age of the customer in days)
    summary['T'] = (df['InvoiceDate'].max() - df.groupby('CustomerID')['InvoiceDate'].min()).dt.days
    
    return summary

def main():
    # Load data
    df = load_data('data/Online Retail.xlsx')
    
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare data for modeling
    summary_data = prepare_data_for_modeling(df_clean)
    
    print("Data preparation completed.")
    print(summary_data.head())

if __name__ == "__main__":
    main()
