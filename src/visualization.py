import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix

def plot_frequency_recency_matrix(bg_nbd_model, summary_data, ax=None):
    """
    Plot the frequency/recency matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    plot_frequency_recency_matrix(bg_nbd_model, 
                                  T=summary_data['T'].max(), 
                                  max_frequency=summary_data['frequency'].max(), 
                                  max_recency=summary_data['T'].max(),
                                  ax=ax)
    ax.set_title('Expected Number of Future Purchases for 1 Unit of Time,\nby Frequency and Recency of a Customer')
    return ax

def plot_probability_alive_matrix(bg_nbd_model, summary_data, ax=None):
    """
    Plot the probability alive matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    plot_probability_alive_matrix(bg_nbd_model, 
                                  max_frequency=summary_data['frequency'].max(), 
                                  max_recency=summary_data['T'].max(),
                                  ax=ax)
    ax.set_title('Probability Customer is Alive,\nby Frequency and Recency of a Customer')
    return ax

def plot_cltv_distribution(cltv_df, ax=None):
    """
    Plot the distribution of CLTV.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(cltv_df['clv'], kde=True, ax=ax)
    ax.set_title('Distribution of Customer Lifetime Value')
    ax.set_xlabel('Customer Lifetime Value')
    return ax

def main():
    # This is just a placeholder. In a real scenario, you would load the model, summary data, and CLTV results here.
    import pandas as pd
    from src.model_fitting import fit_bg_nbd_model
    from src.data_preparation import load_data, clean_data, prepare_data_for_modeling
    from src.cltv_calculation import calculate_cltv
    
    df = load_data('data/online_retail.csv')
    df_clean = clean_data(df)
    summary_data = prepare_data_for_modeling(df_clean)
    
    bg_nbd_model = fit_bg_nbd_model(summary_data)
    gamma_gamma_model = fit_gamma_gamma_model(summary_data)
    
    cltv_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plot_frequency_recency_matrix(bg_nbd_model, summary_data, ax=axes[0, 0])
    plot_probability_alive_matrix(bg_nbd_model, summary_data, ax=axes[0, 1])
    plot_cltv_distribution(cltv_df, ax=axes[1, 0])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()