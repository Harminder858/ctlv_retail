from lifetimes import BetaGeoFitter, GammaGammaFitter

def calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data, time_horizon=12, discount_rate=0.01):
    """
    Calculate Customer Lifetime Value.
    """
    cltv = gamma_gamma_model.customer_lifetime_value(
        bg_nbd_model,
        summary_data['frequency'],
        summary_data['recency'],
        summary_data['T'],
        summary_data['monetary'],
        time=time_horizon,
        freq='D',
        discount_rate=discount_rate
    )
    return cltv.reset_index()

def main():
    # This is just a placeholder. In a real scenario, you would load the models and summary data here.
    import pandas as pd
    from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model
    
    summary_data = pd.DataFrame({
        'frequency': [1, 2, 3],
        'recency': [10, 20, 30],
        'T': [100, 100, 100],
        'monetary': [100, 200, 300]
    })
    
    bg_nbd_model = fit_bg_nbd_model(summary_data)
    gamma_gamma_model = fit_gamma_gamma_model(summary_data)
    
    cltv_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data)
    
    print("CLTV calculation completed.")
    print(cltv_df.head())

if __name__ == "__main__":
    main()