from lifetimes import BetaGeoFitter, GammaGammaFitter

def fit_bg_nbd_model(summary_data):
    """
    Fit the BG/NBD model.
    """
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary_data['frequency'], summary_data['recency'], summary_data['T'])
    return bgf

def fit_gamma_gamma_model(summary_data):
    """
    Fit the Gamma-Gamma model.
    """
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary_data['frequency'], summary_data['monetary'])
    return ggf

def main():
    # This is just a placeholder. In a real scenario, you would load the summary data here.
    import pandas as pd
    summary_data = pd.DataFrame({
        'frequency': [1, 2, 3],
        'recency': [10, 20, 30],
        'T': [100, 100, 100],
        'monetary': [100, 200, 300]
    })
    
    bg_nbd_model = fit_bg_nbd_model(summary_data)
    gamma_gamma_model = fit_gamma_gamma_model(summary_data)
    
    print("Models fitted successfully.")
    print("BG/NBD model parameters:", bg_nbd_model.params_)
    print("Gamma-Gamma model parameters:", gamma_gamma_model.params_)

if __name__ == "__main__":
    main()