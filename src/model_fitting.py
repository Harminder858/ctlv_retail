from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError
from src.data_preparation import load_data, clean_data, prepare_data_for_modeling, calculate_rfm_scores

def fit_bg_nbd_model(summary_data, max_attempts=5):
    """
    Fit the BG/NBD model with error handling and multiple attempts.
    """
    penalizer_coefs = [0.0, 0.001, 0.01, 0.1, 1.0]
    
    for attempt, penalizer_coef in enumerate(penalizer_coefs[:max_attempts], 1):
        try:
            bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
            bgf.fit(summary_data['frequency'], summary_data['recency'], summary_data['T'])
            print(f"BG/NBD model fitted successfully with penalizer_coef = {penalizer_coef}")
            return bgf
        except ConvergenceError:
            print(f"Attempt {attempt} failed with penalizer_coef = {penalizer_coef}")
    
    print("Failed to fit BG/NBD model after multiple attempts. Falling back to RFM analysis.")
    return None

def fit_gamma_gamma_model(summary_data, max_attempts=5):
    """
    Fit the Gamma-Gamma model with error handling and multiple attempts.
    """
    penalizer_coefs = [0.0, 0.001, 0.01, 0.1, 1.0]
    
    for attempt, penalizer_coef in enumerate(penalizer_coefs[:max_attempts], 1):
        try:
            ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
            ggf.fit(summary_data['frequency'], summary_data['monetary'])
            print(f"Gamma-Gamma model fitted successfully with penalizer_coef = {penalizer_coef}")
            return ggf
        except ConvergenceError:
            print(f"Attempt {attempt} failed with penalizer_coef = {penalizer_coef}")
    
    print("Failed to fit Gamma-Gamma model after multiple attempts.")
    return None

def main():
    # Load and prepare the actual data
    df = load_data('data/Online Retail.xlsx')
    df_clean = clean_data(df)
    summary_data = prepare_data_for_modeling(df_clean)
    
    bg_nbd_model = fit_bg_nbd_model(summary_data)
    if bg_nbd_model is None:
        summary_data_with_rfm = calculate_rfm_scores(summary_data)
        print("RFM analysis completed as a fallback method.")
        print(summary_data_with_rfm.head())
    else:
        gamma_gamma_model = fit_gamma_gamma_model(summary_data)
        if gamma_gamma_model is not None:
            print("Both models fitted successfully.")
            print("BG/NBD model parameters:", bg_nbd_model.params_)
            print("Gamma-Gamma model parameters:", gamma_gamma_model.params_)
        else:
            print("Only BG/NBD model fitted successfully.")
            print("BG/NBD model parameters:", bg_nbd_model.params_)

if __name__ == "__main__":
    main()
