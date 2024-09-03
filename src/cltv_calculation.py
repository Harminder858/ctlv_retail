import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data, time_horizon=12, discount_rate=0.01):
    """
    Calculate Customer Lifetime Value.
    """
    logger.info("Calculating CLTV")
    
    try:
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
        
        cltv_df = pd.DataFrame(cltv).reset_index()
        cltv_df.columns = ['CustomerID', 'CLV']
        
        logger.info("CLTV calculation completed successfully")
        logger.info(f"CLTV statistics:\n{cltv_df['CLV'].describe()}")
        logger.info(f"Sample of CLTV results:\n{cltv_df.sample(5)}")
        
        return cltv_df
    except Exception as e:
        logger.error(f"Error in CLTV calculation: {e}")
        return pd.DataFrame(columns=['CustomerID', 'CLV'])

def main():
    pass

if __name__ == "__main__":
    main()
