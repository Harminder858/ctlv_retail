import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from src.data_preparation import load_data, clean_data, prepare_data_for_modeling, calculate_rfm_scores
from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model
from src.cltv_calculation import calculate_cltv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting data loading and preparation...")
# Load and prepare data
df = load_data('data/Online Retail.xlsx')
df_clean = clean_data(df)
summary_data = prepare_data_for_modeling(df_clean)
logger.info("Data preparation completed.")

logger.info("Starting model fitting...")
# Fit models
bg_nbd_model = fit_bg_nbd_model(summary_data)
if bg_nbd_model is None:
    logger.info("Using RFM analysis as fallback method.")
    summary_data_with_rfm = calculate_rfm_scores(summary_data)
    analysis_type = 'RFM'
    result_df = summary_data_with_rfm
else:
    gamma_gamma_model = fit_gamma_gamma_model(summary_data)
    if gamma_gamma_model is not None:
        logger.info("CLTV calculation completed.")
        cltv_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data)
        analysis_type = 'CLTV'
        result_df = cltv_df
    else:
        logger.info("Using RFM analysis as fallback method.")
        summary_data_with_rfm = calculate_rfm_scores(summary_data)
        analysis_type = 'RFM'
        result_df = summary_data_with_rfm

logger.info(f"Analysis type: {analysis_type}")
logger.info(f"Result dataframe shape: {result_df.shape}")
logger.info(f"Result dataframe head:\n{result_df.head()}")

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for Gunicorn

# Define the layout
app.layout = html.Div([
    html.H1(f"{analysis_type} Dashboard"),
    
    dcc.Graph(id='distribution-plot'),
    
    dcc.Graph(id='top-customers'),
    
    dcc.Slider(
        id='top-n-slider',
        min=5,
        max=50,
        step=5,
        value=20,
        marks={i: str(i) for i in range(5, 51, 5)}
    )
])

@app.callback(
    Output('distribution-plot', 'figure'),
    Input('top-n-slider', 'value')
)
def update_distribution_plot(
