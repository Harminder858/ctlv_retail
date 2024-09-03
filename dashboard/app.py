import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from src.data_preparation import load_data, clean_data, prepare_data_for_modeling
from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model
from src.cltv_calculation import calculate_cltv

# Load and prepare data
df = load_data('data/Online Retail.xlsx')
df_clean = clean_data(df)
summary_data = prepare_data_for_modeling(df_clean)

# Fit models
try:
    bg_nbd_model = fit_bg_nbd_model(summary_data)
    gamma_gamma_model = fit_gamma_gamma_model(summary_data)
    
    # Calculate CLTV
    cltv_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data)
except ValueError as e:
    print(f"Error in model fitting or CLTV calculation: {e}")
    cltv_df = pd.DataFrame(columns=['CustomerID', 'clv'])

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for Gunicorn

# Define the layout
app.layout = html.Div([
    html.H1("CLTV Dashboard"),
    
    dcc.Graph(id='cltv-histogram'),
    
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
    Output('cltv-histogram', 'figure'),
    Input('top-n-slider', 'value')
)
def update_cltv_histogram(top_n):
    if cltv_df.empty:
        return px.histogram(title="No CLTV data available")
    fig = px.histogram(cltv_df.nlargest(top_n, 'clv'), x='clv', nbins=20,
                       title=f'CLTV Distribution (Top {top_n} Customers)')
    return fig

@app.callback(
    Output('top-customers', 'figure'),
    Input('top-n-slider', 'value')
)
def update_top_customers(top_n):
    if cltv_df.empty:
        return px.bar(title="No CLTV data available")
    top_customers = cltv_df.nlargest(top_n, 'clv')
    fig = px.bar(top_customers, x=top_customers.index, y='clv',
                 title=f'Top {top_n} Customers by CLTV')
    return fig

