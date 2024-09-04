import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
df = pd.read_excel('data/Online Retail.xlsx')

# Data preparation
logger.info("Preparing data for CLTV analysis...")
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
summary_data = summary_data_from_transaction_data(
    df, 'CustomerID', 'InvoiceDate', 'TotalAmount', 
    observation_period_end=df['InvoiceDate'].max()
)

# Fit models and calculate CLTV
logger.info("Fitting models and calculating CLTV...")
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(summary_data['frequency'], summary_data['recency'], summary_data['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(summary_data['frequency'], summary_data['monetary_value'])

time_horizon = 12  # months
cltv = ggf.customer_lifetime_value(
    bgf, summary_data['frequency'], summary_data['recency'], summary_data['T'], 
    summary_data['monetary_value'], time=time_horizon, freq='D', discount_rate=0.01
)

result_df = summary_data.join(cltv.rename('CLV'))

logger.info(f"CLTV calculation completed. Result shape: {result_df.shape}")
logger.info(f"Result dataframe head:\n{result_df.head()}")

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Customer Lifetime Value (CLTV) Analysis Dashboard"),
    
    html.Div([
        html.Label("Select number of customers to display:"),
        dcc.Slider(
            id='customer-slider',
            min=10,
            max=100,
            step=10,
            value=20,
            marks={i: str(i) for i in range(10, 101, 10)},
        ),
    ], style={'width': '50%', 'margin': '20px auto'}),
    
    dcc.Tabs([
        dcc.Tab(label='CLTV Overview', children=[
            html.Div([
                html.H3('CLTV Distribution'),
                dcc.Graph(id='cltv-distribution-plot'),
                
                html.H3('Recency vs Frequency'),
                dcc.Graph(id='recency-frequency-plot'),
                
                html.H3('Top Customers by CLTV'),
                dcc.Graph(id='top-customers'),
            ])
        ]),
        dcc.Tab(label='Customer Details', children=[
            html.Div([
                html.H3('Customer Information'),
                dash_table.DataTable(
                    id='customer-table',
                    columns=[{"name": i, "id": i} for i in result_df.reset_index().columns],
                    data=result_df.reset_index().to_dict('records'),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    sort_action='native',
                    filter_action='native'
                )
            ])
        ])
    ])
])

@app.callback(
    [Output('cltv-distribution-plot', 'figure'),
     Output('recency-frequency-plot', 'figure'),
     Output('top-customers', 'figure')],
    Input('customer-slider', 'value')
)
def update_graphs(n_customers):
    logger.info(f"Updating graphs for {n_customers} randomly selected customers")
    try:
        # Randomly select customers
        selected_customers = result_df.sample(n=n_customers)
        
        # CLTV Distribution
        cltv_dist = px.histogram(selected_customers, x='CLV', nbins=20,
                                 title=f'CLTV Distribution (Random {n_customers} Customers)')
        cltv_dist.update_layout(
            xaxis_title="Customer Lifetime Value",
            yaxis_title="Count",
            bargap=0.2
        )
        
        # Recency vs Frequency
        recency_freq = px.scatter(selected_customers, x='recency', y='frequency', 
                                  size='monetary_value', color='CLV', hover_name=selected_customers.index,
                                  title=f'Recency vs Frequency (Random {n_customers} Customers)')
        recency_freq.update_layout(
            xaxis_title="Recency (days)",
            yaxis_title="Frequency",
            coloraxis_colorbar_title="CLTV"
        )
        
        # Top Customers
        top_customers = selected_customers.nlargest(10, 'CLV')
        top_cust_plot = px.bar(top_customers, x=top_customers.index, y='CLV',
                               title=f'Top 10 Customers by CLTV (from Random {n_customers} Customers)')
        top_cust_plot.update_layout(
            xaxis_title="Customer ID",
            yaxis_title="Customer Lifetime Value"
        )
        
        return cltv_dist, recency_freq, top_cust_plot
    except Exception as e:
        logger.error(f"Error in update_graphs: {e}")
        return px.histogram(title="Error in generating plot"), px.scatter(title="Error in generating plot"), px.bar(title="Error in generating plot")

logger.info("Dashboard setup completed.")

if __name__ == '__main__':
    app.run_server(debug=True)
