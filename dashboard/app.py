import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from src.data_preparation import load_data, clean_data, prepare_data_for_modeling, calculate_rfm_scores
from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model
from src.cltv_calculation import calculate_cltv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and prepare data
logger.info("Starting data loading and preparation...")
df = load_data('data/Online Retail.xlsx')
df_clean = clean_data(df)
summary_data = prepare_data_for_modeling(df_clean)
logger.info("Data preparation completed.")

# Fit models and calculate CLTV or RFM
logger.info("Starting model fitting...")
bg_nbd_model = fit_bg_nbd_model(summary_data)
if bg_nbd_model is None:
    logger.info("Using RFM analysis as fallback method.")
    result_df = calculate_rfm_scores(summary_data)
    analysis_type = 'RFM'
else:
    gamma_gamma_model = fit_gamma_gamma_model(summary_data)
    if gamma_gamma_model is not None:
        logger.info("CLTV calculation completed.")
        result_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data)
        analysis_type = 'CLTV'
    else:
        logger.info("Using RFM analysis as fallback method.")
        result_df = calculate_rfm_scores(summary_data)
        analysis_type = 'RFM'

logger.info(f"Analysis type: {analysis_type}")
logger.info(f"Result dataframe shape: {result_df.shape}")
logger.info(f"Result dataframe head:\n{result_df.head()}")
logger.info(f"Result dataframe description:\n{result_df.describe()}")
logger.info(f"Number of unique values: {result_df.nunique()}")

# Merge result_df with summary_data for additional insights
result_df = result_df.merge(summary_data, left_index=True, right_index=True)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1(f"{analysis_type} Dashboard"),
    
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.H3('Distribution Plot'),
                dcc.Graph(id='distribution-plot'),
                
                html.H3('Scatter Plot'),
                dcc.Graph(id='scatter-plot'),
                
                html.H3('Top Customers'),
                dcc.Graph(id='top-customers'),
                
                html.Div([
                    html.Label('Select number of top customers:'),
                    dcc.Slider(
                        id='top-n-slider',
                        min=5,
                        max=100,
                        step=5,
                        value=20,
                        marks={i: str(i) for i in range(0, 101, 10)}
                    )
                ])
            ])
        ]),
        dcc.Tab(label='Customer Details', children=[
            html.Div([
                html.H3('Customer Information'),
                dash_table.DataTable(
                    id='customer-table',
                    columns=[{"name": i, "id": i} for i in result_df.columns],
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
    Output('distribution-plot', 'figure'),
    Input('top-n-slider', 'value')
)
def update_distribution_plot(top_n):
    logger.info(f"Updating distribution plot for top {top_n} customers")
    try:
        if analysis_type == 'CLTV':
            fig = px.histogram(result_df.nlargest(top_n, 'CLV'), x='CLV', nbins=20,
                               title=f'CLTV Distribution (Top {top_n} Customers)')
        else:
            fig = px.histogram(result_df.nlargest(top_n, 'RFM_Score'), x='RFM_Score', nbins=20,
                               title=f'RFM Score Distribution (Top {top_n} Customers)')
        fig.update_layout(bargap=0.2)
        return fig
    except Exception as e:
        logger.error(f"Error in update_distribution_plot: {e}")
        return px.histogram(title="Error in generating plot")

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('top-n-slider', 'value')
)
def update_scatter_plot(top_n):
    logger.info(f"Updating scatter plot for top {top_n} customers")
    try:
        if analysis_type == 'CLTV':
            fig = px.scatter(result_df.nlargest(top_n, 'CLV'), x='recency', y='frequency', 
                             size='monetary', color='CLV', hover_name=result_df.index,
                             title=f'Recency vs Frequency (Top {top_n} Customers)')
        else:
            fig = px.scatter(result_df.nlargest(top_n, 'RFM_Score'), x='recency', y='frequency', 
                             size='monetary', color='RFM_Score', hover_name=result_df.index,
                             title=f'Recency vs Frequency (Top {top_n} Customers)')
        return fig
    except Exception as e:
        logger.error(f"Error in update_scatter_plot: {e}")
        return px.scatter(title="Error in generating plot")

@app.callback(
    Output('top-customers', 'figure'),
    Input('top-n-slider', 'value')
)
def update_top_customers(top_n):
    logger.info(f"Updating top {top_n} customers plot")
    try:
        if analysis_type == 'CLTV':
            top_customers = result_df.nlargest(top_n, 'CLV')
            fig = px.bar(top_customers, x=top_customers.index, y='CLV',
                         title=f'Top {top_n} Customers by CLTV')
        else:
            top_customers = result_df.nlargest(top_n, 'RFM_Score')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_customers.index, y=top_customers['R_Score'], name='Recency', marker_color='blue'))
            fig.add_trace(go.Bar(x=top_customers.index, y=top_customers['F_Score'], name='Frequency', marker_color='green'))
            fig.add_trace(go.Bar(x=top_customers.index, y=top_customers['M_Score'], name='Monetary', marker_color='red'))
            fig.update_layout(barmode='group', title=f'Top {top_n} Customers by RFM Score (Grouped)')
        return fig
    except Exception as e:
        logger.error(f"Error in update_top_customers: {e}")
        return px.bar(title="Error in generating plot")

logger.info("Dashboard setup completed.")

if __name__ == '__main__':
    app.run_server(debug=True)
