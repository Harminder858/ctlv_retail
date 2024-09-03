# CLTV Analysis with Pareto/NBD Model

This project demonstrates Customer Lifetime Value (CLTV) calculation using the Pareto/NBD model for a retail dataset. It includes data preparation, model fitting, CLTV calculation, and visualization components.

## Dataset

We use the "Online Retail" dataset from the UCI Machine Learning Repository. This dataset contains transactions from a UK-based online retail store.

## Project Structure

```
cltv_pareto_nbd_project/
│
├── data/
│   └── online_retail.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── model_fitting.py
│   ├── cltv_calculation.py
│   └── visualization.py
│
├── notebooks/
│   └── cltv_analysis.ipynb
│
├── dashboard/
│   ├── app.py
│   └── requirements.txt
│
├── tests/
│   ├── __init__.py
│   ├── test_data_preparation.py
│   ├── test_model_fitting.py
│   └── test_cltv_calculation.py
│
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cltv_pareto_nbd_project.git
   cd cltv_pareto_nbd_project
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the data and fit the models:
   ```
   python src/cltv_calculation.py
   ```

2. Run the dashboard:
   ```
   python dashboard/app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:8050/` to view the dashboard.

## Testing

Run the tests using pytest:
```
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).