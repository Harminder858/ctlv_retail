{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CLTV Analysis with Pareto/NBD Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from src.data_preparation import load_data, clean_data, prepare_data_for_modeling\n",
    "from src.model_fitting import fit_bg_nbd_model, fit_gamma_gamma_model\n",
    "from src.cltv_calculation import calculate_cltv\n",
    "from src.visualization import plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_cltv_distribution"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = load_data('../data/online_retail.csv')\n",
    "df_clean = clean_data(df)\n",
    "summary_data = prepare_data_for_modeling(df_clean)\n",
    "summary_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Fitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bg_nbd_model = fit_bg_nbd_model(summary_data)\n",
    "gamma_gamma_model = fit_gamma_gamma_model(summary_data)\n",
    "print(\"BG/NBD model parameters:\", bg_nbd_model.params_)\n",
    "print(\"Gamma-Gamma model parameters:\", gamma_gamma_model.params_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## CLTV Calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cltv_df = calculate_cltv(bg_nbd_model, gamma_gamma_model, summary_data)\n",
    "cltv_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(20, 16))\n",
    "plot_frequency_recency_matrix(bg_nbd_model, summary_data, ax=axes[0, 0])\n",
    "plot_probability_alive_matrix(bg_nbd_model, summary_data, ax=axes[0, 1])\n",
    "plot_cltv_distribution(cltv_df, ax=axes[1, 0])\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
