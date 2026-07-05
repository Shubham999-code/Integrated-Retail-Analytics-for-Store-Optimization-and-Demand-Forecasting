# Integrated Retail Analytics for Store Optimization and Demand Forecasting

This repository contains an end-to-end Machine Learning and Data Analysis project aimed at optimizing retail store performance, forecasting demand, and enhancing customer experiences through personalized strategies.

## Project Objective
To utilize machine learning and data analysis techniques to optimize store performance, forecast demand, and enhance customer experience through segmentation and personalized marketing strategies based on historical sales and macroeconomic data.

## Features

1. **Data Preprocessing & Feature Engineering**: Handles missing markdown data and extracts temporal features for time-series analysis.
2. **Anomaly Detection**: Uses `IsolationForest` to identify massive sales spikes (holidays) that deviate from normal weekly trends.
3. **Customer Segmentation**: Employs `K-Means Clustering` on store profiles (sales volume, store size, markdown responsiveness) to create distinct store tiers. Evaluated using the Silhouette score.
4. **Market Basket Analysis**: Infers product associations by analyzing the correlation matrix of weekly sales between different departments to identify cross-selling opportunities.
5. **Demand Forecasting**: Uses `RandomForestRegressor` to predict future weekly sales based on internal features and external economic factors (CPI, Unemployment, Fuel Prices, Temperature).

## Repository Structure

- `Retail_Analytics_Project.ipynb`: The main Google Colab / Jupyter Notebook containing all data preprocessing, EDA, ML models, and evaluation metrics.
- `Project_Documentation.md`: Detailed documentation explaining the methodology, insights, and real-world strategies derived from the models.
- `generate_colab.py`: Python script used to automatically generate the Notebook structure (can be ignored for standard usage).
- Data files (`sales data-set.csv`, `Features data set.csv`, `stores data-set.csv`) should be placed in the root directory to run the notebook.

## Setup and Usage

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   ```
2. **Upload to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload the `Retail_Analytics_Project.ipynb` file.
   - Upload the three CSV datasets to the Colab session storage.
3. **Run the Notebook:**
   - Execute the cells sequentially to perform the analysis, view data visualizations, train the models, and read the strategic insights.

## Strategic Highlights
- **Inventory Management**: Dynamic scaling prior to detected anomalies (Thanksgiving, Christmas) to prevent stock-outs.
- **Personalized Marketing**: Targeted promotions based on the K-Means cluster of the store (e.g., highly price-sensitive segments vs. high-volume general segments).
- **Store Optimization**: Placing highly correlated departments physically closer in-store to encourage spontaneous cross-selling.

## Technologies Used
- **Python 3**
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine Learning (IsolationForest, KMeans, RandomForest)
