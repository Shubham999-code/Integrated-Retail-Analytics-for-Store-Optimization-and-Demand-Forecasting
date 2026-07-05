# Integrated Retail Analytics for Store Optimization and Demand Forecasting

## 1. Project Overview

**Objective:**
To utilize machine learning and data analysis techniques to optimize store performance, forecast demand, and enhance customer experience through segmentation and personalized marketing strategies.

This document details the methodologies, machine learning models, and strategic insights generated from analyzing retail sales data (including features like MarkDowns, CPI, Unemployment, and Temperature).

---

## 2. Data Analysis and Preprocessing

**Methodology:**
1. **Data Integration:** The `sales data-set.csv`, `Features data set.csv`, and `stores data-set.csv` were merged using `Store`, `Date`, and `IsHoliday` as common keys to form a unified dataset.
2. **Missing Values Handling:**
   - Promotional `MarkDown` features contained significant missing values, which were imputed with `0` under the assumption that a missing value indicates no promotional markdown occurred on that date.
   - `CPI` (Consumer Price Index) and `Unemployment` rate missing values were handled using forward-fill/backward-fill to maintain temporal continuity.
3. **Feature Engineering:**
   - The `Date` string was parsed into a datetime object to extract temporal features such as `Year`, `Month`, and `Week`. These are critical for time-series forecasting and seasonal analysis.
   - The categorical `Store Type` (A, B, C) was encoded numerically (3, 2, 1) to represent the tier/scale of the store for downstream clustering and regression models.

---

## 3. Anomaly Detection

**Methodology:**
- **Time-Series Aggregation:** Sales data was aggregated at a weekly level across all stores to identify macro-level trends.
- **Model:** An `Isolation Forest` (with a 5% contamination rate) was employed to detect statistical anomalies in the weekly sales distribution.

**Insights:**
- The model successfully flagged specific weeks as anomalies. Upon review, these anomalies perfectly align with the massive sales spikes observed during the Thanksgiving and Christmas holiday weeks. 
- **Impact:** By formally identifying these anomalies, we avoid treating these spikes as "errors," and instead recognize them as highly predictable recurring seasonal events that require distinct inventory management strategies.

---

## 4. Customer Segmentation Analysis

**Methodology:**
- **Store Profiling:** Stores were aggregated by calculating their mean Weekly Sales, size, and their average reliance on MarkDowns.
- **Scaling:** Data was standardized using a `StandardScaler`.
- **Model:** `K-Means Clustering` (k=3) was applied to group stores into distinct segments.
- **Evaluation:** The clustering quality was evaluated using the `Silhouette Score`, ensuring that the stores within each cluster are homogenous and well-separated from other clusters.

**Insights (Store Tiers):**
- **Cluster 0:** Represents massive-scale stores with very high baseline sales volume.
- **Cluster 1:** Represents mid-sized stores that exhibit a high responsiveness to promotional markdowns.
- **Cluster 2:** Represents small-scale stores with lower overall volume.

---

## 5. Market Basket Analysis (Department Correlation)

**Methodology:**
Since individual customer receipt/transaction data is not available, product associations were inferred by examining how weekly sales of different departments fluctuate together across stores over time.
- A correlation matrix was computed on the weekly sales of the top 20 revenue-generating departments.

**Insights:**
- Strong positive correlations were found between specific departments, suggesting that their demand is driven by similar customer footfall or seasonal trends.
- **Cross-Selling Strategy:** By physically placing highly correlated departments closer to one another in-store, we can capitalize on associated purchasing behaviors (e.g., placing Holiday Goods near Grocery items during Q4).

---

## 6. Demand Forecasting and External Factors

**Methodology:**
- **Model:** A `Random Forest Regressor` was built to forecast `Weekly_Sales` based on both internal features (Store, Dept, Size, MarkDowns) and external macroeconomic factors.
- **Validation:** The dataset was split temporally/randomly to train the model, which was then evaluated using the Root Mean Squared Error (RMSE) metric.

**Impact of External Factors:**
- An analysis of the model's `feature_importances_` revealed the weight of different predictors.
- Internal attributes (`Dept`, `Size`) and temporal indicators (`Week`) hold the highest predictive power, establishing a strong seasonal baseline.
- **External factors** like `CPI` and `Temperature` play a smaller but statistically significant role in modifying the baseline demand, allowing the model to dynamically adjust to changing economic and regional climates.

---

## 7. Real-World Application and Strategy Formulation

Based on the machine learning outputs, the following comprehensive retail strategy is formulated:

1. **Inventory Management Strategy (Anomaly-Driven):**
   - The Demand Forecasting model and Anomaly Detection both point to massive holiday-driven demand spikes. Inventory procurement must dynamically scale up 2 to 3 weeks prior to Thanksgiving and Christmas to prevent costly stock-outs on high-margin items.

2. **Personalized Marketing Strategy (Segmentation-Driven):**
   - **Cluster 1 (Price-Sensitive):** These stores respond aggressively to markdowns. Marketing spend here should focus on highly personalized couponing and targeted digital promotions.
   - **Cluster 0 & 2:** These stores are driven by baseline size and community engagement, respectively. Focus on broad advertising for massive stores, and optimizing essential local inventory for the smaller stores rather than burning margin on heavy markdowns.

3. **Store Optimization Strategy (Market Basket-Driven):**
   - The correlation insights suggest an optimized floor plan. Correlated departments should share adjacent real estate or end-cap displays to encourage spontaneous cross-category purchasing.

### Practical Challenges & Optimization
- **Noisy Data:** Missing MarkDown data required assumptions (imputation). Real-world implementations require strict data governance at the point-of-sale to capture accurate promotional data.
- **Economic Shocks:** Models heavily reliant on external factors (CPI) can struggle during sudden, unprecedented economic events (like a pandemic). Implementing a robust MLOps pipeline to continually retrain models on recent data is necessary for long-term reliability.
