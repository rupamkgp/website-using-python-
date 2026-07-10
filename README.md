# Budget Optimization & Marketing Mix Modeling (MMM) Suite

An end-to-end data science framework to forecast revenue, estimate media channel contributions, evaluate marketing ROI, and run budget allocation optimization using PyMC and Streamlit.

## Project Overview & Business Problem

Marketing departments often deploy substantial budgets across multiple media channels. Understanding how much revenue each channel generates—while accounting for the delayed response of advertising (Adstock) and diminishing marginal returns (Saturation)—is a fundamental challenge. 

This project solves this problem by:
1.  **Forecasting Revenue**: Predicting future revenue trends to establish budget baselines.
2.  **Attribution Modeling (MMM)**: Quantifying the ROI of each marketing channel.
3.  **Budget Optimization**: Reallocating budget to maximize revenue and simulating a **55% spend reduction** scenario with minimal impact on sales.

---

## Project Architecture

```
Budget-Optimisation-MMM/
│
├── data/                       # Multi-source raw datasets
│    ├── sales.csv
│    ├── weather.csv
│    ├── holidays.csv
│    ├── discount.csv
│    └── marketing.csv
│
├── notebooks/                  # Step-by-step Jupyter Notebook workflows
│    ├── 01_EDA.ipynb
│    ├── 02_Feature_Engineering.ipynb
│    ├── 03_TimeSeries_Forecasting.ipynb
│    ├── 04_Bayesian_MMM.ipynb
│    └── 05_Budget_Optimization.ipynb
│
├── src/                        # Modular pipeline components
│    ├── preprocessing.py
│    ├── feature_engineering.py
│    ├── forecasting.py
│    ├── mmm.py
│    └── optimizer.py
│
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Dataset Schema

The raw dataset spans **730 days (2 years)** across five distinct files merged on `Date`:

*   **`sales.csv`**: `Date`, `Revenue`, `Orders`, `Visitors`
*   **`marketing.csv`**: `Date`, `TV Spend`, `Facebook Spend`, `Google Spend`, `Instagram Spend`, `Email Spend`, `Affiliate Spend`
*   **`weather.csv`**: `Date`, `Temperature`, `Humidity`, `Rainfall`, `Wind`
*   **`holiday.csv`**: `Date`, `Holiday`, `Festival`, `Weekend`
*   **`discount.csv`**: `Date`, `Discount%`, `Coupon`, `Flash Sale`

---

## Feature Engineering (35+ Features)

We derive **40 engineered features** to feed our forecasting models:
*   **Time (10 features)**: Year, Month, Week, Quarter, Day, Weekday, Weekend, Month Start, Month End, Day of Year.
*   **Rolling (5 features)**: Moving averages and standard deviations of Revenue, Orders, and Visitors.
*   **Lag (5 features)**: Prior day sales, 7d/14d/30d revenue lags, and spend lags.
*   **Marketing (5 features)**: Budget share percentages per channel.
*   **Weather (5 features)**: Temperature, humidity, rainfall, Heat Index, Rain Category.
*   **Holiday (5 features)**: Holiday/Festival flags, days before/after holiday.
*   **Discount (5 features)**: Coupon flags, Offer Intensity index, and discount magnitude categories.

---

## Forecasting Model Evaluation

We compare 6 forecasting models on a 20% out-of-sample chronological test split:

| Model | MAE | RMSE | MAPE | R² |
| :--- | :--- | :--- | :--- | :--- |
| **ARIMA** | 1,842.10 | 2,120.45 | 11.8% | -1.301 |
| **SARIMA** | 1,698.40 | 1,940.32 | 10.8% | -0.817 |
| **Prophet** | 768.45 | 920.12 | 4.9% | 0.630 |
| **XGBoost** | 740.15 | 884.23 | 4.7% | 0.687 |
| **LightGBM** | 682.40 | 818.10 | 4.2% | 0.730 |
| **LSTM (PyTorch)** | 854.20 | 1,012.30 | 5.4% | 0.543 |

*LightGBM achieved the best forecasting performance (MAPE = 4.2%, R² = 0.73).*

---

## Bayesian MMM Methodology

We implement a custom Bayesian Marketing Mix Model in **PyMC** with:
*   **Adstock decay**: Geometric carryover rate ($\alpha_c$).
*   **Saturation curve**: Hill Function modeling diminishing returns.
*   **Hierarchical Priors**: Channel coefficients ($\beta_c$) are modeled with a shared half-normal hyper-prior distribution to stabilize coefficients.
*   **Controls**: Includes weather parameters, holiday impacts, weekends, and promotional discounts.

### Channel ROI Estimates
*   **Email Spend**: ROI = 7.27
*   **Affiliate Spend**: ROI = 2.85
*   **Google Spend**: ROI = 1.13
*   **Facebook Spend**: ROI = 0.92
*   **TV Spend**: ROI = 0.65
*   **Instagram Spend**: ROI = 0.15

---

## Budget Reallocation Optimization (55% Spend Reduction)

Using `scipy.optimize.minimize` (SLSQP), we optimize budget allocations.

### 55% Budget Cut Simulation
*   **Average Historical Spend**: \$6,133.01 / day
*   **Slashed Target Spend**: \$2,759.85 / day (55% saving)
*   **Baseline Contribution**: \$6,636.80
*   **Optimized Contribution under Slashed Budget**: \$4,697.22
*   **Result**: We retain **70.8% of media-driven revenue** (and **~90% of total revenue**) while **slashing marketing spend by 55%** by completely defunding low-ROI channels (TV, Instagram) and shifting resources to high-ROI channels (Email, Affiliate, Google).

---

## Streamlit Dashboard Pages

Run the dashboard to interactively visualize:
*   **Home**: Dashboard summaries and KPI cards.
*   **Revenue Analysis**: Historical revenue trend, orders, visitors, and weather/holiday correlations.
*   **Spend Analysis**: Stacked marketing spends timeline and donut chart shares.
*   **Revenue Forecast**: Compare 6 models and toggle forecast lines.
*   **Bayesian MMM**: Visual ROIs, contribution shares, decay rates, and Hill curves.
*   **Budget Optimizer**: Interactive budget sliders and min/max limits, featuring a **Simulate 55% Budget Slashing** preset.

---

## Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/budget-optimization-mmm.git
    cd budget-optimization-mmm
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run backend pipeline**:
    ```bash
    python src/preprocessing.py
    python src/forecasting.py
    python src/mmm.py
    python src/optimizer.py
    ```

4.  **Launch the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
