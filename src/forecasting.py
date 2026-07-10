import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch LSTM Definition
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

def prepare_lstm_data(X, y, seq_len=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:(i + seq_len)])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

def train_lstm(X_train, y_train, input_dim, epochs=30, seq_len=7):
    model = LSTMRegressor(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Scale variables
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_seq, y_seq = prepare_lstm_data(X_train_scaled, y_train_scaled, seq_len=seq_len)
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
    return model, scaler_X, scaler_y

def predict_lstm(model, X, scaler_X, scaler_y, seq_len=7):
    model.eval()
    X_scaled = scaler_X.transform(X)
    X_seq = []
    # To predict the sequence, we need padding or we slide over windows.
    # For prediction of test set, we construct sequential inputs.
    for i in range(len(X)):
        if i < seq_len:
            # Pad with first values for initial step
            pad = np.repeat(X_scaled[0:1], seq_len - i, axis=0)
            window = np.vstack([pad, X_scaled[0:i]])
        else:
            window = X_scaled[(i - seq_len):i]
        X_seq.append(window)
        
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()
        
    return scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

def run_forecasting_pipeline(df_feat, output_cache="forecast_results.pkl"):
    # Target and Features
    target = 'Revenue'
    feature_cols = [
        'Year', 'Month', 'Week', 'Quarter', 'Day', 'Weekday', 'Weekend_time',
        'Month_Start', 'Month_End', 'Day_of_Year', 'Rev_Roll_Mean_7',
        'Rev_Roll_Mean_30', 'Rev_Roll_Std_7', 'Visitors_Roll_7', 'Orders_Roll_7',
        'Rev_Lag_1', 'Rev_Lag_7', 'Rev_Lag_14', 'Rev_Lag_30', 'Google_Spend_Lag_1',
        'TV_Share', 'Google_Share', 'Facebook_Share', 'Instagram_Share', 'Email_Share',
        'Heat_Index', 'Rain_Category', 'Holiday_Before', 'Holiday_After', 'Offer_Intensity',
        'Discount_Category', 'Temperature', 'Humidity', 'Rainfall', 'Holiday', 'Weekend'
    ]
    
    # Split chronologically
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols].values, train_df[target].values
    X_test, y_test = test_df[feature_cols].values, test_df[target].values
    
    predictions = {}
    
    # 1. ARIMA (univariate)
    arima = ARIMA(y_train, order=(2, 1, 1))
    arima_fit = arima.fit()
    predictions['ARIMA'] = arima_fit.forecast(steps=len(test_df))
    
    # 2. SARIMA (univariate weekly)
    sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
    sarima_fit = sarima.fit(disp=False)
    predictions['SARIMA'] = sarima_fit.forecast(steps=len(test_df))
    
    # 3. Prophet
    prophet_train = train_df[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'})
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet.fit(prophet_train)
    future_test = pd.DataFrame({'ds': test_df['Date']})
    prophet_forecast = prophet.predict(future_test)
    predictions['Prophet'] = prophet_forecast['yhat'].values
    
    # 4. XGBoost
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_reg.fit(X_train, y_train)
    predictions['XGBoost'] = xgb_reg.predict(X_test)
    
    # 5. LightGBM
    lgb_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_reg.fit(X_train, y_train)
    predictions['LightGBM'] = lgb_reg.predict(X_test)
    
    # 6. LSTM
    lstm_model, scaler_X, scaler_y = train_lstm(X_train, y_train, len(feature_cols))
    predictions['LSTM'] = predict_lstm(lstm_model, X_test, scaler_X, scaler_y)
    
    # Evaluate metrics
    metrics = {}
    for name, preds in predictions.items():
        preds = np.clip(preds, 0, None)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics[name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2)
        }
        
    # Generate 1-year future forecast
    # We retrain the best model (typically Prophet or XGBoost for stable projection, let's output future forecast for all models)
    # Fit ARIMA & SARIMA on full data
    arima_full = ARIMA(df_feat[target].values, order=(2, 1, 1)).fit()
    sarima_full = SARIMAX(df_feat[target].values, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
    
    prophet_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_full.fit(df_feat[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'}))
    
    # ML Models full fit
    X_full, y_full = df_feat[feature_cols].values, df_feat[target].values
    xgb_full = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42).fit(X_full, y_full)
    lgb_full = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1).fit(X_full, y_full)
    lstm_full, full_scaler_X, full_scaler_y = train_lstm(X_full, y_full, len(feature_cols))
    
    # Future dates (365 days ahead)
    last_date = df_feat['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')
    
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Week'] = future_df['Date'].dt.isocalendar().week.astype(int)
    future_df['Quarter'] = future_df['Date'].dt.quarter
    future_df['Day'] = future_df['Date'].dt.day
    future_df['Weekday'] = future_df['Date'].dt.dayofweek
    future_df['Weekend_time'] = (future_df['Weekday'] >= 5).astype(int)
    future_df['Month_Start'] = future_df['Date'].dt.is_month_start.astype(int)
    future_df['Month_End'] = future_df['Date'].dt.is_month_end.astype(int)
    future_df['Day_of_Year'] = future_df['Date'].dt.dayofyear
    
    # Group historical data by day of year to get seasonal average features
    hist_day_means = df_feat.groupby('Day_of_Year')[['Temperature', 'Humidity', 'Rainfall', 'Holiday', 'Weekend', 'TV_Share', 'Google_Share', 'Facebook_Share', 'Instagram_Share', 'Email_Share', 'Discount%', 'Coupon', 'Flash Sale']].mean().reset_index()
    future_df = future_df.merge(hist_day_means, on='Day_of_Year', how='left')
    
    future_df['Holiday'] = (future_df['Holiday'] > 0.05).astype(int)
    future_df['Weekend'] = (future_df['Weekend'] > 0.5).astype(int)
    future_df['Coupon'] = (future_df['Coupon'] > 0.1).astype(int)
    future_df['Flash Sale'] = (future_df['Flash Sale'] > 0.1).astype(int)
    
    # Calculate days before/after holiday
    fut_holiday_indices = future_df[future_df['Holiday'] == 1].index.values
    fut_days_before = np.zeros(len(future_df))
    fut_days_after = np.zeros(len(future_df))
    for i in range(len(future_df)):
        if len(fut_holiday_indices) == 0:
            fut_days_before[i] = 15
            fut_days_after[i] = 15
            continue
        diffs = fut_holiday_indices - i
        before_diffs = diffs[diffs >= 0]
        after_diffs = diffs[diffs <= 0]
        fut_days_before[i] = before_diffs[0] if len(before_diffs) > 0 else 15
        fut_days_after[i] = abs(after_diffs[-1]) if len(after_diffs) > 0 else 15
    future_df['Holiday_Before'] = np.clip(fut_days_before, 0, 15)
    future_df['Holiday_After'] = np.clip(fut_days_after, 0, 15)
    
    future_df['Offer_Intensity'] = future_df['Discount%'] * (1 + future_df['Coupon'] + future_df['Flash Sale'])
    
    def get_discount_category(disc):
        if disc == 0:
            return 0
        elif disc < 0.15:
            return 1
        else:
            return 2
    future_df['Discount_Category'] = future_df['Discount%'].apply(get_discount_category)
    
    future_df['Heat_Index'] = future_df['Temperature'] + 0.15 * future_df['Humidity']
    
    def get_rain_category(rain):
        if rain == 0:
            return 0
        elif rain <= 2.0:
            return 1
        else:
            return 2
    future_df['Rain_Category'] = future_df['Rainfall'].apply(get_rain_category)
    
    # Lags and Rolling values recursively generated
    future_xgb = []
    future_lgb = []
    future_lstm = []
    
    xgb_buffer = list(df_feat[target].values[-30:])
    lgb_buffer = list(df_feat[target].values[-30:])
    lstm_buffer = list(df_feat[target].values[-30:])
    
    # Visitors/Orders roll mean for helper features
    vis_buffer = list(df_feat['Visitors'].values[-7:])
    ord_buffer = list(df_feat['Orders'].values[-7:])
    
    # Spend lag helper
    g_spend_buffer = list(df_feat['Google Spend'].values[-1:])
    
    for i in range(len(future_df)):
        row = future_df.iloc[[i]].copy()
        
        def fill_row_features(row_df, buffer):
            row_df['Rev_Roll_Mean_7'] = np.mean(buffer[-7:])
            row_df['Rev_Roll_Mean_30'] = np.mean(buffer[-30:])
            row_df['Rev_Roll_Std_7'] = np.std(buffer[-7:])
            row_df['Visitors_Roll_7'] = np.mean(vis_buffer[-7:])
            row_df['Orders_Roll_7'] = np.mean(ord_buffer[-7:])
            
            row_df['Rev_Lag_1'] = buffer[-1]
            row_df['Rev_Lag_7'] = buffer[-7]
            row_df['Rev_Lag_14'] = buffer[-14]
            row_df['Rev_Lag_30'] = buffer[-30]
            row_df['Google_Spend_Lag_1'] = g_spend_buffer[-1]
            return row_df
            
        # Predict XGBoost
        row_xgb = fill_row_features(row.copy(), xgb_buffer)
        p_xgb = max(0, float(xgb_full.predict(row_xgb[feature_cols].values)[0]))
        future_xgb.append(p_xgb)
        xgb_buffer.append(p_xgb)
        
        # Predict LightGBM
        row_lgb = fill_row_features(row.copy(), lgb_buffer)
        p_lgb = max(0, float(lgb_full.predict(row_lgb[feature_cols].values)[0]))
        future_lgb.append(p_lgb)
        lgb_buffer.append(p_lgb)
        
        # Predict LSTM
        row_lstm = fill_row_features(row.copy(), lstm_buffer)
        p_lstm = max(0, float(predict_lstm(lstm_full, row_lstm[feature_cols].values, full_scaler_X, full_scaler_y)[0]))
        future_lstm.append(p_lstm)
        lstm_buffer.append(p_lstm)
        
        # Update extra buffers
        # Assume future visitors and orders scale with predicted revenue
        pred_rev = p_xgb
        pred_vis = pred_rev / 5.2
        pred_ord = pred_vis * (0.02 + 0.04 * row['Discount%'].values[0])
        vis_buffer.append(pred_vis)
        ord_buffer.append(pred_ord)
        # Spend remains seasonal average
        g_spend_buffer.append(row['Google_Share'].values[0] * 1000) # placeholder scale
        
    # ARIMA & SARIMA forecast
    future_arima = arima_full.forecast(steps=365)
    future_sarima = sarima_full.forecast(steps=365)
    
    # Prophet forecast
    prophet_fut_dates = prophet_full.make_future_dataframe(periods=365, freq='D')
    prophet_full_forecast = prophet_full.predict(prophet_fut_dates)
    future_prophet = prophet_full_forecast['yhat'].values[-365:]
    
    future_forecasts = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'ARIMA': np.round(np.clip(future_arima, 0, None), 2),
        'SARIMA': np.round(np.clip(future_sarima, 0, None), 2),
        'Prophet': np.round(np.clip(future_prophet, 0, None), 2),
        'XGBoost': np.round(future_xgb, 2),
        'LightGBM': np.round(future_lgb, 2),
        'LSTM': np.round(future_lstm, 2)
    })
    
    results = {
        'metrics': metrics,
        'forecasts': future_forecasts.to_dict(orient='list')
    }
    
    with open(output_cache, 'wb') as f:
        pickle.dump(results, f)
        
    return results

if __name__ == "__main__":
    from preprocessing import load_merged_data
    from feature_engineering import engineer_all_features
    
    df = load_merged_data()
    df_feat = engineer_all_features(df)
    res = run_forecasting_pipeline(df_feat)
    print("Forecasting pipeline completed successfully.")
    print("Test set metrics:")
    for m, vals in res['metrics'].items():
        print(f"  {m}: R2={vals['R2']:.3f}, MAPE={vals['MAPE']:.3f}")
