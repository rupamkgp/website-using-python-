import numpy as np
import pandas as pd

def add_time_features(df):
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek
    df['Weekend_time'] = (df['Weekday'] >= 5).astype(int)
    df['Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df['Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    return df

def add_rolling_features(df):
    df = df.copy()
    # Shifting by 1 to prevent data leakage in forecasting
    df['Rev_Roll_Mean_7'] = df['Revenue'].shift(1).rolling(window=7).mean()
    df['Rev_Roll_Mean_30'] = df['Revenue'].shift(1).rolling(window=30).mean()
    df['Rev_Roll_Std_7'] = df['Revenue'].shift(1).rolling(window=7).std()
    df['Visitors_Roll_7'] = df['Visitors'].shift(1).rolling(window=7).mean()
    df['Orders_Roll_7'] = df['Orders'].shift(1).rolling(window=7).mean()
    return df

def add_lag_features(df):
    df = df.copy()
    df['Rev_Lag_1'] = df['Revenue'].shift(1)
    df['Rev_Lag_7'] = df['Revenue'].shift(7)
    df['Rev_Lag_14'] = df['Revenue'].shift(14)
    df['Rev_Lag_30'] = df['Revenue'].shift(30)
    df['Google_Spend_Lag_1'] = df['Google Spend'].shift(1)
    return df

def add_marketing_features(df):
    df = df.copy()
    spends = ['TV Spend', 'Facebook Spend', 'Google Spend', 'Instagram Spend', 'Email Spend', 'Affiliate Spend']
    total_spend = df[spends].sum(axis=1)
    # Avoid division by zero
    total_spend_safe = np.where(total_spend == 0, 1.0, total_spend)
    
    df['TV_Share'] = df['TV Spend'] / total_spend_safe
    df['Google_Share'] = df['Google Spend'] / total_spend_safe
    df['Facebook_Share'] = df['Facebook Spend'] / total_spend_safe
    df['Instagram_Share'] = df['Instagram Spend'] / total_spend_safe
    df['Email_Share'] = df['Email Spend'] / total_spend_safe
    return df

def add_weather_features(df):
    df = df.copy()
    # Simple heat index approximation
    df['Heat_Index'] = df['Temperature'] + 0.15 * df['Humidity']
    
    # Categorize rainfall
    def get_rain_category(rain):
        if rain == 0:
            return 0
        elif rain <= 2.0:
            return 1
        else:
            return 2
            
    df['Rain_Category'] = df['Rainfall'].apply(get_rain_category)
    return df

def add_holiday_features(df):
    df = df.copy()
    # Compute days before/after holiday
    holiday_indices = df[df['Holiday'] == 1].index.values
    days_before = np.zeros(len(df))
    days_after = np.zeros(len(df))
    
    for i in range(len(df)):
        if len(holiday_indices) == 0:
            days_before[i] = 15
            days_after[i] = 15
            continue
            
        diffs = holiday_indices - i
        before_diffs = diffs[diffs >= 0]
        after_diffs = diffs[diffs <= 0]
        
        days_before[i] = before_diffs[0] if len(before_diffs) > 0 else 15
        days_after[i] = abs(after_diffs[-1]) if len(after_diffs) > 0 else 15
        
    df['Holiday_Before'] = np.clip(days_before, 0, 15)
    df['Holiday_After'] = np.clip(days_after, 0, 15)
    return df

def add_discount_features(df):
    df = df.copy()
    df['Offer_Intensity'] = df['Discount%'] * (1 + df['Coupon'] + df['Flash Sale'])
    
    # Categorize discount levels
    def get_discount_category(disc):
        if disc == 0:
            return 0
        elif disc < 0.15:
            return 1
        else:
            return 2
            
    df['Discount_Category'] = df['Discount%'].apply(get_discount_category)
    return df

def engineer_all_features(df):
    df = df.copy()
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_marketing_features(df)
    df = add_weather_features(df)
    df = add_holiday_features(df)
    df = add_discount_features(df)
    
    # Fill any NaNs created by shifts/rolling windows
    df = df.bfill().ffill()
    return df

if __name__ == "__main__":
    from preprocessing import load_merged_data
    df_raw = load_merged_data()
    df_feat = engineer_all_features(df_raw)
    print(f"Feature engineering completed. Total columns: {df_feat.shape[1]}")
    # Print list of engineered columns
    orig_cols = set(df_raw.columns)
    eng_cols = [c for c in df_feat.columns if c not in orig_cols]
    print(f"Engineered {len(eng_cols)} features: {eng_cols}")
