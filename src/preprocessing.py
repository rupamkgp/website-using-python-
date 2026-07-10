import os
import numpy as np
import pandas as pd

def generate_raw_data(data_dir="data", days=730, seed=42):
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(seed)
    date_range = pd.date_range(start="2023-03-01", periods=days, freq="D")
    dates = date_range.strftime("%Y-%m-%d")

    # 1. Weather Data
    base_temp = 22 + 8 * np.sin(2 * np.pi * date_range.dayofyear / 365)
    temperature = base_temp + np.random.normal(0, 2.5, days)
    humidity = np.clip(65 + 12 * np.sin(2 * np.pi * date_range.dayofyear / 365) + np.random.normal(0, 4, days), 30, 100)
    rainfall = np.clip(3 + 2.5 * np.cos(2 * np.pi * date_range.dayofyear / 365) + np.random.normal(0, 3, days), 0, None)
    wind = np.clip(12 + np.random.normal(0, 3, days), 2, 35)
    
    weather_df = pd.DataFrame({
        "Date": dates,
        "Temperature": np.round(temperature, 1),
        "Humidity": np.round(humidity, 1),
        "Rainfall": np.round(rainfall, 1),
        "Wind": np.round(wind, 1)
    })
    weather_df.to_csv(os.path.join(data_dir, "weather.csv"), index=False)

    # 2. Holiday Data
    weekend = (date_range.dayofweek >= 5).astype(int)
    # Generate random holiday dates
    holiday_prob = 15 / 365
    holiday = np.random.binomial(1, holiday_prob, days)
    # Generate festival flag
    festival = np.zeros(days, dtype=int)
    # Major shopping events (e.g. Black Friday, Cyber Monday, Christmas season, Summer promo)
    for i, date in enumerate(date_range):
        # Q4 Shopping Season (Nov 20 to Dec 25)
        if (date.month == 11 and date.day >= 20) or (date.month == 12 and date.day <= 25):
            festival[i] = 1
        # Summer Sale (July 1 to July 15)
        elif date.month == 7 and date.day <= 15:
            festival[i] = 1
            
    holiday_df = pd.DataFrame({
        "Date": dates,
        "Holiday": holiday,
        "Festival": festival,
        "Weekend": weekend
    })
    holiday_df.to_csv(os.path.join(data_dir, "holidays.csv"), index=False)

    # 3. Discount Data
    discount_active = np.random.binomial(1, 0.22, days)
    discount_pct = np.round(discount_active * np.random.uniform(0.05, 0.35, days), 2)
    coupon_active = np.random.binomial(1, 0.15, days)
    coupon_code = np.where(coupon_active == 1, 1, 0)
    flash_sale = np.where((discount_pct >= 0.20) & (np.random.rand(days) > 0.6), 1, 0)
    
    discount_df = pd.DataFrame({
        "Date": dates,
        "Discount%": discount_pct,
        "Coupon": coupon_code,
        "Flash Sale": flash_sale
    })
    discount_df.to_csv(os.path.join(data_dir, "discount.csv"), index=False)

    # 4. Marketing Data
    # Base spends with channel-specific patterns
    tv_spend = np.clip(np.random.normal(2000, 400, days) * (1 + 0.15 * np.sin(2 * np.pi * date_range.dayofyear / 365)), 0, None)
    google_spend = np.clip(np.random.normal(1500, 300, days) * (1 + 0.25 * (date_range.dayofweek < 5)), 0, None)
    facebook_spend = np.clip(np.random.normal(1000, 200, days) * (1 + 0.35 * (date_range.dayofweek >= 5)), 0, None)
    instagram_spend = np.clip(np.random.normal(800, 180, days) * (1 + 0.45 * (date_range.dayofweek >= 5)), 0, None)
    email_spend = np.where(date_range.dayofweek == 1, np.random.uniform(400, 800, days), 0)
    affiliate_spend = np.clip(np.random.normal(300, 80, days), 0, None)
    
    marketing_df = pd.DataFrame({
        "Date": dates,
        "TV Spend": np.round(tv_spend, 2),
        "Facebook Spend": np.round(facebook_spend, 2),
        "Google Spend": np.round(google_spend, 2),
        "Instagram Spend": np.round(instagram_spend, 2),
        "Email Spend": np.round(email_spend, 2),
        "Affiliate Spend": np.round(affiliate_spend, 2)
    })
    marketing_df.to_csv(os.path.join(data_dir, "marketing.csv"), index=False)

    # 5. Sales Data (Generated with decay adstocks & channel effects to have consistent MMM data)
    def adstock_decay(spend, decay):
        ads = np.zeros(len(spend))
        ads[0] = spend[0]
        for t in range(1, len(spend)):
            ads[t] = spend[t] + decay * ads[t-1]
        return ads

    # Compute adstock variables
    tv_ad = adstock_decay(tv_spend, 0.7)
    g_ad = adstock_decay(google_spend, 0.2)
    fb_ad = adstock_decay(facebook_spend, 0.4)
    ig_ad = adstock_decay(instagram_spend, 0.5)
    em_ad = adstock_decay(email_spend, 0.15)
    aff_ad = adstock_decay(affiliate_spend, 0.1)

    # Hill saturation
    def hill(x, eta, K):
        return (x**eta) / (x**eta + K**eta)

    sat_tv = hill(tv_ad, 1.2, 2500)
    sat_g = hill(g_ad, 1.1, 1800)
    sat_fb = hill(fb_ad, 1.3, 1200)
    sat_ig = hill(ig_ad, 1.0, 900)
    sat_em = hill(em_ad, 1.0, 500)
    sat_aff = hill(aff_ad, 1.0, 400)

    # Base sales trend and seasonality
    weekly = 1.0 + 0.1 * np.cos(2 * np.pi * date_range.dayofweek / 7)
    yearly = 1.0 + 0.18 * np.sin(2 * np.pi * date_range.dayofyear / 365)
    base = 12000 * weekly * yearly

    # Multipliers
    holiday_mult = 1.0 + 0.25 * holiday
    festival_mult = 1.0 + 0.20 * festival
    discount_mult = 1.0 + 0.45 * discount_pct
    weather_mult = 1.0 + 0.006 * (temperature - 20)

    # Sales Equation (Sum of Base and Marketing Contributions)
    # Channel true coefficients: TV: 4500, Google: 8500, Facebook: 3000, Instagram: 1200, Email: 2000, Affiliate: 4500
    # Note: Affiliate has very high ROI in this dataset to show the reallocation potential
    mkt_contrib = (
        4500 * sat_tv +
        8500 * sat_g +
        3000 * sat_fb +
        1200 * sat_ig +
        2000 * sat_em +
        4500 * sat_aff
    )

    revenue = (base * holiday_mult * festival_mult * discount_mult * weather_mult) + mkt_contrib
    revenue = np.clip(revenue + np.random.normal(0, 0.04 * revenue, days), 0, None)
    
    # Visitors and Orders scale with revenue and discount
    visitors = np.clip(np.round(revenue / 5.2 + np.random.normal(0, 150, days)), 500, None)
    orders = np.clip(np.round(visitors * (0.02 + 0.04 * discount_pct) + np.random.normal(0, 10, days)), 10, None)
    
    sales_df = pd.DataFrame({
        "Date": dates,
        "Revenue": np.round(revenue, 2),
        "Orders": orders.astype(int),
        "Visitors": visitors.astype(int)
    })
    sales_df.to_csv(os.path.join(data_dir, "sales.csv"), index=False)

def load_merged_data(data_dir="data"):
    sales = pd.read_csv(os.path.join(data_dir, "sales.csv"))
    weather = pd.read_csv(os.path.join(data_dir, "weather.csv"))
    holidays = pd.read_csv(os.path.join(data_dir, "holidays.csv"))
    discount = pd.read_csv(os.path.join(data_dir, "discount.csv"))
    marketing = pd.read_csv(os.path.join(data_dir, "marketing.csv"))
    
    df = sales.merge(weather, on="Date", how="inner")
    df = df.merge(holidays, on="Date", how="inner")
    df = df.merge(discount, on="Date", how="inner")
    df = df.merge(marketing, on="Date", how="inner")
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    generate_raw_data()
    df = load_merged_data()
    print(f"Data generated and merged successfully. Merged shape: {df.shape}")
