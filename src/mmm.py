import os
import pickle
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

MMM_CHANNELS = ["TV Spend", "Facebook Spend", "Google Spend", "Instagram Spend", "Email Spend", "Affiliate Spend"]
MMM_CONTROLS = ["Holiday", "Temperature", "Discount%", "Weekend", "Rainfall"]

def fit_mmm(df, output_path="mmm_results.pkl"):
    # Normalize data for numerical stability in PyMC
    y_mean = df['Revenue'].mean()
    y_norm = df['Revenue'].values / y_mean
    
    # Spends normalized by max spend
    spends_max = {c: df[c].max() for c in MMM_CHANNELS}
    spends_norm = {c: df[c].values / spends_max[c] for c in MMM_CHANNELS}
    
    # Controls normalized
    controls_mean = {c: df[c].mean() for c in MMM_CONTROLS}
    controls_std = {c: df[c].std() if df[c].std() > 0 else 1.0 for c in MMM_CONTROLS}
    controls_norm = {c: (df[c].values - controls_mean[c]) / controls_std[c] for c in MMM_CONTROLS}
    
    T = len(df)
    t_indices = np.arange(T)
    t_norm = t_indices / T # normalized time for trend
    
    # Fourier terms for seasonality (weekly and annual)
    sin_weekly = np.sin(2 * np.pi * t_indices / 7)
    cos_weekly = np.cos(2 * np.pi * t_indices / 7)
    sin_yearly = np.sin(2 * np.pi * t_indices / 365)
    cos_yearly = np.cos(2 * np.pi * t_indices / 365)
    
    with pm.Model() as model:
        # Intercept and Trend
        beta_0 = pm.Normal('beta_0', mu=1.0, sigma=0.5)
        trend = pm.Normal('trend', mu=0.0, sigma=0.2)
        
        # Seasonality Coefficients
        b_sin_w = pm.Normal('b_sin_w', mu=0.0, sigma=0.2)
        b_cos_w = pm.Normal('b_cos_w', mu=0.0, sigma=0.2)
        b_sin_y = pm.Normal('b_sin_y', mu=0.0, sigma=0.2)
        b_cos_y = pm.Normal('b_cos_y', mu=0.0, sigma=0.2)
        
        # Control variables Coefficients
        gammas = pm.Normal('gammas', mu=0.0, sigma=0.5, shape=len(MMM_CONTROLS))
        
        # Hierarchical hyper-priors for channel contributions
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.5)
        betas = pm.HalfNormal('betas', sigma=sigma_beta, shape=len(MMM_CHANNELS))
        
        # Media parameters (Adstock and Saturation)
        alphas = pm.Beta('alphas', alpha=2.0, beta=2.0, shape=len(MMM_CHANNELS))
        etas = pm.Gamma('etas', alpha=3.0, beta=1.5, shape=len(MMM_CHANNELS))
        Ks = pm.HalfNormal('Ks', sigma=1.0, shape=len(MMM_CHANNELS))
        
        # Compute adstock and saturation for each channel
        media_contributions = []
        for i, c in enumerate(MMM_CHANNELS):
            spend_c = spends_norm[c]
            alpha_c = alphas[i]
            eta_c = etas[i]
            K_c = Ks[i]
            
            # Vectorized geometric adstock decay
            exponents = pt.maximum(0, t_indices[:, None] - t_indices[None, :])
            weights = alpha_c ** exponents
            mask = t_indices[:, None] >= t_indices[None, :]
            weights_masked = weights * mask
            adstock_c = pt.dot(weights_masked, spend_c)
            
            # Hill saturation function
            sat_c = (adstock_c + 1e-5)**eta_c / ((adstock_c + 1e-5)**eta_c + K_c**eta_c)
            media_contributions.append(betas[i] * sat_c)
            
        # Expected sales prediction equation
        control_contribution = sum(gammas[j] * controls_norm[ctrl] for j, ctrl in enumerate(MMM_CONTROLS))
        seasonality = b_sin_w * sin_weekly + b_cos_w * cos_weekly + b_sin_y * sin_yearly + b_cos_y * cos_yearly
        
        mu = beta_0 + trend * t_norm + seasonality + control_contribution + sum(media_contributions)
        
        sigma = pm.HalfNormal('sigma', sigma=0.2)
        
        # Likelihood
        pm.Normal('obs', mu=mu, sigma=sigma, observed=y_norm)
        
        # Run Maximum A Posteriori (MAP) fitting
        map_estimate = pm.find_MAP()
        
    # Extract fitted values
    alpha_fit = map_estimate['alphas']
    eta_fit = map_estimate['etas']
    K_fit = map_estimate['Ks']
    beta_fit = map_estimate['betas'] * y_mean
    beta_0_fit = map_estimate['beta_0'] * y_mean
    trend_fit = map_estimate['trend'] * y_mean
    gammas_fit = map_estimate['gammas'] * y_mean
    
    # Compute channel-specific contributions, ROIs, and adstock curves
    contributions = {}
    rois = {}
    adstocks = {}
    
    for i, c in enumerate(MMM_CHANNELS):
        # Calculate historical adstock
        spend_c = df[c].values / spends_max[c]
        a_c = np.zeros(T)
        a_c[0] = spend_c[0]
        for t in range(1, T):
            a_c[t] = spend_c[t] + alpha_fit[i] * a_c[t-1]
            
        adstocks[c] = a_c.tolist()
        
        # Saturation and contribution
        sat_c = (a_c ** eta_fit[i]) / (a_c ** eta_fit[i] + K_fit[i] ** eta_fit[i])
        contr_c = map_estimate['betas'][i] * sat_c * y_mean
        
        contributions[c] = float(np.sum(contr_c))
        total_spend = df[c].sum()
        rois[c] = float(contributions[c] / total_spend) if total_spend > 0 else 0.0
        
    # Pack up results
    results = {
        'alpha': {c: float(alpha_fit[i]) for i, c in enumerate(MMM_CHANNELS)},
        'eta': {c: float(eta_fit[i]) for i, c in enumerate(MMM_CHANNELS)},
        'K': {c: float(K_fit[i]) for i, c in enumerate(MMM_CHANNELS)},
        'beta': {c: float(beta_fit[i]) for i, c in enumerate(MMM_CHANNELS)},
        'beta_0': float(beta_0_fit),
        'trend': float(trend_fit),
        'gammas': {ctrl: float(gammas_fit[j]) for j, ctrl in enumerate(MMM_CONTROLS)},
        'contributions': contributions,
        'rois': rois,
        'spends_max': spends_max,
        'y_mean': y_mean
    }
    
    # Generate saturation curve grids
    saturation_curves = {}
    for c in MMM_CHANNELS:
        max_sp = spends_max[c]
        spend_grid = np.linspace(0, max_sp * 1.5, 100)
        norm_spend = spend_grid / max_sp
        # steady state adstock: spend / (1 - alpha)
        adstock = norm_spend / (1.0 - results['alpha'][c])
        sat = (adstock ** results['eta'][c]) / (adstock ** results['eta'][c] + results['K'][c] ** results['eta'][c])
        rev_contrib = results['beta'][c] * sat
        saturation_curves[c] = {
            'spend': spend_grid.tolist(),
            'revenue': rev_contrib.tolist()
        }
    results['saturation_curves'] = saturation_curves
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    return results

if __name__ == "__main__":
    from preprocessing import load_merged_data
    df = load_merged_data()
    res = fit_mmm(df)
    print("MMM model fitting completed.")
    print("Estimated Channel ROIs:")
    for c in MMM_CHANNELS:
        print(f"  {c}: ROI={res['rois'][c]:.2f}, Contribution=${res['contributions'][c]:,.2f}")
