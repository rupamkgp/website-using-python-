import os
import pickle
import numpy as np
from scipy.optimize import minimize

CHANNELS = ["TV Spend", "Facebook Spend", "Google Spend", "Instagram Spend", "Email Spend", "Affiliate Spend"]

def optimize_budget(mmm_results, total_budget, min_spends=None, max_spends=None):
    alpha = mmm_results['alpha']
    eta = mmm_results['eta']
    K = mmm_results['K']
    beta = mmm_results['beta']
    spends_max = mmm_results['spends_max']
    
    n_channels = len(CHANNELS)
    
    if min_spends is None:
        min_spends = {c: 0.0 for c in CHANNELS}
    if max_spends is None:
        # Limit max daily spend to 1.8x historical max spend to prevent extrapolation
        max_spends = {c: float(spends_max[c] * 1.8) for c in CHANNELS}
        
    bounds = [(min_spends[c], max_spends[c]) for c in CHANNELS]
    
    def objective_fn(x):
        # We want to maximize sales, which is equivalent to minimizing negative sales contribution
        media_rev = 0
        for i, c in enumerate(CHANNELS):
            norm_spend = x[i] / spends_max[c]
            # Steady state adstock: adstock = spend / (1 - alpha)
            adstock = norm_spend / (1.0 - alpha[c])
            sat = (adstock ** eta[c]) / (adstock ** eta[c] + K[c] ** eta[c])
            media_rev += beta[c] * sat
        return -media_rev
        
    # Budget constraint: sum(spends) = total_budget
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
    
    # Start guess (uniform distribution)
    x0 = np.array([total_budget / n_channels] * n_channels)
    
    # Run optimization
    res = minimize(objective_fn, x0, method='SLSQP', bounds=bounds, constraints=constraint)
    
    optimized_spends = {c: float(res.x[i]) for i, c in enumerate(CHANNELS)}
    optimized_revenue = -res.fun
    
    return optimized_spends, float(optimized_revenue)

def run_optimization_pipeline(mmm_cache_path="mmm_results.pkl", output_path="optimization_results.pkl"):
    with open(mmm_cache_path, 'rb') as f:
        mmm_results = pickle.load(f)
        
    spends_max = mmm_results['spends_max']
    
    # Compute baseline average spends (using the generator logic, which are roughly the same as historical means)
    # We can load the original data to calculate actual average daily spends
    import pandas as pd
    df = pd.read_csv("data/marketing.csv")
    current_avg = {c: float(df[c].mean()) for c in CHANNELS}
    current_total_budget = sum(current_avg.values())
    
    # Compute baseline revenue contribution under current allocation
    baseline_rev = 0
    for c in CHANNELS:
        norm_spend = current_avg[c] / spends_max[c]
        adstock = norm_spend / (1.0 - mmm_results['alpha'][c])
        sat = (adstock ** mmm_results['eta'][c]) / (adstock ** mmm_results['eta'][c] + mmm_results['K'][c] ** mmm_results['eta'][c])
        baseline_rev += mmm_results['beta'][c] * sat
        
    # Scenario 1: Maximize revenue at current total budget
    opt_spends_current, opt_rev_current = optimize_budget(mmm_results, current_total_budget)
    
    # Scenario 2: 55% spend reduction simulation
    # New budget is 45% of current budget (55% reduction)
    reduced_budget = current_total_budget * 0.45
    opt_spends_reduced, opt_rev_reduced = optimize_budget(mmm_results, reduced_budget)
    
    results = {
        'current_avg_spends': current_avg,
        'current_total_budget': current_total_budget,
        'baseline_revenue_contribution': float(baseline_rev),
        
        # Max revenue at full budget
        'opt_spends_current': opt_spends_current,
        'opt_rev_current': opt_rev_current,
        
        # Max revenue at 55% reduced budget
        'reduced_total_budget': reduced_budget,
        'opt_spends_reduced': opt_spends_reduced,
        'opt_rev_reduced': opt_rev_reduced,
        
        # Lift and efficiency metrics
        'spend_saving_pct': 55.0,
        'revenue_retention_pct': float((opt_rev_reduced / baseline_rev) * 100)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    return results

if __name__ == "__main__":
    res = run_optimization_pipeline()
    print("Optimization pipeline completed successfully.")
    print(f"Current Daily Spend: ${res['current_total_budget']:,.2f}")
    print(f"Current Media Revenue Contribution: ${res['baseline_revenue_contribution']:,.2f}")
    print("\n--- Optimized Budget Allocation (Same Budget) ---")
    for c in CHANNELS:
        print(f"  {c}: Current=${res['current_avg_spends'][c]:,.2f} -> Opt=${res['opt_spends_current'][c]:,.2f}")
    print(f"Optimized Media Revenue: ${res['opt_rev_current']:,.2f} (Lift: {((res['opt_rev_current'] - res['baseline_revenue_contribution'])/res['baseline_revenue_contribution'])*100:.2f}%)")
    
    print(f"\n--- 55% Spend Reduction Allocation ---")
    print(f"Reduced Daily Spend: ${res['reduced_total_budget']:,.2f}")
    for c in CHANNELS:
        print(f"  {c}: Current=${res['current_avg_spends'][c]:,.2f} -> Opt=${res['opt_spends_reduced'][c]:,.2f}")
    print(f"Optimized Media Revenue (Reduced Budget): ${res['opt_rev_reduced']:,.2f} ({res['revenue_retention_pct']:.2f}% of baseline revenue retained!)")
