import pandas as pd
import numpy as np

def run_analytics():
    master_path = 'data/processed/master_etf_macro.csv'
    if not os.path.exists(master_path):
        print(f"Error: {master_path} not found. Please run 'python src/ingest.py' first.")
        return

    print("Loading master dataset...")
    df = pd.read_csv(master_path, index_col=0, parse_dates=True)

    # Define ETF asset columns vs Macro columns
    etf_tickers = [
        'NDQ.AX', 'HACK.AX', 'IXJ.AX', 'GOLD.AX', 
        'VAS.AX', 'VGS.AX', 'VGAD.AX', 'VAF.AX', 
        'AAA.AX', 'QRE.AX', 'VAP.AX', 'VHY.AX'
    ]
    
    macro_cols = [
        'cash_rate_target', 'cpi_inflation', 
        'fed_funds_rate', 'us_cpi', 'audusd_fx'
    ]

    # Ensure only existing columns are analyzed
    active_etfs = [t for t in etf_tickers if t in df.columns]
    active_macros = [m for m in macro_cols if m in df.columns]

    # Calculate daily returns for the ETFs
    df_prices = df[active_etfs]
    df_returns = df_prices.pct_change().dropna()

    # Equal-weight portfolio daily returns
    portfolio_returns = df_returns.mean(axis=1)
    df['Portfolio_Return'] = portfolio_returns

    # 1. Core Portfolio Risk & Performance Metrics
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Use latest daily RBA cash rate as risk-free rate benchmark
    latest_rba_rate = df['cash_rate_target'].iloc[-1] / 100.0
    sharpe_ratio = (ann_return - latest_rba_rate) / ann_vol if ann_vol > 0 else 0.0

    # Maximum Drawdown Calculation
    cum_returns = (1 + portfolio_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    print("\n" + "="*50)
    print("--- Equal-Weight Portfolio Risk Metrics (2018–2026) ---")
    print(f"Annualized Return: {ann_return * 100:.2f}%")
    print(f"Annualized Volatility: {ann_vol * 100:.2f}%")
    print(f"Sharpe Ratio (vs RBA Cash Rate): {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
    print("="*50)

    # 2. Expanded Macro Sensitivities & Correlations
    print("\n--- Extended Macro Sensitivities (AU & US Drivers) ---")
    macro_corr_data = pd.DataFrame({'Portfolio': portfolio_returns})
    
    for col in active_macros:
        # Align daily returns with daily macro levels (or changes)
        macro_corr_data[col] = df[col].reindex(portfolio_returns.index)

    corr_matrix = macro_corr_data.corr()
    print(corr_matrix.to_string())
    print("="*50)

if __name__ == "__main__":
    import os
    run_analytics()