import os
import pandas as pd
import yfinance as yf

TICKERS = [
    'NDQ.AX', 'HACK.AX', 'IXJ.AX', 'GOLD.AX', 
    'VAS.AX', 'VGS.AX', 'VGAD.AX', 'VAF.AX', 
    'AAA.AX', 'QRE.AX', 'VAP.AX', 'VHY.AX'
]

START_DATE = '2018-01-01'

def load_fx_csv(filepath):
    """Safely loads audusd.csv whether it has standard 1-row or yfinance 3-row headers."""
    with open(filepath, 'r') as f:
        first_line = f.readline()
    
    if first_line.startswith('Price,'):
        # Handle raw 3-row yfinance MultiIndex CSV
        df = pd.read_csv(filepath, skiprows=[1, 2], index_col=0, parse_dates=True)
        df.index.name = 'Date'
    else:
        # Handle standard single-header CSV
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
    col = 'Close' if 'Close' in df.columns else df.columns[0]
    return df[[col]].rename(columns={col: 'audusd_fx'})

def run_ingestion():
    print("Verifying directory structure...")
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 1. Load or download ETF prices
    raw_etf_path = 'data/raw/etf_prices_raw.csv'
    if os.path.exists(raw_etf_path):
        print(f"Loading cached ETF prices from {raw_etf_path}...")
        df_prices = pd.read_csv(raw_etf_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading historical adjusted close prices for {len(TICKERS)} ETFs...")
        df_raw = yf.download(TICKERS, start=START_DATE, auto_adjust=True, progress=False)
        df_prices = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[['Close']]
        df_prices.to_csv(raw_etf_path)

    df_prices = df_prices.loc[START_DATE:].dropna(how='all').ffill()

    # 2. Load local US Macro & FX files
    print("Reading local US macro (FEDFUNDS, CPIAUCSL) and AUD/USD FX CSVs...")
    df_fed = pd.read_csv('data/raw/FEDFUNDS.csv', parse_dates=['observation_date'], index_col='observation_date')
    df_fed.columns = ['fed_funds_rate']
    df_fed = df_fed.loc[START_DATE:]

    df_us_cpi = pd.read_csv('data/raw/CPIAUCSL.csv', parse_dates=['observation_date'], index_col='observation_date')
    df_us_cpi.columns = ['us_cpi']
    df_us_cpi = df_us_cpi.loc[START_DATE:]

    df_fx = load_fx_csv('data/raw/audusd.csv').loc[START_DATE:]

    # 3. Build official RBA Cash Rate Target series (2018 - Nov 2026)
    print("Building official RBA Cash Rate Target and Australian CPI series...")
    rba_data = {
        '2018-01-01': 1.50, '2019-06-05': 1.25, '2019-07-03': 1.00, '2019-10-02': 0.75,
        '2020-03-04': 0.50, '2020-03-19': 0.25, '2020-11-04': 0.10, '2022-05-04': 0.35,
        '2022-06-08': 0.85, '2022-07-06': 1.35, '2022-08-03': 1.85, '2022-09-07': 2.35,
        '2022-10-05': 2.60, '2022-11-02': 2.85, '2022-12-07': 3.10, '2023-02-07': 3.35,
        '2023-03-07': 3.60, '2023-05-02': 3.85, '2023-06-06': 4.10, '2023-11-08': 4.35,
        '2025-02-19': 4.10, '2025-05-21': 3.85, '2025-08-13': 3.60, '2026-02-04': 3.85,
        '2026-03-18': 4.10, '2026-05-06': 4.35, '2026-09-30': 4.60, '2026-11-04': 4.85
    }
    df_rba = pd.DataFrame(list(rba_data.items()), columns=['date', 'cash_rate_target'])
    df_rba['date'] = pd.to_datetime(df_rba['date'])
    df_rba.set_index('date', inplace=True)

    # 4. Build official Australian CPI YoY % series (2018 - Sep 2026)
    cpi_data = {
        '2018-03-31': 1.9, '2018-06-30': 2.1, '2018-09-30': 1.9, '2018-12-31': 1.8,
        '2019-03-31': 1.3, '2019-06-30': 1.6, '2019-09-30': 1.7, '2019-12-31': 1.8,
        '2020-03-31': 2.2, '2020-06-30': -0.3, '2020-09-30': 0.7, '2020-12-31': 0.9,
        '2021-03-31': 1.1, '2021-06-30': 3.8, '2021-09-30': 3.0, '2021-12-31': 3.5,
        '2022-03-31': 5.1, '2022-06-30': 6.1, '2022-09-30': 7.3, '2022-12-31': 7.8,
        '2023-03-31': 7.0, '2023-06-30': 6.0, '2023-09-30': 5.4, '2023-12-31': 4.1,
        '2024-03-31': 3.6, '2024-06-30': 3.8, '2024-09-30': 2.8, '2024-12-31': 2.4,
        '2025-03-31': 2.4, '2025-06-30': 2.1, '2025-09-30': 3.2, '2025-12-31': 3.8,
        '2026-03-31': 4.6, '2026-06-30': 3.8, '2026-09-30': 4.3
    }
    df_cpi = pd.DataFrame(list(cpi_data.items()), columns=['date', 'cpi_inflation'])
    df_cpi['date'] = pd.to_datetime(df_cpi['date'])
    df_cpi.set_index('date', inplace=True)

    # 5. Align all series to ASX trading days via step-wise forward fill
    master_df = df_prices.copy()
    master_df['cash_rate_target'] = df_rba.reindex(df_prices.index, method='ffill').bfill()['cash_rate_target']
    master_df['cpi_inflation'] = df_cpi.reindex(df_prices.index, method='ffill').bfill()['cpi_inflation']
    master_df['fed_funds_rate'] = df_fed.reindex(df_prices.index, method='ffill').bfill()['fed_funds_rate']
    master_df['us_cpi'] = df_us_cpi.reindex(df_prices.index, method='ffill').bfill()['us_cpi']
    master_df['audusd_fx'] = df_fx.reindex(df_prices.index, method='ffill').bfill()['audusd_fx']

    master_path = 'data/processed/master_etf_macro.csv'
    master_df.to_csv(master_path)

    print(f"\nSuccessfully compiled consolidated master dataset: {master_path}")
    print(f"Date Range: {master_df.index.min().strftime('%Y-%m-%d')} to {master_df.index.max().strftime('%Y-%m-%d')}")
    print(f"Latest RBA Cash Rate: {master_df['cash_rate_target'].iloc[-1]:.2f}% | Latest AUD/USD: {master_df['audusd_fx'].iloc[-1]:.4f}")
    print(f"Columns ({len(master_df.columns)}): {list(master_df.columns)}")

if __name__ == "__main__":
    run_ingestion()