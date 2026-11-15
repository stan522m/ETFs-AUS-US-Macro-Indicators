# ASX Multi-Asset Quantitative Risk & Macro-Correlation Engine

An end-to-end quantitative data pipeline and interactive web application evaluating a 12-ETF Australian portfolio against domestic and international macroeconomic drivers from 2018 to 2026.

## Key Features
- **ETL Ingestion Pipeline (`src/ingest.py`):** Consolidates daily adjusted close prices for 12 ASX ETFs (`NDQ.AX`, `VAS.AX`, `GOLD.AX`, `VAF.AX`, etc.) with official RBA Cash Rates, ABS CPI, US Fed Funds Rates, US CPI, and AUD/USD exchange rates.
- **Risk & Performance Analytics (`src/analytics.py`):** Computes portfolio CAGR, annualized volatility, dynamic Sharpe ratios, maximum drawdowns, and cross-asset correlation matrices.
- **Interactive Web App (`app.py`):** Deployed via Streamlit, allowing users to run custom weight simulations, inspect wealth trajectories, and visualize macro sensitivities.

## Live Demo
🔗 [View Live Streamlit App](https://your-streamlit-app-url.streamlit.app)
