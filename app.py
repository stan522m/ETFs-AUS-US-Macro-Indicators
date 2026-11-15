import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ASX 12-ETF Macro Risk Engine", layout="wide")

st.title("ASX 12-ETF Quantitative Risk & Macro-Correlation Engine")
st.markdown(
    "Interactive multi-asset risk engine analyzing historical performance (2018–2026) "
    "against **RBA Monetary Policy**, **Australian CPI**, **US Fed Funds**, **US CPI**, and **AUD/USD FX**."
)

# 1. Load Data
@st.cache_data
def load_data():
    path = "data/processed/master_etf_macro.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

if not os.path.exists("data/processed/master_etf_macro.csv"):
    st.error("Master dataset not found. Run `python src/ingest.py` first to generate `data/processed/master_etf_macro.csv`.")
    st.stop()

df = load_data()

etf_tickers = [
    'NDQ.AX', 'HACK.AX', 'IXJ.AX', 'GOLD.AX', 
    'VAS.AX', 'VGS.AX', 'VGAD.AX', 'VAF.AX', 
    'AAA.AX', 'QRE.AX', 'VAP.AX', 'VHY.AX'
]
active_etfs = [t for t in etf_tickers if t in df.columns]
macro_cols = ['cash_rate_target', 'cpi_inflation', 'fed_funds_rate', 'us_cpi', 'audusd_fx']
active_macros = [m for m in macro_cols if m in df.columns]

# 2. Sidebar: Interactive Portfolio Weighting
st.sidebar.header("Portfolio Allocation Settings")
allocation_mode = st.sidebar.radio(
    "Weighting Strategy",
    ["Equal-Weight (8.33% each)", "Custom Slider Weights"]
)

weights = {}
if allocation_mode == "Equal-Weight (8.33% each)":
    for t in active_etfs:
        weights[t] = 1.0 / len(active_etfs)
else:
    st.sidebar.caption("Adjust raw weights below (automatically normalized to 100%):")
    raw_weights = {t: st.sidebar.slider(t, 0, 100, 10) for t in active_etfs}
    total_w = sum(raw_weights.values())
    weights = {t: (w / total_w if total_w > 0 else 1.0 / len(active_etfs)) for t, w in raw_weights.items()}

weight_series = pd.Series(weights)

# 3. Portfolio Return & Risk Calculations
df_returns = df[active_etfs].pct_change().dropna()
portfolio_returns = df_returns.dot(weight_series)

ann_return = portfolio_returns.mean() * 252
ann_vol = portfolio_returns.std() * np.sqrt(252)
latest_rba = df['cash_rate_target'].iloc[-1] / 100.0
sharpe_ratio = (ann_return - latest_rba) / ann_vol if ann_vol > 0 else 0.0

cum_returns = (1 + portfolio_returns).cumprod()
drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
max_drawdown = drawdown.min()

# 4. Top KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualized Return (PA)", f"{ann_return * 100:.2f}%")
col2.metric("Annualized Volatility", f"{ann_vol * 100:.2f}%")
col3.metric("Sharpe Ratio (vs RBA)", f"{sharpe_ratio:.2f}")
col4.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")

st.divider()

# 5. Interactive Charts
tab1, tab2, tab3 = st.tabs(["Cumulative Growth & Drawdown", "AU & US Macro Correlation Matrix", "Asset-Level Macro Sensitivities"])

with tab1:
    st.subheader("Portfolio Cumulative Wealth Index ($1 Base) & Drawdown")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name="Portfolio Growth ($1)", line=dict(color="#1f77b4", width=2)))
    fig_cum.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Growth of $1")
    st.plotly_chart(fig_cum, use_container_width=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown (%)", fill='tozeroy', line=dict(color="#d62728", width=1.5)))
    fig_dd.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    st.subheader("Extended Macro Sensitivities (AU & US Drivers)")
    macro_corr_df = pd.DataFrame({'Portfolio': portfolio_returns})
    for col in active_macros:
        macro_corr_df[col] = df[col].reindex(portfolio_returns.index)
    
    corr_matrix = macro_corr_df.corr()
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_heat.update_layout(height=480)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(corr_matrix.style.format("{:.6f}"), use_container_width=True)

with tab3:
    st.subheader("Individual ETF Sensitivity to US Fed Rate & AUD/USD FX")
    asset_corr = pd.concat([df_returns, df[active_macros].reindex(df_returns.index)], axis=1).corr()
    etf_macro_table = asset_corr.loc[active_etfs, active_macros]
    st.dataframe(etf_macro_table.style.background_gradient(cmap="RdBu_r", vmin=-0.5, vmax=0.5).format("{:.4f}"), use_container_width=True)