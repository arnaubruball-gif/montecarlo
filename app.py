import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Configuración de Élite
st.set_page_config(page_title="Halcón 4.0 - Swing Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3d4463; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🦅 Halcón de Guerra 4.0 | Swing Intelligence")
st.write("Análisis de Ineficiencias Estructurales y Fuerza Relativa")

# --- 2. MOTOR DE ANÁLISIS (CONFIGURACIÓN SWING 30/40 DÍAS) ---
ASSETS = {
    'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'AUD/USD': 'AUDUSD=X',
    'NZD/USD': 'NZDUSD=X', 'USD/JPY': 'JPY=X', 'USD/CHF': 'CHF=X',
    'USD/CAD': 'CAD=X', 'BITCOIN': 'BTC-USD', 'ORO (Spot)': 'GC=F',
    'S&P 500': '^SPX', 'NASDAQ 100': '^IXIC'
}

def analyze_asset_swing(name, ticker):
    try:
        df = yf.download(ticker, period='200d', interval='1d', progress=False)
        if df.empty: return None, None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Cálculo de Retornos y Flujo
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
        df['RMF'] = df['Close'] * df['Vol_Proxy']
        
        # R2 Dinámico (Ventana 30 días)
        r2_series = []
        for i in range(len(df)):
            if i < 30: r2_series.append(0); continue
            subset = df.iloc[i-30:i].dropna()
            r2 = sm.OLS(subset['Ret'], sm.add_constant(subset['RMF'])).fit().rsquared
            r2_series.append(r2)
        df['R2_Dynamic'] = r2_series
        
        # Z-Diff (Ventana 40 días)
        diff = df['Ret'].rolling(40).sum() - df['RMF'].pct_change().rolling(40).sum()
        z_val = ((diff - diff.rolling(40).mean()) / (diff.rolling(40).std() + 1e-10)).iloc[-1]
        
        # Amihud (Iliquidez)
        amihud = (df['Ret'].abs() / (df['RMF'].replace(0, np.nan) / 1e6)).fillna(df['Ret'].abs() * 100).rolling(20).mean().iloc[-1]
        
        last_r2 = df['R2_Dynamic'].iloc[-1]
        last_price = df['Close'].iloc[-1]
        std_dev = df['Ret'].tail(40).std()
        
        veredicto = "⚪ NEUTRAL"
        tp_val = "N/A"
        
        if last_r2 < 0.10:
            if z_val > 1.6: 
                veredicto = "🚨 VENTA (Ficción)"
                tp_val = f"{last_price * (1 - (abs(z_val) * std_dev)):.4f}"
            elif z_val < -1.6: 
                veredicto = "🟢 COMPRA (Oportunidad)"
                tp_val = f"{last_price * (1 + (abs(z_val) * std_dev)):.4f}"
        elif last_r2 > 0.30: 
            veredicto = "💎 TENDENCIA REAL"

        return df, [name, f"{last_price:.4f}", round(last_r2, 3), round(z_val, 2), round(amihud, 4), veredicto, tp_val]
    except:
        return None, None

# --- 3. PANEL DE CONTROL ---
tab1, tab2 = st.tabs(["📊 Matriz & ADN", "🧮 Calculadora de Cruces"])

with tab1:
    col_btn, col_empty = st.columns([1, 3])
    if col_btn.button('📡 ESCANEAR MATRIZ USD'):
        all_data = []
        for name, ticker in ASSETS.items():
            _, s = analyze_asset_swing(name, ticker)
            if s: all_data.append(s)
        
        df_res = pd.DataFrame(all_data, columns=['Activo', 'Precio', 'R2 (30d)', 'Z-Diff (40d)', 'Amihud', 'Veredicto', 'TP Swing'])
        
        def style_v(val):
            if 'VENTA' in val: return 'color: #ff4b4b; font-weight: bold'
            if 'COMPRA' in val: return 'color: #00ffcc; font-weight: bold'
            if 'TENDENCIA' in val: return 'color: #1c83e1; font-weight: bold'
            return ''
        
        st.dataframe(df_res.style.applymap(style_v, subset=['Veredicto']), use_container_width=True)

    st.write("---")
    selected = st.selectbox("🎯 Visualizar ADN del Activo:", list(ASSETS.keys()))
    df_plot, _ = analyze_asset_swing(selected, ASSETS[selected])
    if df_plot is not None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        colors = ['cyan' if r > 0.30 else ('lightgrey' if r < 0.10 else 'orange') for r in df_plot['R2_Dynamic']]
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['R2_Dynamic'], marker_color=colors, name="R2"), row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Gestión de Riesgo por Volatilidad (ATR)")
    c1, c2, c3 = st.columns(3)
    with c1: pair = st.text_input("Par Cruce (ej: GBPAUD=X)", "GBPAUD=X")
    with c2: side = st.selectbox("Dirección", ["COMPRA (Largo)", "VENTA (Corto)"])
    with c3: risk = st.number_input("% Riesgo Cuenta", 1.0)
    
    if st.button("Calcular Niveles ATR"):
        data = yf.download(pair, period='30d', interval='1d', progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1]
            price = data['Close'].iloc[-1]
            
            sl = price - (1.5 * atr) if "COMPRA" in side else price + (1.5 * atr)
            tp = price + (3.0 * atr) if "COMPRA" in side else price - (3.0 * atr)
            
            st.success(f"### Setup {pair}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Entrada", f"{price:.4f}")
            mc2.metric("STOP LOSS", f"{sl:.4f}")
            mc3.metric("TAKE PROFIT", f"{tp:.4f}")
            st.info(f"Riesgo basado en Volatilidad Real: {round(atr*10000, 0)} pips de ATR.")

st.sidebar.markdown("""
### 🧠 Manual del Halcón
1. **Z-Diff > 1.6 & R2 < 0.10**: Ineficiencia de sobrevaloración (Vender).
2. **Z-Diff < -1.6 & R2 < 0.10**: Ineficiencia de infravaloración (Comprar).
3. **R2 > 0.30**: Dinero real empujando. No operes en contra.
4. **Swing**: Revisa solo al cierre de vela diaria.
""")
