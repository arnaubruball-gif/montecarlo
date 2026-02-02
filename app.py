import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Halcón 4.0: Fractal & Volume", layout="wide", page_icon="🦅")

# --- 2. FUNCIONES MATEMÁTICAS (Sin dependencias externas) ---

def calcular_hurst(ts):
    """Mide si el precio es Reversivo (H < 0.5) o Tendencial (H > 0.5)"""
    if len(ts) < 30: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=600)
def fetch_and_calculate(tickers):
    results = []
    for ticker in tickers:
        try:
            # Descargamos 70 días para tener margen de cálculo
            df_hist = yf.download(ticker, period="70d", interval="1d", progress=False)
            
            # Limpiar Multi-index de Yahoo Finance
            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)
            df_hist = df_hist.dropna()

            if len(df_hist) > 40:
                prices = df_hist['Close'].values.flatten().astype(float)
                volumes = df_hist['Volume'].values.flatten().astype(float)
                
                # --- MÉTRICAS HALCÓN ---
                window = prices[-40:]
                ma40 = np.mean(window)
                std40 = np.std(window)
                z_diff = (prices[-1] - ma40) / std40 if std40 != 0 else 0
                
                # R-Squared (Convicción del movimiento)
                x = np.arange(len(window))
                coeffs = np.polyfit(x, window, 1)
                y_hat = np.poly1d(coeffs)(x)
                r2 = 1 - (np.sum((window - y_hat)**2) / np.sum((window - np.mean(window))**2))
                
                # Hurst Exponent (Fractalidad)
                hurst = calcular_hurst(prices[-50:])
                
                # Volumen Relativo (Ficción vs Realidad)
                vol_avg = np.mean(volumes[-20:])
                vol_rel = volumes[-1] / vol_avg if vol_avg > 0 else 1
                
                volatilidad = np.std(np.diff(prices[-20:]) / prices[-21:-1])

                results.append({
                    'Ticker': ticker, 'Precio': round(prices[-1], 4),
                    'Z-Diff': round(z_diff, 2), 'R2': round(r2, 3),
                    'Hurst': round(hurst, 2), 'Vol_Rel': round(vol_rel, 2),
                    'Volatilidad': volatilidad, 'MA40': ma40
                })
        except: continue
    return pd.DataFrame(results)

# --- 3. DASHBOARD PRINCIPAL ---
st.title("🦅 Halcón 4.0: Terminal de Ficción Estadística")
st.markdown("Análisis fractal de reversión a la media con validación de volumen.")

assets = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X', 
    'USDCAD=X', 'USDCHF=X', 'BTC-USD', 'GC=F', 'ES=F'
]

with st.spinner('Cazando ineficiencias...'):
    df = fetch_and_calculate(assets)

# Score Halcón: Premia Z alto + Hurst bajo (Reversivo) + Vol bajo (Ficción)
df['Score_Halcon'] = (abs(df['Z-Diff']) * (1 - df['Hurst']) / (df['Vol_Rel'] + 0.1)).round(2)
df = df.sort_values(by='Score_Halcon', ascending=False)

# Visualización superior
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("📊 Matriz de Oportunidad")
    st.dataframe(df.style.background_gradient(subset=['Score_Halcon'], cmap='YlOrRd'), use_container_width=True)

with c2:
    st.subheader("🎯 Radar Fractal (Z vs Hurst)")
    # El tamaño de la burbuja es el Volumen Relativo
    fig = px.scatter(df, x="Z-Diff", y="Hurst", size="Vol_Rel", text="Ticker", 
                     color="Score_Halcon", color_continuous_scale="Viridis",
                     range_x=[-4, 4], range_y=[0.2, 0.8])
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="Límite Fractal")
    st.plotly_chart(fig, use_container_width=True)

# --- 4. ANÁLISIS DE PROYECCIÓN (MONTECARLO) ---
st.divider()
target = st.selectbox("Selecciona un activo para proyectar:", df['Ticker'])
d = df[df['Ticker'] == target].iloc[0]

col_a, col_b = st.columns([1, 2])

with col_a:
    st.write(f"### Veredicto: {target}")
    st.metric("Confianza Fractal", f"{(1-d['Hurst'])*100:.1f}%", help="H < 0.5 indica alta probabilidad de reversión")
    st.metric("Volumen Relativo", d['Vol_Rel'])
    
    if d['Hurst'] < 0.45 and abs(d['Z-Diff']) > 1.6:
        st.success("🔥 SEÑAL: Reversión de alta probabilidad. El elástico está estirado.")
    elif d['Hurst'] > 0.55:
        st.warning("⚠️ TENDENCIA: El mercado tiene inercia. No operar contra tendencia.")
    else:
        st.info("NEUTRAL: Esperando confirmación de ineficiencia.")

with col_b:
    st.subheader("🎲 Nube de Probabilidad (5 días)")
    sims, days = 250, 5
    rets = np.random.normal(0, d['Volatilidad'], (days, sims))
    paths = np.zeros((days+1, sims)); paths[0] = d['Precio']
    for t in range(1, days+1): paths[t] = paths[t-1] * (1 + rets[t-1])
    
    p10, p25, p50, p75, p90 = [np.percentile(paths, p, axis=1) for p in [10, 25, 50, 75, 90]]
    
    fig_mc = go.Figure()
    # Nube 80%
    fig_mc.add_trace(go.Scatter(x=list(range(6))+list(range(6))[::-1], y=list(p90)+list(p10[::-1]), 
                                fill='toself', fillcolor='rgba(0,150,255,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Rango 80%'))
    # Nube 50%
    fig_mc.add_trace(go.Scatter(x=list(range(6))+list(range(6))[::-1], y=list(p75)+list(p25[::-1]), 
                                fill='toself', fillcolor='rgba(0,150,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Rango 50%'))
    # Eje central
    fig_mc.add_trace(go.Scatter(x=list(range(6)), y=p50, line=dict(color='cyan', width=3), name='Trayectoria Media'))
    fig_mc.update_layout(height=400, hovermode="x")
    st.plotly_chart(fig_mc, use_container_width=True)
