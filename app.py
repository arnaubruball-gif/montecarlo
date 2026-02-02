import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Halcón 4.0 Pro: Fractal & Volume", layout="wide", page_icon="🦅")

# --- 2. FUNCIONES MATEMÁTICAS ---

def calcular_hurst(ts):
    """Calcula el Exponente de Hurst para detectar reversión (H<0.5) o tendencia (H>0.5)"""
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
            # Descargamos 70 días para asegurar ventana de cálculo
            df_hist = yf.download(ticker, period="70d", interval="1d", progress=False)
            
            # Limpieza de Multi-index de Yahoo Finance
            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)
            df_hist = df_hist.dropna()

            if len(df_hist) > 40:
                prices = df_hist['Close'].values.flatten().astype(float)
                volumes = df_hist['Volume'].values.flatten().astype(float)
                
                # --- MÉTRICAS CUANTITATIVAS ---
                window = prices[-40:]
                ma40 = np.mean(window)
                std40 = np.std(window)
                z_diff = (prices[-1] - ma40) / std40 if std40 != 0 else 0
                
                # R-Squared (Convicción del movimiento)
                x = np.arange(len(window))
                coeffs = np.polyfit(x, window, 1)
                y_hat = np.poly1d(coeffs)(x)
                r2 = 1 - (np.sum((window - y_hat)**2) / np.sum((window - np.mean(window))**2))
                
                # Hurst Exponent
                hurst = calcular_hurst(prices[-50:])
                
                # Volumen Relativo (Ficción vs Realidad)
                vol_avg = np.mean(volumes[-20:])
                vol_rel = volumes[-1] / vol_avg if vol_avg > 0 else 1
                
                # Volatilidad para Montecarlo
                volatilidad = np.std(np.diff(prices[-20:]) / prices[-21:-1])

                results.append({
                    'Ticker': ticker, 'Precio': round(prices[-1], 4),
                    'Z-Diff': round(z_diff, 2), 'R2': round(r2, 3),
                    'Hurst': round(hurst, 2), 'Vol_Rel': round(vol_rel, 2),
                    'Volatilidad': volatilidad, 'MA40': ma40
                })
        except: continue
    return pd.DataFrame(results)

# --- 3. INTERFAZ Y LÓGICA DE CONTROL ---
st.title("🦅 Halcón 4.0: Fractal & Volume Terminal")
st.markdown("Sistema avanzado de detección de ineficiencias mediante Exponente de Hurst y Presión de Volumen.")

assets = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X', 
    'USDCAD=X', 'USDCHF=X', 'BTC-USD', 'GC=F', 'ES=F'
]

with st.spinner('Escaneando el ecosistema financiero...'):
    df = fetch_and_calculate(assets)

# --- VALIDACIÓN ANTIFALLOS ---
if df.empty:
    st.error("⚠️ Error: No se pudieron obtener datos. Reintenta en unos segundos.")
    st.stop()

# Score Halcón: Premia Z alto (desviación) + Hurst bajo (reversión) + Volumen bajo (ficción)
df['Score_Halcon'] = (abs(df['Z-Diff']) * (1 - df['Hurst']) / (df['Vol_Rel'] + 0.1)).round(2)
df = df.sort_values(by='Score_Halcon', ascending=False)

# --- 4. VISUALIZACIÓN ---
col_table, col_radar = st.columns([1, 1])

with col_table:
    st.subheader("📊 Matriz de Oportunidad")
    st.dataframe(df.style.background_gradient(subset=['Score_Halcon'], cmap='YlOrRd'), use_container_width=True)

with col_radar:
    st.subheader("🎯 Radar Fractal (Z-Diff vs Hurst)")
    # El tamaño de la burbuja es el Volumen Relativo
    fig = px.scatter(df, x="Z-Diff", y="Hurst", size="Vol_Rel", text="Ticker", 
                     color="Score_Halcon", color_continuous_scale="Viridis",
                     range_x=[-4, 4], range_y=[0.2, 0.8])
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="Límite Fractal (Random Walk)")
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

# --- 5. ANÁLISIS PROFUNDO (MONTECARLO) ---
st.divider()
target = st.selectbox("Selecciona un activo para Proyección de Reversión:", df['Ticker'])
d = df[df['Ticker'] == target].iloc[0]

c_verdict, c_monte = st.columns([1, 2])

with c_verdict:
    st.write(f"### Veredicto Táctico: {target}")
    st.metric("Confianza Fractal", f"{(1-d['Hurst'])*100:.1f}%")
    st.metric("Volumen Relativo", d['Vol_Rel'])
    
    if d['Hurst'] < 0.45 and abs(d['Z-Diff']) > 1.6:
        st.success("🔥 SEÑAL: Alta probabilidad de REVERSIÓN. Mercado sobreextendido.")
    elif d['Hurst'] > 0.55:
        st.warning("⚠️ TENDENCIA: Inercia detectada. Evitar operar en contra.")
    else:
        st.info("NEUTRAL: Sin ventaja estadística clara.")

with c_monte:
    st.subheader("🎲 Nube de Probabilidad (5 días)")
    sims, days = 250, 5
    rets = np.random.normal(0, d['Volatilidad'], (days, sims))
    paths = np.zeros((days+1, sims)); paths[0] = d['Precio']
    for t in range(1, days+1): paths[t] = paths[t-1] * (1 + rets[t-1])
    
    p10, p25, p50, p75, p90 = [np.percentile(paths, p, axis=1) for p in [10, 25, 50, 75, 90]]
    
    fig_mc = go.Figure()
    # Nube 80% (Zona Exterior)
    fig_mc.add_trace(go.Scatter(x=list(range(6))+list(range(6))[::-1], y=list(p90)+list(p10[::-1]), 
                                fill='toself', fillcolor='rgba(0,150,255,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Rango 80%'))
    # Nube 50% (Zona Interior)
    fig_mc.add_trace(go.Scatter(x=list(range(6))+list(range(6))[::-1], y=list(p75)+list(p25[::-1]), 
                                fill='toself', fillcolor='rgba(0,150,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Rango 50%'))
    # Trayectoria Central
    fig_mc.add_trace(go.Scatter(x=list(range(6)), y=p50, line=dict(color='cyan', width=3), name='Eje Central'))
    fig_mc.update_layout(template="plotly_dark", height=400, hovermode="x")
    st.plotly_chart(fig_mc, use_container_width=True)
