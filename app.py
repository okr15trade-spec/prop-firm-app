import io
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Probabilidad de pasar el reto", page_icon="📈", layout="wide")
st.title("Probabilidad de pasar el reto (Prop Firm)")
st.caption("Compara estrategias a partir del historial de operaciones y estima la probabilidad de superar tus reglas.")

@st.cache_data(show_spinner=False)
def read_csv(raw: bytes):
    last = None
    for sep in [None, ',', ';', '\\t', '|']:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, engine='python' if sep is None else 'c')
            if len(df.columns) > 1:
                return df
        except Exception as e:
            last = e
    raise ValueError(str(last) if last else 'No se pudo detectar el formato del CSV.')

def norm(s):
    return ''.join(c.lower() for c in str(s) if c.isalnum())

def detect(columns, candidates):
    normalized = {norm(c): c for c in columns}
    for token in candidates:
        for n, original in normalized.items():
            if token in n:
                return original
    return None

def simulate(values, start, target_pct, daily_pct, total_pct, mae_limit, stop_after, count_zero_loss, mae_abs, end_policy, n_sims, max_trades, seed):
    rng = np.random.default_rng(seed)
    profits = np.asarray(values, dtype=float)
    # Each path samples historical trades with replacement, preserving empirical trade outcomes.
    idx = rng.integers(0, len(profits), size=(n_sims, max_trades))
    sampled = profits[idx]
    target = start * (1 + target_pct / 100)
    total_floor = start * (1 - total_pct / 100)
    daily_floor_amount = start * daily_pct / 100
    equity = np.full(n_sims, start, dtype=float)
    day_start = np.full(n_sims, start, dtype=float)
    losses = np.zeros(n_sims, dtype=int)
    alive = np.ones(n_sims, dtype=bool)
    passed = np.zeros(n_sims, dtype=bool)
    pass_steps = np.zeros(n_sims, dtype=int)
    for t in range(max_trades):
        active = alive & ~passed
        if not active.any(): break
        # Approximate a new trading day every five trades; daily limit resets then.
        if t > 0 and t % 5 == 0:
            day_start[active] = equity[active]
        p = sampled[:, t]
        equity[active] += p[active]
        hit_daily = (day_start - equity) >= daily_floor_amount
        hit_total = equity <= total_floor
        breach = hit_daily | hit_total
        alive[active & breach] = False
        just_passed = active & (equity >= target) & ~breach
        passed[just_passed] = True
        pass_steps[just_passed] = t + 1
        is_loss = p < 0 if count_zero_loss else p <= 0
        losses[active] = np.where(is_loss[active], losses[active] + 1, 0)
        if stop_after > 0:
            alive[active & (losses >= stop_after)] = False
    # Paths that hit a loss rule or the consecutive-loss stop count as failures.
    # Still-running paths are either counted as failures or excluded, per the UI setting.
    if end_policy == 'Fin de datos cuenta como fallo':
        denominator = n_sims
    else:
        denominator = int((passed | ~alive).sum())
    probability = (passed.sum() / denominator * 100) if denominator else 0.0
    return {
        'probability': float(probability),
        'pass_steps': pass_steps[passed].tolist(),
        'resolved': denominator,
    }

with st.sidebar:
    st.subheader("Parámetros del reto")
    start = st.number_input("Balance inicial ($)", min_value=1.0, value=10000.0, step=500.0)
    target_pct = st.number_input("Objetivo (%)", min_value=0.1, value=10.0, step=0.5)
    daily_pct = st.number_input("Pérdida diaria máx. (%)", min_value=0.1, value=4.0, step=0.5)
    total_pct = st.number_input("Pérdida total máx. (%)", min_value=0.1, value=7.0, step=0.5)
    mae_limit = st.number_input("Límite MAE por trade ($; 0 desactiva)", min_value=0.0, value=500.0, step=50.0)
    stop_after = st.number_input("Parar tras K pérdidas consecutivas (0 desactiva)", min_value=0, value=0, step=1)
    count_zero_loss = st.checkbox("Contar pérdida si P/L < 0 (no ≤ 0)", value=True)
    mae_abs = st.checkbox("Usar |MAE| (en vez de signo negativo)", value=False)
    end_policy = st.selectbox("Si no llega al objetivo al final del recorrido", ["Excluir del cálculo", "Fin de datos cuenta como fallo"])
    with st.expander("Simulación"):
        n_sims = st.slider("Recorridos", 1000, 20000, 5000, 1000)
        max_trades = st.slider("Máximo de operaciones por recorrido", 20, 500, 200, 10)

uploads = st.file_uploader("Sube uno o varios CSV con tu historial de operaciones", type=["csv"], accept_multiple_files=True, help="Se detectan delimitador, P/L y MAE automáticamente cuando los nombres de columna son reconocibles.")
if not uploads:
    st.info("Carga tus CSV para ver una comparativa. Cada fila debe representar una operación y el archivo debe contener P/L por operación.")
    st.stop()

rows=[]
details={}
with st.spinner("Analizando archivos y simulando recorridos…"):
    for up in uploads:
        try:
            df = read_csv(up.getvalue())
            pnl_col = detect(df.columns, ['profitloss','profit','pnl','pl','beneficio','ganancia','resultado','netpnl'])
            mae_col = detect(df.columns, ['mae','maxadverse','excursionadversa'])
            if not pnl_col:
                st.warning(f"{up.name}: no encontré la columna P/L. Columnas disponibles: {', '.join(map(str, df.columns))}")
                continue
            pnl = pd.to_numeric(df[pnl_col].astype(str).str.replace(',', '.', regex=False).str.replace(r'[^0-9.\\-]', '', regex=True), errors='coerce')
            valid = pnl.notna()
            pnl = pnl[valid].to_numpy()
            if len(pnl) < 2:
                st.warning(f"{up.name}: hacen falta al menos dos operaciones válidas.")
                continue
            mae_values = None
            if mae_col and mae_limit > 0:
                m = pd.to_numeric(df.loc[valid, mae_col].astype(str).str.replace(',', '.', regex=False).str.replace(r'[^0-9.\\-]', '', regex=True), errors='coerce')
                m = m.fillna(0).to_numpy()
                m = np.abs(m) if mae_abs else np.maximum(0, -m)
                pnl = pnl[m <= mae_limit]
            if len(pnl) < 2:
                st.warning(f"{up.name}: quedan menos de dos operaciones tras aplicar el filtro MAE.")
                continue
            sim = simulate(pnl, start, target_pct, daily_pct, total_pct, mae_limit, stop_after, count_zero_loss, mae_abs, end_policy, n_sims, max_trades, 1000 + len(rows))
            rows.append({"Archivo":up.name,"Operaciones":len(pnl),"P/L medio ($)":np.mean(pnl),"P/L neto ($)":np.sum(pnl),"Ganadas":int((pnl>0).sum()),"Perdidas":int((pnl<0).sum()),"Prob. estimada (%)":sim['probability'],"Columna P/L":str(pnl_col),"Columna MAE":str(mae_col or "—")})
            details[up.name] = {"sim": sim, "pnl": pnl, "pnl_col": str(pnl_col), "mae_col": str(mae_col or "—")}
        except Exception as e:
            st.error(f"No pude leer {up.name}: {e}")

if rows:
    result = pd.DataFrame(rows).sort_values("Prob. estimada (%)", ascending=False).reset_index(drop=True)
    st.subheader("Comparativa de CSVs")
    c1,c2=st.columns(2)
    with c1: sort_col=st.selectbox("Ordenar por", result.columns, index=list(result.columns).index("Prob. estimada (%)"))
    with c2: ascending=st.toggle("Ascendente", value=False)
    result=result.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    st.dataframe(result, use_container_width=True, hide_index=True, column_config={"Prob. estimada (%)":st.column_config.ProgressColumn("Prob. estimada (%)", min_value=0,max_value=100,format="%.1f%%"),"P/L medio ($)":st.column_config.NumberColumn(format="$%.2f"),"P/L neto ($)":st.column_config.NumberColumn(format="$%.2f")})
    st.caption("La estimación remuestrea operaciones históricas con reemplazo. Reinicia el límite diario cada cinco operaciones; esto es una aproximación. No representa una garantía ni una predicción financiera. Revisa las reglas concretas de tu firma.")
    st.download_button("Descargar comparativa", result.to_csv(index=False).encode('utf-8-sig'), "comparativa_prop_firm.csv", "text/csv")

    st.subheader("Detalle y gráficos")
    selected = st.selectbox("Elegir CSV", result["Archivo"].tolist())
    detail = details[selected]
    sim = detail["sim"]
    objective_dollars = start * target_pct / 100
    daily_dollars = start * daily_pct / 100
    wins = sim["pass_steps"]
    metrics = st.columns(5)
    metrics[0].metric("Objetivo ($)", f"{objective_dollars:,.2f}")
    metrics[1].metric("Pérdida diaria ($)", f"{daily_dollars:,.2f}")
    metrics[2].metric("Inicios simulados", f"{n_sims:,}")
    metrics[3].metric("Prob. estimada", f"{sim['probability']:.2f}%")
    metrics[4].metric("Media de trades si pasa", f"{np.mean(wins):.2f}" if wins else "—")
    st.caption(f"Columnas usadas en '{selected}': P/L = {detail['pnl_col']} · MAE = {detail['mae_col']}")
    st.markdown("**Distribución de operaciones para pasar**")
    if wins:
        distribution = pd.Series(wins, name="# operaciones").value_counts().sort_index()
        chart_data = distribution.rename("Frecuencia").to_frame()
        st.bar_chart(chart_data, use_container_width=True, x_label="# operaciones", y_label="Frecuencia")
    else:
        st.info("Ningún recorrido simulado alcanzó el objetivo con estos parámetros.")
