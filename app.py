# app.py
import io
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st

# --------------------
# Paths / config
# --------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
NOTEBOOKS_DIR = BASE / "notebooks"

DEFAULT_CSV = DATA_DIR / "estadisticas_tmin.csv"
DEFAULT_VECTOR = DATA_DIR / "DISTRITOS.shp"

st.set_page_config(page_title="Perú Tmin — Zonal Stats", layout="wide")
st.title("Perú Tmin — Estadísticas zonales y políticas públicas")

# --------------------
# Utility: safe read CSV (try comma then tab)
# --------------------
def read_stats_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
            return df
        except Exception as e:
            raise RuntimeError(f"Error leyendo CSV: {e}")

# --------------------
# Load CSV and shapefile
# --------------------
try:
    df = read_stats_csv(DEFAULT_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

try:
    gdf = gpd.read_file(DEFAULT_VECTOR)
except Exception as e:
    st.error(f"No se pudo leer el shapefile: {e}")
    st.stop()

# normalize column names
df.columns = [c for c in df.columns]

# --------------------
# Show full table
# --------------------
st.header("Resultados (CSV)")
st.markdown(f"- Distritos en tabla: **{len(df)}**")
st.dataframe(df, use_container_width=True)

# CSV download
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV (estadísticas zonales)",
    csv_bytes,
    file_name="tmin_zonal_stats.csv",
    mime="text/csv"
)

# --------------------
# Visualizaciones
# --------------------
st.header("Visualizaciones")

# Histogram of mean
if "mean" in df.columns:
    mean_col = "mean"
elif "MEAN" in df.columns:
    mean_col = "MEAN"
else:
    mean_col = None

if mean_col:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df[mean_col].dropna(), bins=40, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Mean Tmin (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribución de la Tmin promedio por distrito")
    st.subheader("Distribución")
    st.pyplot(fig)
else:
    st.warning("No se encontró la columna 'mean' para graficar la distribución.")

# Top 15 cold / hot
st.subheader("Clasificación (Top 15)")

name_col = None
for cand in ["DISTRITO", "distrito", "DISTRICT", "NOMBDIST", "NAME"]:
    if cand in df.columns:
        name_col = cand
        break

if name_col is None:
    text_cols = [c for c in df.columns if df[c].dtype == object]
    name_col = text_cols[0] if text_cols else df.columns[0]

p10_col = "p10" if "p10" in df.columns else ("P10" if "P10" in df.columns else None)
p90_col = "p90" if "p90" in df.columns else ("P90" if "P90" in df.columns else None)

if mean_col:
    cold = df.nsmallest(15, mean_col)
    hot = df.nlargest(15, mean_col)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 15 más fríos**")
        if len(cold) > 0:
            st.table(
                cold[[c for c in [name_col, mean_col, p10_col] if c]]
                .rename(columns={mean_col: "mean", p10_col: "p10"})
                .style.format({"mean": "{:.2f}", "p10": "{:.2f}"})
            )
        else:
            st.write("No hay datos para Top fríos.")
    with c2:
        st.markdown("**Top 15 más cálidos**")
        if len(hot) > 0:
            st.table(
                hot[[c for c in [name_col, mean_col, p90_col] if c]]
                .rename(columns={mean_col: "mean", p90_col: "p90"})
                .style.format({"mean": "{:.2f}", "p90": "{:.2f}"})
            )
        else:
            st.write("No hay datos para Top cálidos.")
else:
    st.warning("No hay columna 'mean' para crear rankings.")

# --------------------
# Choropleth map
# --------------------
st.subheader("Mapa estático (choropleth)")

merged = None
try:
    if "UBIGEO" in df.columns and "UBIGEO" in gdf.columns:
        df_ = df.copy()
        gdf_ = gdf.copy()
        df_["UBIGEO"] = df_["UBIGEO"].astype(str).str.zfill(6)
        gdf_["UBIGEO"] = gdf_["UBIGEO"].astype(str).str.zfill(6)
        merged = gdf_.merge(df_, on="UBIGEO")
except Exception as e:
    st.warning(f"No se pudo realizar merge para mapa: {e}")

if merged is None or len(merged) == 0:
    st.warning("No se pudo generar el mapa por falta de coincidencia entre shapefile y CSV.")
else:
    try:
        merged = merged.to_crs(epsg=4326)
    except Exception:
        pass

    if mean_col and mean_col in merged.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        merged.plot(
            column=mean_col,
            ax=ax,
            legend=True,
            cmap="coolwarm",
            edgecolor="0.6",
            linewidth=0.2
        )
        ax.set_axis_off()
        ax.set_title("Choropleth: Mean Tmin (°C)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        st.image(buf, caption="Mapa Tmin — Choropleth", use_column_width=True)
        st.download_button(
            "Descargar mapa (PNG)",
            data=buf.getvalue(),
            file_name="map_tmin_mean.png",
            mime="image/png"
        )
    else:
        st.warning("No hay columna 'mean' en el dataset fusionado para pintar el mapa.")


# --------------------
# Show pre-rendered figures
# --------------------
st.subheader("Figuras pre-renderizadas (notebooks)")

cold_img_path = NOTEBOOKS_DIR / "figures_top15_cold.png"
hot_img_path = NOTEBOOKS_DIR / "figures_top15_hot.png"
map_img_path = NOTEBOOKS_DIR / "figures_mapa_tmin.png"

c1, c2 = st.columns(2)
with c1:
    if cold_img_path.exists():
        with open(cold_img_path, "rb") as f:
            img_bytes = f.read()
            st.image(img_bytes, caption="Top15 Cold (pre-rendered)", use_column_width=True)
            st.download_button("Descargar figura cold (PNG)", data=img_bytes,
                               file_name="figures_top15_cold.png", mime="image/png")
    else:
        st.write("No se encontró notebooks/figures_top15_cold.png")

with c2:
    if hot_img_path.exists():
        with open(hot_img_path, "rb") as f:
            img_bytes = f.read()
            st.image(img_bytes, caption="Top15 Hot (pre-rendered)", use_column_width=True)
            st.download_button("Descargar figura hot (PNG)", data=img_bytes,
                               file_name="figures_top15_hot.png", mime="image/png")
    else:
        st.write("No se encontró notebooks/figures_top15_hot.png")

st.markdown("---")
if map_img_path.exists():
    with open(map_img_path, "rb") as f:
        img_bytes = f.read()
        st.image(img_bytes, caption="Mapa Tmin (pre-rendered)", use_column_width=True)
        st.download_button("Descargar mapa Tmin (PNG)", data=img_bytes,
                           file_name="figures_mapa_tmin.png", mime="image/png")
else:
    st.write("No se encontró notebooks/figures_mapa_tmin.png")


# --------------------
# Public policy section
# --------------------
st.header("Políticas Públicas — Diagnóstico y medidas priorizadas")
st.markdown("""
**Diagnóstico (resumen):**  
Los distritos con P10 ≤ 0°C y altos porcentajes de heladas presentan riesgos en agricultura, ganadería y salud.

**Medidas priorizadas:**

1. **Kits antihelada para ganadería**  
   - **Territorio:** Distritos con alto % de heladas.  
   - **Presupuesto:** S/ 1,000 por familia.  
   - **KPI:** −25% mortalidad de alpacas en 2 años.

2. **Aislamiento térmico en escuelas y postas**  
   - **Territorio:** Zonas con P10 ≤ 0°C.  
   - **Presupuesto:** S/ 6,000 por institución.  
   - **KPI:** +10% asistencia escolar en meses fríos.

3. **Alerta temprana y capacitación comunitaria**  
   - **Territorio:** Comunidades altoandinas.  
   - **Presupuesto:** S/ 3,000 por comunidad.  
   - **KPI:** Respuesta < 48h ante eventos de friaje.
""")