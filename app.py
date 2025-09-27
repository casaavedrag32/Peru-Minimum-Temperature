# app.py
import os
from pathlib import Path
import tempfile
import io

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import streamlit as st

# --------------------
# Config / paths
# --------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DEFAULT_RASTER = DATA_DIR / "tmin_raster.tif"
DEFAULT_VECTOR = DATA_DIR / "DISTRITOS.shp"

st.set_page_config(page_title="Perú Tmin — Zonal Stats", layout="wide")
st.title("Perú Tmin — Estadísticas zonales y políticas públicas")

# --------------------
# Helpers
# --------------------
def ensure_geometry_column(gdf):
    """If geometry column isn't named 'geometry', set it to the first geometry dtype column."""
    if "geometry" not in gdf.columns:
        geom_cols = [c for c in gdf.columns if gdf[c].dtype.name == "geometry"]
        if geom_cols:
            gdf = gdf.set_geometry(geom_cols[0])
    return gdf

def detect_name_column(gdf):
    """Return a name column to label polygons for tables/plots (create if none)."""
    candidates = ["DISTRITO", "NOMB_DIST", "NOMBDIST", "NOMBRE", "NAME", "DISTRICT", "DIST"]
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback: create 'NAME' from index
    gdf["NAME"] = gdf.index.astype(str)
    return "NAME"

def compute_zonal_stats(vector_gdf, raster_path):
    """Compute zonal stats and a custom frost metric (pct pixels < 0). Return GeoDataFrame (lowercase cols)."""
    # open raster to get crs and nodata
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

    gdf = vector_gdf.copy()
    gdf = ensure_geometry_column(gdf)

    # Reproject to raster CRS for correct overlay
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if raster_crs is not None:
        try:
            gdf = gdf.to_crs(raster_crs)
        except Exception:
            pass

    # Use rasterstats: base stats + percentiles + custom frost count
    stats = ["count", "min", "max", "mean", "std"]
    percentiles = [10, 90]

    try:
        zs = zonal_stats(
            gdf.geometry,
            raster_path,
            stats=stats,
            percentiles=percentiles,
            add_stats={"frost_pixels": lambda arr: int(np.sum(np.array(arr) < 0))},
            nodata=nodata,
            geojson_out=False,
            all_touched=False
        )
    except Exception as e:
        raise RuntimeError(f"Error en zonal_stats: {e}")

    df = pd.DataFrame(zs)

    # Ensure percentile keys normalized: rasterstats may name them 'percentile_10' etc.
    if "percentile_10" in df.columns:
        df["p10"] = df["percentile_10"]
    elif "10" in df.columns:
        df["p10"] = df["10"]
    if "percentile_90" in df.columns:
        df["p90"] = df["percentile_90"]
    elif "90" in df.columns:
        df["p90"] = df["90"]

    # Merge geometry back
    out = pd.concat([gdf.reset_index(drop=True), df], axis=1)

    # normalize column names to lowercase for consistency with your saved CSV
    rename_map = {}
    for c in ["mean", "min", "max", "std", "count"]:
        if c in out.columns:
            rename_map[c] = c  # already lowercase
    # ensure frost_pixels present
    if "frost_pixels" not in out.columns and "FROST_PIXELS" in out.columns:
        out["frost_pixels"] = out["FROST_PIXELS"]

    # compute frost_pct
    out["pixels"] = out.get("count", np.nan)
    out["frost_pixels"] = out.get("frost_pixels", 0)
    out["frost_pct"] = np.where(out["pixels"] > 0, out["frost_pixels"] / out["pixels"] * 100, 0.0)

    # keep important cols lowercase
    # already many are lowercase (mean,min,max,std,p10,p90)
    return gpd.GeoDataFrame(out, geometry="geometry")

def plot_hist_mean(gdf):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if "mean" in gdf.columns:
        ser = gdf["mean"].dropna()
    elif "MEAN" in gdf.columns:
        ser = gdf["MEAN"].dropna()
    else:
        ser = pd.Series(dtype=float)
    ax.hist(ser, bins=40, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mean Tmin (°C)")
    ax.set_ylabel("Count")
    ax.set_title("Distribución de la Tmin promedio por polígono")
    plt.tight_layout()
    return fig

def plot_top_tables(gdf, name_col, n=15):
    # Use 'mean' lowercase where available
    mean_col = "mean" if "mean" in gdf.columns else ("MEAN" if "MEAN" in gdf.columns else None)
    p10_col = "p10" if "p10" in gdf.columns else ("P10" if "P10" in gdf.columns else None)
    p90_col = "p90" if "p90" in gdf.columns else ("P90" if "P90" in gdf.columns else None)
    frost_pct_col = "frost_pct" if "frost_pct" in gdf.columns else ("FROST_PCT" if "FROST_PCT" in gdf.columns else None)

    if mean_col is None:
        raise RuntimeError("No se encontró columna 'mean' en los resultados zonales.")

    cold_cols = [name_col, mean_col]
    if p10_col:
        cold_cols.append(p10_col)
    if frost_pct_col:
        cold_cols.append(frost_pct_col)

    hot_cols = [name_col, mean_col]
    if p90_col:
        hot_cols.append(p90_col)

    cold = gdf.nsmallest(n, mean_col)[cold_cols].copy()
    hot = gdf.nlargest(n, mean_col)[hot_cols].copy()

    # rename to friendly names for display
    cold = cold.rename(columns={mean_col: "mean", p10_col: "p10", frost_pct_col: "frost_pct"})
    hot = hot.rename(columns={mean_col: "mean", p90_col: "p90"})

    return cold.reset_index(drop=True), hot.reset_index(drop=True)

def plot_choropleth_png(gdf, value_col="mean"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    col = value_col if value_col in gdf.columns else ("MEAN" if "MEAN" in gdf.columns else None)
    if col is None:
        raise RuntimeError("No hay columna para mapear.")
    gdf.plot(column=col, ax=ax, legend=True, cmap="coolwarm", edgecolor="0.6", linewidth=0.2)
    ax.set_axis_off()
    ax.set_title(f"Choropleth: {col}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------
# Sidebar: Uploads / defaults
# --------------------
st.sidebar.header("Datos")
uploaded_raster = st.sidebar.file_uploader("Sube un GeoTIFF (opcional)", type=["tif", "tiff"])
uploaded_vector = st.sidebar.file_uploader("Sube un shapefile (.shp) o GeoJSON (opcional)", type=["geojson", "json", "shp"])

# choose raster path (uploaded overrides default)
if uploaded_raster is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmp.write(uploaded_raster.read())
    tmp.flush()
    raster_path = tmp.name
else:
    raster_path = str(DEFAULT_RASTER) if DEFAULT_RASTER.exists() else None

# choose vector path
if uploaded_vector is not None:
    tmpv = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
    tmpv.write(uploaded_vector.read())
    tmpv.flush()
    vector_path = tmpv.name
else:
    vector_path = str(DEFAULT_VECTOR) if DEFAULT_VECTOR.exists() else None

if raster_path is None:
    st.error("No se encontró ráster. Sube un GeoTIFF o coloca uno en ./data/tmin_raster.tif")
    st.stop()

if vector_path is None:
    st.error("No se encontró shapefile/geojson. Sube uno o coloca DISTRITOS.shp en ./data/")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write(f"Ráster usado: `{Path(raster_path).name}`")
st.sidebar.write(f"Vector usado: `{Path(vector_path).name}`")

# --------------------
# Load vector & compute
# --------------------
with st.spinner("Cargando límites..."):
    try:
        gdf = gpd.read_file(vector_path)
    except Exception as e:
        st.error(f"Error leyendo vector: {e}")
        st.stop()

    gdf = ensure_geometry_column(gdf)
    name_col = detect_name_column(gdf)

with st.spinner("Calculando estadísticas zonales (puede tardar)..."):
    try:
        zonal = compute_zonal_stats(gdf, raster_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

# --------------------
# Show results: table + CSV download
# --------------------
st.header("Resultados")
st.markdown(f"- Polígonos: **{len(zonal)}**")
display_cols = [name_col, "mean", "min", "max", "std", "p10", "p90", "frost_pct"]
display_cols = [c for c in display_cols if c in zonal.columns]
st.dataframe(zonal[display_cols].head(20))

# CSV download (use lowercase names for portability)
out_df = zonal.drop(columns=[c for c in ["geometry"] if "geometry" in zonal.columns], errors="ignore")
# ensure columns as in your example: UBIGEO, DEPARTAMEN, PROVINCIA, DISTRITO, year, count, mean, min...
# if those exist already keep them; otherwise export the computed table
csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV (estadísticas zonales)", csv_bytes, file_name="tmin_zonal_stats.csv", mime="text/csv")

# --------------------
# Plots: distribution, ranking, map
# --------------------
st.header("Visualizaciones")

# 1) Distribution
fig_hist = plot_hist_mean(zonal)
st.subheader("Distribución")
st.pyplot(fig_hist)

# 2) Ranking (top 15 cold/hot)
st.subheader("Clasificación (Top 15)")
try:
    cold_df, hot_df = plot_top_tables(zonal, name_col, n=15)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 15 más fríos**")
        st.table(cold_df.style.format({"mean":"{:.2f}","p10":"{:.2f}","frost_pct":"{:.1f}"}))
    with c2:
        st.markdown("**Top 15 más cálidos**")
        st.table(hot_df.style.format({"mean":"{:.2f}","p90":"{:.2f}"}))
except Exception as e:
    st.warning(f"No se pudo generar ranking completo: {e}")

# 3) Static map
st.subheader("Mapa estático (choropleth)")
try:
    png_buf = plot_choropleth_png(zonal, value_col="mean")
    st.image(png_buf)
    st.download_button("Descargar mapa (PNG)", data=png_buf.getvalue(), file_name="map_tmin_mean.png", mime="image/png")
except Exception as e:
    st.warning(f"No se pudo generar el mapa: {e}")

# --------------------
# Public policy section
# --------------------
st.header("Políticas Públicas — Diagnóstico y medidas priorizadas")
st.markdown("""
**Diagnóstico (resumen):**  
El análisis zonal muestra distritos con P10 ≤ 0°C y altos porcentajes de píxeles bajo 0°C (FROST_PCT) concentrados en zonas altoandinas —esto indica riesgo para agricultura, ganadería y salud (IRA/ARI).

**Medidas priorizadas (ejemplo):**

1. **Kits antihelada para ganadería**  
   - **Población/territorio:** Productores en distritos con FROST_PCT ≥ 30%.  
   - **Presupuesto estimado:** S/ 1,000 por familia (kit alimentario + cobertores).  
   - **KPI:** −25% mortalidad de alpacas en 2 años.

2. **Programas de aislamiento térmico en escuelas y postas**  
   - **Población/territorio:** Instituciones en distritos con P10 ≤ 0°C.  
   - **Presupuesto estimado:** S/ 6,000 por escuela/posta.  
   - **KPI:** +10% asistencia escolar durante meses fríos; −15% consultas por ARI en postas.

3. **Alerta temprana y capacitación comunitaria**  
   - **Población/territorio:** Comunidades campesinas en provincias más afectadas.  
   - **Presupuesto estimado:** S/ 3,000 por comunidad (sensores + capacitación).  
   - **KPI:** Tiempo de respuesta < 48h ante eventos de friaje.
""")

