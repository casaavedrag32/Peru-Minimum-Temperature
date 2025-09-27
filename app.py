import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.plot import show
from io import BytesIO

st.set_page_config(page_title="App Estadísticas Zonas", layout="wide")

st.title("📊 Estadísticas Zonales de Temperatura Mínima")

# -------------------------------

# 1. Carga de ráster

# -------------------------------

st.sidebar.header("Carga de datos")

uploaded_file = st.sidebar.file_uploader("Sube un ráster (.tif)", type=["tif", "tiff"])
if uploaded_file:
raster = rasterio.open(uploaded_file)
else:
st.sidebar.info("Usando ráster de ejemplo incluido")
raster = rasterio.open("data/tmin.tif")  # asegúrate de incluirlo en tu repo

array = raster.read(1).astype(float)
array[array == raster.nodata] = np.nan

# -------------------------------

# 2. Estadísticas zonales

# -------------------------------

st.subheader("Estadísticas básicas")
stats = {
"mean": np.nanmean(array),
"min": np.nanmin(array),
"max": np.nanmax(array),
"std": np.nanstd(array),
"p10": np.nanpercentile(array, 10),
"p90": np.nanpercentile(array, 90),
"custom_metric": np.nanmean(array) - np.nanstd(array),  # ejemplo métrica personalizada
}

df_stats = pd.DataFrame.from_dict(stats, orient="index", columns=["valor"])
st.table(df_stats)

# -------------------------------

# 3. Gráficos

# -------------------------------

st.subheader("Gráficos")

col1, col2, col3 = st.columns(3)

with col1:
st.markdown("**Distribución (Histograma)**")
fig, ax = plt.subplots()
ax.hist(array[~np.isnan(array)], bins=30, color="skyblue", edgecolor="black")
ax.set_xlabel("Valor")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

with col2:
st.markdown("**Clasificación (Boxplot)**")
fig, ax = plt.subplots()
ax.boxplot(array[~np.isnan(array)], vert=True)
ax.set_ylabel("Valor")
st.pyplot(fig)

with col3:
st.markdown("**Mapa Estático**")
fig, ax = plt.subplots(figsize=(4, 4))
show(raster, ax=ax, cmap="coolwarm")
st.pyplot(fig)

# -------------------------------

# 4. Descargar tabla

# -------------------------------

st.subheader("Descargar resultados")

csv = df_stats.to_csv().encode("utf-8")
st.download_button(
label="⬇️ Descargar estadísticas en CSV",
data=csv,
file_name="estadisticas_tmin.csv",
mime="text/csv",
)

# -------------------------------

# 5. Políticas Públicas

# -------------------------------

st.subheader("🏛 Políticas Públicas")

st.markdown("""

### Diagnóstico

Las temperaturas mínimas extremas impactan en la salud y productividad agrícola,
especialmente en zonas altoandinas con poblaciones vulnerables.

### Medidas priorizadas

1. **Programa de techos térmicos para viviendas rurales**

   * **Población objetivo:** familias en sierra sur (>3,500 msnm).
   * **Presupuesto estimado:** USD 10 millones.
   * **KPI:** 20,000 hogares beneficiados en 5 años.

2. **Seguro agrícola frente a heladas**

   * **Territorio:** principales cuencas productoras de papa.
   * **Presupuesto estimado:** USD 15 millones anuales.
   * **KPI:** 80% de agricultores familiares cubiertos.

3. **Alertas tempranas comunitarias**

   * **Población objetivo:** comunidades campesinas altoandinas.
   * **Presupuesto estimado:** USD 5 millones.
   * **KPI:** Tiempo de respuesta < 48h en 90% de eventos.
     """)
