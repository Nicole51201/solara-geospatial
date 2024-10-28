import solara
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import unicodedata

# Función para normalizar nombres
def normalize_name(name):
    name = name.upper()
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = name.strip()
    return name

# Cargar los datos de las Ollas Comunes
@solara.memoize
def load_ollas_data():
    # Load the data
    ollas = pd.read_excel('/home/jovyan/data/OLLAS DE LIMA.xlsx')

    # Replace commas with dots in latitude and longitude
    ollas['Latitude'] = ollas['Latitude'].astype(str).str.replace(',', '.')
    ollas['Longitude'] = ollas['Longitude'].astype(str).str.replace(',', '.')

    # Convert to numeric, setting errors='coerce' to handle invalid entries
    ollas['Latitude'] = pd.to_numeric(ollas['Latitude'], errors='coerce')
    ollas['Longitude'] = pd.to_numeric(ollas['Longitude'], errors='coerce')

    # Drop rows with NaN values in 'Latitude' or 'Longitude'
    ollas = ollas.dropna(subset=['Latitude', 'Longitude'])

    # Optional: Remove entries with invalid coordinate ranges
    ollas = ollas[
        (ollas['Latitude'] >= -90) & (ollas['Latitude'] <= 90) &
        (ollas['Longitude'] >= -180) & (ollas['Longitude'] <= 180)
    ]

    return ollas

# Cargar el GeoDataFrame de Lima
@solara.memoize
def load_lima_geo():
    # Load the shapefile of the departments
    gdf = gpd.read_file('/home/jovyan/data/DEPARTAMENTOS.shp')

    # Check if CRS is set
    if gdf.crs is None:
        # Set the CRS based on your shapefile's coordinate system
        # Replace 'epsg=4326' with the correct EPSG code if necessary
        gdf.set_crs(epsg=4326, inplace=True)

    # Normalize names
    gdf['NOMBDEP_NORM'] = gdf['NOMBDEP'].apply(normalize_name)

    # Filter Lima
    lima = gdf[gdf['NOMBDEP_NORM'] == 'LIMA']

    return lima

# Componente para mostrar el mapa
@solara.component
def MapaOllasComunes():
    # Load the data for community pots
    ollas = load_ollas_data()
    
    # Drop rows with missing lat/lon values
    ollas = ollas.dropna(subset=['Latitude', 'Longitude'])

    # Ensure latitude/longitude are floats
    ollas['Latitude'] = ollas['Latitude'].astype(float)
    ollas['Longitude'] = ollas['Longitude'].astype(float)

    # Create the map with Plotly
    fig = px.scatter_mapbox(
        ollas,
        lat="Latitude",
        lon="Longitude",
        hover_name="Nombre de las Ollas Comunes",  # Show pot name on hover
        hover_data={"N° Benef. Padrón": True},  # Show number of beneficiaries
        color_discrete_sequence=["blue"],  # Blue markers
        zoom=9,  # Adjust zoom level for Lima
        height=500
    )

    # Set mapbox style and centering
    fig.update_layout(
        mapbox_style="carto-positron",  # Light style for better contrast
        mapbox_zoom=9,  # Zoom into Lima
        mapbox_center={"lat": -12.0464, "lon": -77.0428},  # Center on Lima
        margin={"r":0,"t":0,"l":0,"b":0}  # Remove margins
    )

    return solara.FigurePlotly(fig)
# Página principal
@solara.component
def Page():
    with solara.Column():
        solara.Markdown("## Mapa de Ollas Comunes en Lima")
        MapaOllasComunes()
