import solara
import pandas as pd
import plotly.express as px
import geopandas as gpd
import json
import unicodedata
import plotly.graph_objects as go
# Define the list of years and months
years = list(range(2015, 2024))  # 2015 to 2023
months = ['All'] + list(range(1, 13))  # 'All' and months 1 to 12

# Reactive variables for year, month, and hortaliza selection
year = solara.reactive(2015)
month = solara.reactive('All')
hortaliza = solara.reactive('All')
# Function to normalize department names
def normalize_name(name):
    name = name.upper()
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = name.strip()
    return name

# Read and prepare the data
@solara.memoize
def load_data():
    # Load your data files (ensure paths are correct)
    data = pd.read_parquet('/home/jovyan/data/aggregated_data.parquet')
    gdf = gpd.read_file('/home/jovyan/data/DEPARTAMENTOS.shp')

    # Adjust column names to uppercase for consistency
    data.columns = [col.upper() for col in data.columns]

    # Process the data
    data['DEPARTAMENTO_NORM'] = data['DEPARTAMENTO'].apply(normalize_name)
    data['HORTALIZA_AGRUPADA'] = data['HORTALIZA_AGRUPADA'].apply(normalize_name)
    gdf['NOMBDEP_NORM'] = gdf['NOMBDEP'].apply(normalize_name)

    # No need to aggregate; data is already aggregated
    aggregated_data = data.copy()
    geojson_data = json.loads(gdf.to_json())

    return aggregated_data, geojson_data
# Function to plot the map for the selected year and month
# Get the list of available hortalizas from the data
@solara.memoize
def get_hortalizas():
    aggregated_data, _ = load_data()
    hortalizas_list = aggregated_data['HORTALIZA_AGRUPADA'].dropna().unique().tolist()
    hortalizas_list = sorted(hortalizas_list)
    return hortalizas_list
# Set default hortaliza to the first in the list
hortalizas_list = get_hortalizas()
default_hortaliza = hortalizas_list[0] if hortalizas_list else 'All'
hortaliza = solara.reactive(default_hortaliza)

@solara.component
def SowingMap(year, month, hortaliza):
    aggregated_data, geojson_data = load_data()

    # Filter data
    df_filtered = aggregated_data
    df_filtered = df_filtered[df_filtered['AÑO'] == year]
    if month != 'All':
        df_filtered = df_filtered[df_filtered['MES'] == int(month)]
    if hortaliza != 'All':
        df_filtered = df_filtered[df_filtered['HORTALIZA_AGRUPADA'] == hortaliza]

    # Aggregate over departments
    df_filtered = df_filtered.groupby('DEPARTAMENTO_NORM')['SIEMBRA'].sum().reset_index()

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        df_filtered,
        geojson=geojson_data,
        locations='DEPARTAMENTO_NORM',
        featureidkey='properties.NOMBDEP_NORM',
        color='SIEMBRA',
        color_continuous_scale="YlOrRd",
        mapbox_style='carto-positron',
        zoom=4.5,
        center={"lat": -9.19, "lon": -75.0152},
        opacity=0.6,
        labels={'SIEMBRA': 'Sowing (hectares)'},
        hover_data={'SIEMBRA': ':.0f'}
    )
    fig.update_coloraxes(colorbar_tickformat='d')

    # Update the map title
    month_text = 'All Months' if month == 'All' else f"Month {str(month).zfill(2)}"
    hortaliza_text = hortaliza

    fig.update_layout(
        title_text=f"Sowing Data for {year} - {month_text} - {hortaliza_text}",
        margin={"r": 0, "t": 30, "l": 0, "b": 50}  # Adjusted bottom margin
    )

    return solara.FigurePlotly(fig)

@solara.component
def HarvestMap(year, month, hortaliza):
    aggregated_data, geojson_data = load_data()

    # Filter data
    df_filtered = aggregated_data
    df_filtered = df_filtered[df_filtered['AÑO'] == year]
    if month != 'All':
        df_filtered = df_filtered[df_filtered['MES'] == int(month)]
    if hortaliza != 'All':
        df_filtered = df_filtered[df_filtered['HORTALIZA_AGRUPADA'] == hortaliza]

    # Aggregate over departments
    df_filtered = df_filtered.groupby('DEPARTAMENTO_NORM')['COSECHA'].sum().reset_index()

    # Check for empty data
    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text="No data available for the selected period",
            mapbox_style='carto-positron',
            margin={"r": 0, "t": 30, "l": 0, "b": 50}
        )
    else:
        fig = px.choropleth_mapbox(
            df_filtered,
            geojson=geojson_data,
            locations='DEPARTAMENTO_NORM',
            featureidkey='properties.NOMBDEP_NORM',
            color='COSECHA',
            color_continuous_scale="Blues",
            mapbox_style='carto-positron',
            zoom=4.5,
            center={"lat": -9.19, "lon": -75.0152},
            opacity=0.6,
            labels={'COSECHA': 'Harvest (hectares)'},
            hover_data={'COSECHA': ':.0f'}
        )
        fig.update_coloraxes(colorbar_tickformat='d')
        month_text = 'All Months' if month == 'All' else f"Month {str(month).zfill(2)}"
        hortaliza_text = hortaliza
        fig.update_layout(
            title_text=f"Harvest Data for {year} - {month_text} - {hortaliza_text}",
            margin={"r": 0, "t": 30, "l": 0, "b": 50}
        )

    return solara.FigurePlotly(fig)

@solara.component
def BarPlot(hortaliza):
    aggregated_data, _ = load_data()

    # Filter data based on hortaliza
    df_filtered = aggregated_data[aggregated_data['HORTALIZA_AGRUPADA'] == hortaliza]

    # Aggregate data per month and year
    df_monthly = df_filtered.groupby(['AÑO', 'MES'])[['SIEMBRA', 'COSECHA']].sum().reset_index()

    # Create a 'Month-Year' column for x-axis
    df_monthly['MONTH_YEAR'] = df_monthly.apply(lambda row: f"{row['AÑO']}-{str(row['MES']).zfill(2)}", axis=1)

    # Sort by 'AÑO' and 'MES'
    df_monthly = df_monthly.sort_values(['AÑO', 'MES']).reset_index(drop=True)

    # Create the bar plot
    fig = go.Figure(data=[
        go.Bar(name='Sowing', x=df_monthly['MONTH_YEAR'], y=df_monthly['SIEMBRA']),
        go.Bar(name='Harvest', x=df_monthly['MONTH_YEAR'], y=df_monthly['COSECHA'])
    ])

    # Set x-axis to always show 12 months at a time
    fig.update_layout(
        barmode='group',
        xaxis_title='Month-Year',
        yaxis_title='Amount (hectares)',
        title=f"Sowing vs Harvest over time for {hortaliza}",
        xaxis=dict(
            type='category',
            tickangle=45,
            range=[df_monthly['MONTH_YEAR'].iloc[0], df_monthly['MONTH_YEAR'].iloc[min(11, len(df_monthly)-1)]],
            rangeslider=dict(visible=True),
            tickmode='array',
            tickvals=df_monthly['MONTH_YEAR'][::3],  # Show tick labels every 3 months to reduce clutter
            ticktext=df_monthly['MONTH_YEAR'][::3]
        ),
        height=200  # Adjusted height to fit 30% vertically
    )

    return solara.FigurePlotly(fig)
# Page component for the Solara app
@solara.component
def Page():
    hortalizas_list = get_hortalizas()
    with solara.Column():
        # Controls
        solara.Select(label="Crop", value=hortaliza, values=hortalizas_list)
        solara.Select(label="Year", value=year, values=years)
        solara.Select(label="Month", value=month, values=months)

        # Main content
        with solara.Row():
            # Left Column (50% width)
            with solara.Column(style={"width": "50%"}):
                # Top 30%: Bar Plot
                with solara.Column(style={"height": "40%"}):
                    BarPlot(hortaliza.value)

                # Bottom 70%: Sowing Map
                with solara.Column(style={"height": "70vh"}):
                    solara.Markdown("### Sowing Data")
                    SowingMap(year.value, month.value, hortaliza.value)

            # Right Column (50% width)
            with solara.Column(style={"width": "50%"}):
                solara.Markdown("### Harvest Data")
                HarvestMap(year.value, month.value, hortaliza.value)