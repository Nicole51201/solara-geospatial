import solara
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json
import unicodedata
import solara
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Define the list of years and months
years = list(range(2015, 2024))  # 2015 to 2023
months = ['All'] + list(range(1, 13))  # 'All' and months 1 to 12

# Function to normalize names
def normalize_name(name):
    if pd.isnull(name):
        return ''
    name = name.upper()
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = name.strip()
    return name

# Reactive variables for selections
year = solara.reactive(2015)
year_price = solara.reactive(2015)
month = solara.reactive('All')
hortaliza = solara.reactive('LECHUGA')  # Default to 'LECHUGA'

# List of selected hortalizas
selected_hortalizas = ['LECHUGA', 'AJI', 'ESPARRAGO', 'TOMATE', 'BROCOLI']
selected_hortalizas = [normalize_name(h) for h in selected_hortalizas]

# Load and prepare the price data
@solara.memoize
def load_price_data():
    # Load the price data (ensure the path is correct)
    price_data = pd.read_excel('/home/jovyan/data/PRECIO_chakra_5.xlsx')

    return price_data

# Get the list of available hortalizas from the data
@solara.memoize
def get_hortalizas():
    price_data = load_price_data()
    hortalizas_list = price_data['HORTALIZA_AGRUPADA'].dropna().unique().tolist()
    hortalizas_list = sorted(hortalizas_list)
    return hortalizas_list

# Set default hortaliza to the first in the list
hortalizas_list = get_hortalizas()
default_hortaliza = hortalizas_list[0] if hortalizas_list else 'LECHUGA'
hortaliza.value = default_hortaliza

# Get the list of years from the price data
price_data = load_price_data()
years_price_list = sorted(price_data['AÑO'].unique().tolist())
year_price.value = years_price_list[0] if years_price_list else 2015

# Define your plotting components
@solara.component
def PriceLinePlot(year):
    price_data = load_price_data()

    # Filter data for the selected year
    df_filtered = price_data[price_data['AÑO'] == year]

    # Aggregate data per month and hortaliza
    df_grouped = df_filtered.groupby(['MES', 'HORTALIZA_AGRUPADA'])['PRECIO_CHACRA'].mean().reset_index()

    # Sort by month
    df_grouped = df_grouped.sort_values('MES')

    # Create the line plot
    fig = px.line(
        df_grouped,
        x='MES',
        y='PRECIO_CHACRA',
        color='HORTALIZA_AGRUPADA',
        markers=True,
        labels={
            'MES': 'Month',
            'PRECIO_CHACRA': 'Price (S/.)',
            'HORTALIZA_AGRUPADA': 'Crop'
        },
        title=f"Price Variation in {year}"
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis_title='Price (S/.)',
        xaxis_title='Month',
        legend_title='Crop',
        template='plotly_white'
    )

    return solara.FigurePlotly(fig)

@solara.component
def PriceForecast(hortaliza):
    price_data = load_price_data()

    # Filter data for the selected hortaliza
    df_filtered = price_data[price_data['HORTALIZA_AGRUPADA'] == hortaliza]

    # Create a datetime column
    df_filtered['DATE'] = pd.to_datetime(df_filtered['AÑO'].astype(str) + '-' + df_filtered['MES'].astype(str) + '-01')

    # Aggregate data monthly and calculate mean values
    df_monthly = df_filtered.groupby('DATE').agg({
        'PRECIO_CHACRA': 'mean',
        'SIEMBRA': 'mean',
        'COSECHA': 'mean',
        'PRODUCCION': 'mean'
    }).reset_index()

    # Ensure the data is sorted by date
    df_monthly = df_monthly.sort_values('DATE')

    # Set 'DATE' as the index
    df_monthly.set_index('DATE', inplace=True)

    # Split data into training and forecasting sets
    train_data = df_monthly.iloc[:-3]  # Use all but the last 3 months for training
    forecast_data = df_monthly.iloc[-3:]  # Last 3 months for forecasting

    # Prepare the endog and exog variables
    endog = train_data['PRECIO_CHACRA']
    exog = train_data[['SIEMBRA', 'COSECHA', 'PRODUCCION']]

    # Check if we have enough data points
    if len(endog) < 12:  # Need at least 12 data points for SARIMAX
        return solara.Warning("Not enough data to perform forecasting.")

    # Fit the SARIMAX model
    # You may need to adjust the order and seasonal_order parameters
    model = SARIMAX(endog, exog=exog, order=(1,1,1), seasonal_order=(0,1,1,12))
    results = model.fit(disp=False)

    # Prepare exogenous variables for forecasting
    exog_forecast = forecast_data[['SIEMBRA', 'COSECHA', 'PRODUCCION']]

    # Forecast the next 3 months
    forecast = results.get_forecast(steps=3, exog=exog_forecast)
    forecast_ci = forecast.conf_int()

    # Get forecasted values
    forecasted_values = forecast.predicted_mean

    # Combine historical and forecasted data for plotting
    history = endog.copy()
    forecast_index = forecasted_values.index

    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history.values,
        mode='lines+markers',
        name='Historical Price'
    ))

    # Forecasted data
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecasted_values,
        mode='lines+markers',
        name='Forecasted Price'
    ))

    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_ci['upper PRECIO_CHACRA'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_ci['lower PRECIO_CHACRA'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    # Customize layout
    fig.update_layout(
        title=f"Price Forecast for {hortaliza}",
        xaxis_title='Date',
        yaxis_title='Price (S/.)',
        template='plotly_white',
        legend=dict(orientation='h', y=-0.2)
    )

    return solara.FigurePlotly(fig)
@solara.component
def PriceProductionScatter(hortaliza):
    price_data = load_price_data()

    # Filter data for the selected hortaliza
    df_filtered = price_data[price_data['HORTALIZA_AGRUPADA'] == hortaliza]

    # Aggregate data per month
    df_grouped = df_filtered.groupby(['AÑO', 'MES'])[['PRECIO_CHACRA', 'PRODUCCION']].mean().reset_index()

    # Create a scatter plot
    fig = px.scatter(
        df_grouped,
        x='PRODUCCION',
        y='PRECIO_CHACRA',
        trendline='ols',
        labels={
            'PRODUCCION': 'Production (Tonnes)',
            'PRECIO_CHACRA': 'Price (S/.)'
        },
        title=f"Price vs Production for {hortaliza}"
    )

    fig.update_layout(
        template='plotly_white'
    )

    return solara.FigurePlotly(fig)

# If you have BarPlot, SowingMap, and HarvestMap components, define them here as well.
# Since you're focusing on the new data and functionality, we'll proceed without redefining them unless needed.

# Now, define the Page component with all dependencies included
@solara.component
def Page():
    hortalizas_list = get_hortalizas()
    with solara.Column():
        # Controls
        solara.Select(label="Crop", value=hortaliza, values=hortalizas_list)
        solara.Select(label="Year", value=year, values=years_price_list)
        solara.Select(label="Month", value=month, values=months)
        solara.Select(label="Year for Price Plot", value=year_price, values=years_price_list)
    
        # Main content
        with solara.Row():
            # Left Column (50% width)
            with solara.Column(style={"width": "50%"}):
                # Top 30%: Price Line Plot
                with solara.Column(style={"height": "30vh"}):
                    PriceLinePlot(year_price.value)
    
                # Middle 30%: Price vs Production Scatter Plot
                with solara.Column(style={"height": "30vh"}):
                    PriceProductionScatter(hortaliza.value)
    
                # Bottom 40%: (If you have another component or map)
                with solara.Column(style={"height": "40vh"}):
                    # Placeholder for additional component
                    solara.Markdown("### Additional Analysis")
                    # Add your component here
    
            # Right Column (50% width)
            with solara.Column(style={"width": "50%"}):
                # Top: Price Forecast
                with solara.Column(style={"height": "40vh"}):
                    PriceForecast(hortaliza.value)
    
                # Middle and Bottom: (If you have other components)
                # You can adjust the layout as needed
                # For example, include BarPlot, SowingMap, or HarvestMap if they are relevant
