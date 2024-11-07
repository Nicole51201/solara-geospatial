import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns

class VegetablePriceAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'sarima': None  # Se inicializará según los datos
        }
        self.best_model = None
        self.feature_importance = None

    def load_data(self, price_file, weather_file, economic_file):
        """
        Carga y combina datos de diferentes fuentes
        """
        # Cargar datos
        prices = pd.read_csv(price_file, parse_dates=['fecha'])
        weather = pd.read_csv(weather_file, parse_dates=['fecha'])
        economic = pd.read_csv(economic_file, parse_dates=['fecha'])

        # Combinar datasets
        df = prices.merge(weather, on='fecha', how='left')
        df = df.merge(economic, on='fecha', how='left')
        
        # Crear variables adicionales
        df['mes'] = df['fecha'].dt.month
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['es_feriado'] = df['fecha'].isin(self.get_holidays())  # Implementar get_holidays()
        
        return df

    def prepare_features(self, df):
        """
        Prepara características para el modelado
        """
        # Variables climáticas
        weather_features = ['temperatura_max', 'temperatura_min', 'precipitacion', 'humedad']
        
        # Variables económicas
        economic_features = ['tipo_cambio', 'inflacion', 'precio_combustible']
        
        # Variables temporales
        temporal_features = ['mes', 'dia_semana', 'es_feriado']
        
        # Variables de mercado
        market_features = ['stock_disponible', 'demanda_estimada']
        
        features = weather_features + economic_features + temporal_features + market_features
        
        X = df[features]
        y = df['precio']
        
        # Escalamiento de características
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y

    def train_models(self, X, y):
        """
        Entrena múltiples modelos y selecciona el mejor
        """
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # Entrenar Random Forest
        self.models['rf'].fit(X_train, y_train)
        rf_pred = self.models['rf'].predict(X_test)
        results['rf'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred)
        }
        
        # Entrenar XGBoost
        self.models['xgb'].fit(X_train, y_train)
        xgb_pred = self.models['xgb'].predict(X_test)
        results['xgb'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'r2': r2_score(y_test, xgb_pred)
        }
        
        # Entrenar SARIMA para series temporales
        # Implementar entrenamiento SARIMA aquí
        
        # Seleccionar mejor modelo
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])[0]
        self.best_model = self.models[best_model]
        
        # Calcular importancia de características
        if best_model in ['rf', 'xgb']:
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results

    def analyze_external_factors(self, df):
        """
        Analiza el impacto de factores externos en los precios
        """
        # Calcular correlaciones
        corr_matrix = df.corr()['precio'].sort_values(ascending=False)
        
        # Análisis de estacionalidad
        seasonal_analysis = self.analyze_seasonality(df)
        
        # Análisis de eventos externos
        event_impact = self.analyze_events(df)
        
        return {
            'correlations': corr_matrix,
            'seasonality': seasonal_analysis,
            'events': event_impact
        }

    def generate_predictions(self, X_future):
        """
        Genera predicciones para datos futuros
        """
        X_scaled = self.scaler.transform(X_future)
        predictions = self.best_model.predict(X_scaled)
        
        return predictions

    def plot_results(self, df, predictions):
        """
        Genera visualizaciones de resultados
        """
        plt.figure(figsize=(15, 10))
        
        # Plot de precios reales vs predicciones
        plt.subplot(2, 2, 1)
        plt.plot(df['fecha'], df['precio'], label='Real')
        plt.plot(df['fecha'], predictions, label='Predicción')
        plt.title('Precios Reales vs Predicciones')
        plt.legend()
        
        # Plot de importancia de características
        if self.feature_importance is not None:
            plt.subplot(2, 2, 2)
            sns.barplot(data=self.feature_importance.head(10), 
                       x='importance', y='feature')
            plt.title('Importancia de Características')
        
        plt.tight_layout()
        plt.show()

def main():
    analyzer = VegetablePriceAnalyzer()
    
    # Cargar datos
    df = analyzer.load_data('precios.csv', 'clima.csv', 'economia.csv')
    
    # Preparar características
    X, y = analyzer.prepare_features(df)
    
    # Entrenar modelos
    results = analyzer.train_models(X, y)
    
    # Analizar factores externos
    external_analysis = analyzer.analyze_external_factors(df)
    
    # Generar predicciones
    predictions = analyzer.generate_predictions(X)
    
    # Visualizar resultados
    analyzer.plot_results(df, predictions)
    
    return results, external_analysis

if __name__ == "__main__":
    main()
