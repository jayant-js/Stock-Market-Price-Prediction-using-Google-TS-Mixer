import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import yfinance as yf
import ta.momentum, ta.trend
import datetime
from dateutil.relativedelta import relativedelta
import joblib

# From darts
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.metrics import mape  

class DataGatheringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ticker_symbol: str, years_back: int = 7):
        self.ticker_symbol = ticker_symbol
        self.years_back = years_back

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        curr_date = datetime.date.today()
        data = yf.download(self.ticker_symbol, start = curr_date - relativedelta(years=7), end = curr_date, auto_adjust=False)
        if data is None:
            raise ValueError(f"No data found for ticker {self.ticker_symbol}")
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        # Calculate 14-day RSI
        df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Adj Close'].squeeze(), window=14).rsi()
        # calculate the MACD
        macd = ta.trend.MACD(close=df['Adj Close'].squeeze())
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # Calculate Moving Averages
        df["20MA"] = df["Adj Close"].rolling(window=20).mean()  
        df["50MA"] = df["Adj Close"].rolling(window=50).mean()
        df["200MA"] = df["Adj Close"].rolling(window=200).mean()

        df["20EMA"] = df["Adj Close"].ewm(span=20, adjust=False).mean()
        df["50EMA"] = df["Adj Close"].ewm(span=50, adjust=False).mean()
        df["200EMA"] = df["Adj Close"].ewm(span=200, adjust=False).mean()

        df['Close_Lag1'] = df['Adj Close'].shift(1)
        df['Close_Lag2'] = df['Adj Close'].shift(2)

        df.bfill(inplace=True)
        df.ffill(inplace=True)

        df.set_index('Date', inplace=True)
        df = df.asfreq('D')
        df.interpolate(method='linear', inplace=True)
        return df
    
class DartsModelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model_params:dict, output_chunk_length: int=7):
        self.model_params = model_params
        self.output_chunk_length = output_chunk_length
        self.model = None
        self.target_scaler = Scaler()
        self.covariates_scaler = Scaler()
        self.add_encoders = {
            'cyclic' : {'future': ['day', 'dayofweek', 'week', 'month']}, 
            'datetime_attribute' : {'future': ['day', 'dayofweek', 'week', 'month']},
            'position':{'past': ['relative'], 'future':['relative']},
            'custom':{'past':[self.encode_year], 'future':[self.encode_year]},
            'transformer': Scaler(),
            'tz':'Asia/Kolkata'
        }
        
    def encode_year(self, idx):
        return (idx.year - 2000) / 50

    def fit(self, X, y=None):
        df = X.copy()
        df.rename(columns={'Adj Close':'y'}, inplace=True)
        series = TimeSeries.from_dataframe(df, value_cols='y', freq='D')
        past_covariates = TimeSeries.from_dataframe(df.drop('y', axis=1), freq='D')

        series_scaled = self.target_scaler.fit_transform(series)
        past_covariates_scaled = self.covariates_scaler.fit_transform(past_covariates)

        self.model = TSMixerModel(
            output_chunk_length = self.output_chunk_length,
            add_encoders = self.add_encoders, 
            **self.model_params
        )

        train_series = series_scaled[:-self.output_chunk_length]
        val_series = series_scaled[-self.output_chunk_length:]

        train_past_covariates = past_covariates_scaled[:-self.output_chunk_length]

        self.model.fit(
            series = train_series,
            past_covariates = train_past_covariates,
            verbose=True
        )

        y_pred_scaled = self.model.predict(n=self.output_chunk_length)
        validation_mape = mape(val_series, y_pred_scaled)
        print(f"Validation MAPE: {validation_mape:.2f}%")

        print("Retraining model on full dataset for future predictions...")
        self.model.fit(
            series = series_scaled,
            past_covariates = past_covariates_scaled,
            verbose = False
        )
        print('Final Model Fitting complete')
        return self

    def predict(self, X, forecast_horizon, y=None):
        if not self.model:
            raise RuntimeError("The model has not been fitted yet")
    
        if forecast_horizon is None:
            raise ValueError('`forecast_horizon` must be provided to make prediction.')
        
        n_forecast = forecast_horizon
        scaled_prediction = self.model.predict(n=n_forecast)
        original_scale_prediction = self.target_scaler.inverse_transform(scaled_prediction)
        print("Prediction Complete.")
        return original_scale_prediction.to_dataframe()

def create_pipeline(ticker:str, forecast_horizon: int = 7):
    model_params = {
        'input_chunk_length':32,
        'ff_size':32,
        'num_blocks':4,
        'hidden_size': 64,
        'n_epochs':7,
        'use_reversible_instance_norm':True,
        # 'pl_trainer_kwargs':{'accelerator':'gpu', 'devices':[0]}
    }

    return Pipeline([
        ('Data Gathering', DataGatheringTransformer(ticker_symbol=ticker)),
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('model_training_and_prediction', DartsModelWrapper(
            model_params = model_params,
            output_chunk_length=forecast_horizon    
        ))
    ])