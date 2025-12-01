from statsmodels.tsa.arima.model import ARIMA
from models.base_model import BaseModel
import warnings
import numpy as np

class ARIMAModel(BaseModel):
    def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0)):
        """
        order: (p,d,q) для ARIMA
        seasonal_order: (P,D,Q,s) для сезонной ARIMA, s=сезонный период
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        warnings.filterwarnings("ignore")  # подавляем предупреждения

    def fit(self, series):
        """
        series: pd.Series
        """
        self.model_fit = ARIMA(series, order=self.order, seasonal_order=self.seasonal_order).fit()

    def predict(self, horizon, **kwargs):
        """
        horizon: количество шагов для прогноза
        """
        if self.model_fit is None:
            raise ValueError("Call fit() before predict()")
        preds = self.model_fit.forecast(steps=horizon)
        return preds.tolist()
