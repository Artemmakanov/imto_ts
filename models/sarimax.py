# forecasting_module/models/sarimax_forecaster.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as stsSARIMAX
from .base_model import BaseModel

class SARIMAX(BaseModel):
    def __init__(self, order=(1,0,1), seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, series, exog=None):
        """
        series: pd.Series
        exog: pd.DataFrame с регрессорами (год, месяц, праздники...)
        """
        self.model_fit = stsSARIMAX(series,
                                 order=self.order,
                                 seasonal_order=self.seasonal_order,
                                 exog=exog,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)

    def predict(self, horizon, exog=None):
        if self.model_fit is None:
            raise ValueError("Call fit() before predict()")
        preds = self.model_fit.get_forecast(steps=horizon, exog=exog)
        return preds.predicted_mean.values.tolist()
