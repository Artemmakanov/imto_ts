from statsmodels.tsa.statespace.sarimax import SARIMAX

from .metrics import *
from .utils import *


def fit_arimax(train_df, exog_cols=None, order=(1,0,1), seasonal_order=(0,0,0,0)):
    exog_train = train_df[exog_cols] if exog_cols else None
    model = SARIMAX(train_df['cnt'],
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def forecast_arimax(model_fit, test_df, exog_cols=None):
    exog_test = test_df[exog_cols] if exog_cols else None
    pred = model_fit.get_forecast(steps=len(test_df), exog=exog_test)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()
    return pred_mean, pred_ci

class ARIMAXForecaster:
    def __init__(self, order=(1,0,1), seasonal_order=(0,0,0,0), exog_cols=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols
        self.models = {}  # хранит модель на каждый item_id

    def train(self, df, item_id):
        train_df, test_df = train_test_split_series(df)
        model_fit = fit_arimax(train_df, exog_cols=self.exog_cols,
                               order=self.order, seasonal_order=self.seasonal_order)
        self.models[item_id] = model_fit
        return train_df, test_df, model_fit

    def evaluate(self, df, item_id):
        train_df, test_df = train_test_split_series(df)
        model_fit = self.models[item_id]
        forecast, _ = forecast_arimax(model_fit, test_df, exog_cols=self.exog_cols)
        metrics = {
            "MAPE": mape(test_df['cnt'], forecast),
            "RMSE": rmse(test_df['cnt'], forecast),
            "MAE": mae(test_df['cnt'], forecast),
            "SMAPE": smape(test_df['cnt'], forecast),
            "WAPE": wape(test_df['cnt'], forecast),
            "MASE": mase(test_df['cnt'], forecast, train_df['cnt']),
        }
        plot_forecast(train_df, test_df, forecast, title=f"Forecast for {item_id}")
        return metrics

    def forecast_future(self, model_fit, steps, exog_future=None):
        pred = model_fit.get_forecast(steps=steps, exog=exog_future)
        return pred.predicted_mean
