from .base_model import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

class ETSModel(BaseModel):
    def __init__(self, **kwargs):
        """
        seasonal_periods: длина сезонного цикла (например 7 для недели)
        trend: 'add' или 'mul'
        seasonal: 'add' или 'mul' или None
        """
        self.kwargs = kwargs
        self.model_fit = None
        warnings.filterwarnings("ignore")  # подавляем предупреждения statsmodels

    def fit(self, series, **kwargs):
        self.model_fit = ExponentialSmoothing(
            series,
            initialization_method="estimated",
            **self.kwargs
        ).fit(
            smoothing_level=0.05,
            smoothing_slope=0.1,
            smoothing_seasonal=0.3
        )

    def predict(self, horizon, **kwargs):
        if self.model_fit is None:
            raise ValueError("Call fit() before predict()")
        return self.model_fit.forecast(horizon).tolist()
