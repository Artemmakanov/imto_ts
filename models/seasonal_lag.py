from .base_model import BaseModel

class SeasonalLagModel(BaseModel):
    def __init__(self, lag=7):
        self.lag = lag
        self.values = None

    def fit(self, series):
        self.values = series

    def predict(self, horizon, **kwargs):
        preds = []
        for i in range(horizon):
            preds.append(self.values.iloc[-self.lag + (i % self.lag)])
        return preds
