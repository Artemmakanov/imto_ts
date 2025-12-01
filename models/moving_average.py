from .base_model import BaseModel

class MovingAverageModel(BaseModel):
    def __init__(self, window=7):
        self.window = window
        self.mean = None

    def fit(self, series):
        self.mean = series.tail(self.window).mean()

    def predict(self, horizon, **kwargs):
        return [self.mean] * horizon
