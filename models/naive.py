from .base_model import BaseModel

class NaiveModel(BaseModel):
    def __init__(self):
        self.last_value = None

    def fit(self, series):
        self.last_value = series.iloc[-1]

    def predict(self, horizon, **kwargs):
        return [self.last_value] * horizon
