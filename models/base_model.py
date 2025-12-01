class BaseModel:
    def fit(self, series, **kwargs):
        raise NotImplementedError

    def predict(self, horizon, **kwargs):
        raise NotImplementedError
