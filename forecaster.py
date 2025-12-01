import pandas as pd
from models.naive import NaiveModel
from models.moving_average import MovingAverageModel
from models.seasonal_lag import SeasonalLagModel
from evaluation.evaluator import evaluate


class Forecaster:
    def __init__(self, model_name="naive", **kwargs):
        self.model_name = model_name
        self.model = None  # объект модели после fit
        self.model_kwargs = kwargs

    def fit(self, series, exog=None):
        if self.model_name == "ml":
            from models.ml import MLForecaster
            self.model = MLForecaster(**self.model_kwargs)
        elif self.model_name == "sarimax":
            from models.sarimax import SARIMAX
            self.model = SARIMAX(**self.model_kwargs)
        elif self.model_name == "arima":
            from models.arma_arima import ARIMAModel
            self.model = ARIMAModel(**self.model_kwargs)
        elif self.model_name == "naive":
            from models.naive import NaiveModel
            self.model = NaiveModel(**self.model_kwargs)
        elif self.model_name == "ma":
            from models.moving_average import MovingAverageModel
            self.model = MovingAverageModel(**self.model_kwargs)
        elif self.model_name == "seasonal_lag":
            from models.seasonal_lag import SeasonalLagModel
            self.model = SeasonalLagModel(**self.model_kwargs)
        elif self.model_name == "ets":
            from models.ets import ETSModel
            self.model = ETSModel(**self.model_kwargs)
        else:
            raise ValueError("Unknown model")

        self.model.fit(series, exog=exog)

    def predict(self, horizon, exog=None):
        if self.model is None:
            raise ValueError("Call fit() before predict()")
        return self.model.predict(horizon, exog=exog)

    def evaluate(self, series, horizon, exog=None):
        from evaluation.evaluator import evaluate
        train = series.iloc[:-horizon]
        test = series.iloc[-horizon:]
        if exog:
            exog_train = exog.iloc[:-horizon]
            exog_test = exog.iloc[-horizon:]
        else:
            exog_train = None
            exog_test = None

        # обучаем модель на train
        self.fit(train, exog_train)
        preds = self.predict(horizon, exog_test)

        metrics = evaluate(test.values, preds, train.values)
        return metrics, preds
