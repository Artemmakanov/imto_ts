import pandas as pd
from forecaster import Forecaster

class MultiItemForecaster:
    def __init__(self, model_name="ma"):
        self.model_name = model_name
        self.models = {}  # ключ = item_id, значение = Forecaster()
        self.metrics_per_item = {}

    def fit(self, df, exog_cols=None, **model_kwargs):
        """
        df: DataFrame с колонками item_id, date, cnt
        """

        for item_id, group in df.groupby("item_id"):
            group = group.sort_values("date")
            series = group["cnt"]
            if exog_cols:
                exog = group[exog_cols]
            else:
                exog = None

            model = Forecaster(model_name=self.model_name, **model_kwargs)
            model.fit(series, exog=exog)
            self.models[item_id] = model

    def predict(self, horizon, test=None, exog_cols=None):
        """
        Возвращает прогноз для всех товаров в виде dict: item_id -> forecast list
        """
        forecasts = {}
        for item_id, model in self.models.items():
            if exog_cols:
                exog = test[test.item_id == item_id]
                exog = exog[exog_cols][-horizon:]
            else:
                exog = None

            forecasts[item_id] = model.predict(horizon, exog=exog)
        return forecasts

    def evaluate(self, df, horizon, exog_cols=None):
        """
        df: DataFrame с колонками item_id, date, cnt
        horizon: сколько последних точек использовать для теста

        Возвращает:
          - metrics_per_item: dict item_id -> metrics dict
          - metrics_aggregated: dict с усреднёнными метриками
        """
        self.metrics_per_item = {}
        for item_id, group in df.groupby("item_id"):
            group = group.sort_values("date")
            series = group["cnt"]
            if exog_cols:
                exog = group[exog_cols]
            else:
                exog = None
            model = self.models[item_id]
            metrics, _ = model.evaluate(series, horizon, exog)
            self.metrics_per_item[item_id] = metrics

        # агрегируем метрики (среднее по всем товарам)
        agg_metrics = {}
        for key in list(next(iter(self.metrics_per_item.values())).keys()):
            agg_metrics[key] = sum([m[key] for m in self.metrics_per_item.values()]) / len(self.metrics_per_item)
        
        return self.metrics_per_item, agg_metrics
