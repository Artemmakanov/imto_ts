# forecasting_module/models/ml_forecaster.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from lightgbm import LGBMRegressor
from .base_model import BaseModel

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class MLForecaster:
    def __init__(self, lags=[1,7], trend_window=7):
        self.lags = lags
        self.model = LGBMRegressor()
        self.trend_window = trend_window
        self.last_trend = None
        self.series = None

    def _compute_trend(self, df):
        df['trend'] = df['cnt'].rolling(self.trend_window, min_periods=1).mean()
        return df

    def _detrend(self, df):
        df['detrended'] = df['cnt'] - df['trend']
        return df

    def _create_features(self, X):
        X = X.copy()
        for lag in self.lags:
            X[f'lag{lag}'] = X['detrended'].shift(lag)
        X = X.drop(columns=['cnt','trend','detrended'])
        return X

    def fit(self, series, exog=None):
        df = exog.copy()
        df['cnt'] = series
        df = self._compute_trend(df)
        df = self._detrend(df)

        self.last_trend = df['trend'].iloc[-1]
        self.series = df
        y = df['detrended']
        X = self._create_features(df)
        self.model.fit(X, y)

    def predict(self, horizon, exog=None):
        preds = []
        last_values = self.series.copy()

        for step in range(horizon):
            X_step = {}

            for lag in self.lags:
                X_step[f'lag{lag}'] = last_values['detrended'].iloc[-lag]

            if exog is not None:
                for col in exog.columns:
                    X_step[col] = exog[col].iloc[step]

            X_step = pd.DataFrame([X_step])
            detrended_pred = self.model.predict(X_step)[0]
            preds.append(detrended_pred)

            new_trend = self.last_trend
            new_cnt = detrended_pred + new_trend

            new_row = {
                'cnt': new_cnt,
                'trend': new_trend,
                'detrended': detrended_pred
            }
            last_values = pd.concat([last_values, pd.DataFrame([new_row])], ignore_index=True)

        restored = np.array(preds) + self.last_trend
        return restored
