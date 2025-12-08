# src/ml_model.py
import os
import joblib
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

from .utils import reindex_item, make_lags, make_rollings, time_train_test_split, rmse, mape, wape

class MLForecaster:
    """
    Feature-based forecaster trained on all items together.
    Uses LightGBM by default.
    """

    def __init__(self, freq: str = "D", lgb_params: Optional[dict] = None, lags: Optional[List[int]] = None):
        self.freq = freq
        default = {"n_estimators": 1000, "learning_rate": 0.05, "random_state": 42}
        self.lgb_params = {**default, **(lgb_params or {})}
        self.model = LGBMRegressor(**self.lgb_params)
        self.lags = lags or [1,7,14,28]
        self.features: List[str] = []
        self.trained = False

    def preprocess_all(self, df: pd.DataFrame, exog_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Align all items to freq and return single dataframe with item_id as column + date index.
        """
        dfs = []
        for item in df["item_id"].unique():
            df_item = df[df["item_id"] == item].copy()
            df_item = reindex_item(df_item, date_col="date", freq=self.freq, fill_sales=0.0, exog_cols=exog_cols)
            df_item["item_id"] = item
            dfs.append(df_item)
        all_df = pd.concat(dfs).reset_index().rename(columns={"index": "date"}).set_index("date").sort_index()
        return all_df

    def make_features(self, df_all: pd.DataFrame, exog_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        df_all indexed by date and contains item_id, sales, exog columns.
        Returns features for supervised learning.
        """
        df = df_all.copy()
        # create item_id categorical
        df["item_code"] = df["item_id"].astype("category").cat.codes
        # lag & rolling per item
        out = []
        for item in df["item_id"].unique():
            dfi = df[df["item_id"] == item].copy()
            dfi = make_lags(dfi, target_col="sales", lags=self.lags)
            dfi = make_rollings(dfi, target_col="sales", windows=[7,28,90])
            # calendar
            dfi["dow"] = dfi.index.dayofweek
            dfi["month"] = dfi.index.month
            # exog pass-through
            if exog_cols:
                for c in exog_cols:
                    if c not in dfi.columns:
                        dfi[c] = 0.0
            out.append(dfi)
        df_feat = pd.concat(out).sort_index()
        # drop rows with NaN in lag features
        lag_cols = [f"sales_lag_{l}" for l in self.lags]
        df_feat = df_feat.dropna(subset=lag_cols)
        # define feature set
        feat_cols = []
        feat_cols += lag_cols
        feat_cols += [c for c in df_feat.columns if c.startswith("sales_rmean_") or c.startswith("sales_rstd_")]
        feat_cols += ["item_code", "dow", "month"]
        if exog_cols:
            feat_cols += exog_cols
        self.features = feat_cols
        return df_feat

    def train(self, df: pd.DataFrame, exog_cols: Optional[List[str]] = None, test_size_days: int = 28):
        df_all = self.preprocess_all(df, exog_cols=exog_cols)
        df_feat = self.make_features(df_all, exog_cols=exog_cols)
        # split by time (global last N days as test for all items)
        df_feat = df_feat.sort_index()
        train, test = time_train_test_split(df_feat, test_size_days=test_size_days)
        X_train = train[self.features]
        y_train = train["sales"]
        X_test = test[self.features]
        y_test = test["sales"]
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
        self.trained = True
        preds = self.model.predict(X_test)
        return {"rmse": rmse(y_test.values, preds), "mape": mape(y_test.values, preds), "wape": wape(y_test.values, preds)}

    def predict_df(self, df_feat: pd.DataFrame) -> pd.Series:
        if not self.trained:
            raise ValueError("model not trained")
        return pd.Series(self.model.predict(df_feat[self.features]), index=df_feat.index)

    def recursive_forecast(self, history_df: pd.DataFrame, item_id: str, steps: int, exog_future: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        history_df: preprocessed per-item dataframe (indexed by date) with sales and exog present
        returns pd.Series indexed by future dates
        """
        hist = history_df.copy()
        last_date = hist.index.max()
        future_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(self.freq), periods=steps, freq=self.freq)
        preds = []
        for d in future_index:
            # build a row with required features
            row = {}
            # compute lags from hist
            for lag in self.lags:
                val = hist["sales"].shift(lag).iloc[-1] if len(hist) >= lag else hist["sales"].iloc[-1]
                # alternative: use last known value
                row[f"sales_lag_{lag}"] = val
            # rolling features
            for w in [7,28,90]:
                row[f"sales_rmean_{w}"] = hist["sales"].iloc[-min(len(hist), w):].mean()
                row[f"sales_rstd_{w}"] = hist["sales"].iloc[-min(len(hist), w):].std() if len(hist) >= 2 else 0.0
            row["item_code"] = hist["item_id"].astype("category").cat.codes.iloc[0] if "item_id" in hist.columns else 0
            row["dow"] = d.dayofweek
            row["month"] = d.month
            # exog
            if exog_future is not None and d in exog_future.index:
                for c in exog_future.columns:
                    row[c] = exog_future.loc[d, c]
            # convert to DataFrame
            row_df = pd.DataFrame([row], index=[d])
            # ensure all features present
            for f in self.features:
                if f not in row_df.columns:
                    row_df[f] = 0.0
            val = float(self.model.predict(row_df[self.features])[0])
            preds.append(val)
            # append to history to create autoregressive lags
            newrow = {"sales": val, "item_id": hist["item_id"].iloc[0]} if "item_id" in hist.columns else {"sales": val}
            if exog_future is not None and d in exog_future.index:
                for c in exog_future.columns:
                    newrow[c] = exog_future.loc[d, c]
            hist = pd.concat([hist, pd.DataFrame(newrow, index=[d])])
        return pd.Series(preds, index=future_index, name="pred")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features, "params": self.lgb_params}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.features = obj["features"]
        self.lgb_params = obj.get("params", self.lgb_params)
        self.trained = True
