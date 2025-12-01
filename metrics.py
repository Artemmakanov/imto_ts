import numpy as np

def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )

def wape(y_true, y_pred, eps=1e-9):
    return np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + eps)

def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def mase(y_true, y_pred, y_train):
    naive_diff = np.mean(np.abs(np.diff(y_train)))
    if naive_diff == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / naive_diff

def mape(y_true, y_pred, eps=1e-9):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps)))

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))