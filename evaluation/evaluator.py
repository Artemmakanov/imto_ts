from metrics import *

def evaluate(y_true, y_pred, y_train):
    return {
        "SMAPE": smape(y_true, y_pred),
        "WAPE":  wape(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MASE":  mase(y_true, y_pred, y_train),
        "MAE":  mae(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred)
    }
