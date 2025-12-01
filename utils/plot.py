import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(
    df,
    forecasts, 
    item_id, 
    horizon, 
    last_k_days: int | None = None,
    holiday_cols: list | None = None
):
    """
    df: DataFrame с колонками ['item_id','date','cnt']
    forecasts: dict item_id -> list of predictions
    item_id: int
    horizon: int, количество дней для прогноза
    """
    # реальные данные
    series = df[df['item_id'] == item_id].sort_values('date')

    if last_k_days:
        series = series[-last_k_days:]
    
    # Приводим даты к pd.Timestamp
    dates = pd.to_datetime(series['date'])
    actual = series['cnt'].values

    # прогноз
    if item_id not in forecasts:
        raise ValueError(f"No forecasts for item_id={item_id}")

    preds = forecasts[item_id]

    # формируем даты для прогноза
    last_date = dates.iloc[-horizon]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    # график
    plt.figure(figsize=(10,5))
    plt.plot(dates, actual, label="Actual", marker='o')
    plt.plot(forecast_dates, preds, label="Forecast", marker='x')

    # вертикальная линия
    if holiday_cols:
        for col in holiday_cols:
            for vline in series[series[col] == 1].date:
                vline = pd.to_datetime(vline)
                plt.axvline(x=vline, color='red', linestyle='--')

    plt.title(f"Forecast vs Actual for item_id={item_id}")
    plt.xlabel("Date")
    plt.ylabel("cnt")
    plt.legend()
    plt.grid(True)
    plt.show()
