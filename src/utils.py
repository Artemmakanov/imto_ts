import matplotlib.pyplot as plt


def plot_forecast(train_df, test_df, forecast, title="Forecast", margin: int=30):
    left = len(train_df['date']) - 1 - margin
    right = len(train_df['date']) - 1 + len(test_df['date']) - 1 

    plt.figure(figsize=(10,5))
    plt.xlim(left, right)
    plt.xticks(rotation=90)
    plt.plot(train_df['date'], train_df['cnt'], label='Train')
    plt.plot(test_df['date'], test_df['cnt'], label='Test', color='orange')
    plt.plot(test_df['date'], forecast, label='Forecast', color='green')
    plt.legend()
    plt.title(title)
    plt.show()

def train_test_split_series(df, test_days=28):
    df = df.sort_values("date")
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test
