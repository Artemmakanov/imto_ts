import pandas as pd

def make_time_splits(df, horizon):
    """
    horizon: number of days: 7, 30, 90.
    
    Returns:
        train_df, val_df, test_df
    """

    max_date = df['date'].max()

    test_start = max_date - pd.Timedelta(days=horizon) + pd.Timedelta(days=1)
    val_start  = test_start - pd.Timedelta(days=horizon)

    train_df = df[df['date'] < val_start]
    val_df   = df[(df['date'] >= val_start) & (df['date'] < test_start)]
    test_df  = df[df['date'] >= test_start]

    return train_df, val_df, test_df
