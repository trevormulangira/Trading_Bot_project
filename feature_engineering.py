import pandas as pd
def create_features(df):
    df = df.copy()
    ##Price returns
    df["return_1"] = df["Close"].pct_change() #percentage_change function

    ##Moving averages
    df["ma_5"] = df["Close"].rolling(window=5).mean() #rolling calculations
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    ###Volatility--Standard deviation of recent returns
    df["volatility_20"] = (
        df["return_1"].rolling(window=20).std()
    )

    ###Difference between current price and moving average
    df["price_vs_ma20"] = df["Close"] / df["ma_20"] - 1

    return df

