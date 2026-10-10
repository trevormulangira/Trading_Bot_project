##Binary classification problem

def create_target(df):
    df = df.copy()

    #Next candles closing price
    df["next_close"] = df["Close"].shift(-1) #only used to create target. this column can't be used in training data;
                                             #it'll cause data leakage
    #shift(-1 or 1) moves data up(+) or down(-) by a specifies number of rows

    #1, if candle closes higher, 0 otherwise
    df["target"] = (
        df["next_close"] > df["Close"]
    ).astype(int)

    #Remove the last row; its next close isn't known. Re-assign the result
    df = df.iloc[:-1].copy()
    return df