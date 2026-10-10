def remove_duplicates(df):
    df = df.drop_duplicates().reset_index(drop=True)

    ###Keep one row per timestamp
    df = df.drop_duplicates(subset="Time", keep="first")
    df = df.reset_index(drop=True)

    ###Verify the cleaned data
    print("Shape  after cleaning:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate timestamps:", df["Time"].duplicated().sum())

    return df