import pandas as pd

def inspect_data(df):
    ###Convert time to datetime
    df["Time"] = pd.to_datetime(df["Time"])

    ###Sort data chronologically
    df = df.sort_values("Time").reset_index(drop=True)

    ###Check missinng values
    print("Missing values\n")
    print(df.isnull().sum())

    ###Check duplicate rows
    print("Duplicate Rows\n: ", df.duplicated().sum())

    ###Check data types
    print("Data types\n: ")
    print(df.dtypes)

    ###Check the first and last timestamps
    print("First timestamp\n: ", df["Time"].min())
    print("Last timestamp\n: ", df["Time"].max())

    ###Statistical summary
    print("Statistical summary\n: ")
    print(df[["Open", "High", "Low", "Close"]].describe())