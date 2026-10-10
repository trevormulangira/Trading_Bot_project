
def is_removing_duplicates_safe(df):
    ###Find completely identical rows
    duplicates = df[df.duplicated(keep=False)]

    print("Rows involved in duplicates:", len(duplicates))
    print(duplicates .sort_values("Time").head(20))

    ###Check whether timestamps repeat
    print("Duplicate timestamps:", df["Time"].duplicated().sum())

    ###Rows with repeated timestamps
    repeated = df[df["Time"].duplicated(keep=False)].sort_values("Time")

    ###Count timestamps that have different price/volume data
    differences = repeated.groupby("Time")[
        ["Open", "High", "Low", "Close", "Volume"]].nunique()

    print("Repeated timestamps:", len(repeated))
    print(
        "Timestamps with conflicting data:",
        (differences>1).any(axis=1).sum()
    )

    print(repeated.head(20))