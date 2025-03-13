import pandas as pd


def clear_dataset():
    pd.set_option('future.no_silent_downcasting', True)

    df = pd.read_csv("data/smartwatch.csv")
    original_length = len(df)
    # print(df)

    ### DATA CLEAR
    print("***DATA CLEARING***\n")
    # Number of incomplete records (rows with at least one NaN)
    incomplete_rows = df.isna().any(axis=1).sum()
    print(f"Incomplete records: {incomplete_rows}\n")
    # Drop incomplete records
    df = df.dropna()
    

    ### Activity Level column (with strings)
    num_unique = df["Activity Level"].nunique()
    print(f"Number of unique values in Activity Level column: {num_unique}")
    val_unique = df["Activity Level"].unique()
    print(f"Unique values of Activity Level column: {val_unique}\n")

    # Replacing Activity levels with numerical values
    df["Activity Level"] = df["Activity Level"].replace(["Highly Active", "Highly_Active"], 3)
    df["Activity Level"] = df["Activity Level"].replace(["Active", "Actve"], 2)
    df["Activity Level"] = df["Activity Level"].replace(["Seddentary", "Sedentary"], 1)

    # Checking if all other values are numeric
    is_numeric = df.map(lambda x: pd.to_numeric(x, errors='coerce')).notna().all().all()
    print(f"All values are numeric: {is_numeric}")
    if not is_numeric:
        # Columns which contain non numeric values
        non_numeric_cols = df.columns[~df.map(lambda x: isinstance(x, (int, float))).all()]
        print(f"Columns with non-numeric values: {list(non_numeric_cols)}\n")


    ### Stress Level column (limited value range)
    num_unique = df["Stress Level"].nunique()
    print(f"Number of unique values in Stress Level column: {num_unique}")
    val_unique = df["Stress Level"].unique()
    print(f"Unique values of Stress Level column: {val_unique}\n")

    # Assuming that Very high could relate to value 8
    df["Stress Level"] = df["Stress Level"].replace("Very High", 8)
    # Other string values contains numbers => replace them with actual numbers
    df["Stress Level"] = df["Stress Level"].apply(pd.to_numeric)


    ### Sleep Duration column (spectrum as a range)
    non_numeric = df["Sleep Duration (hours)"][~df["Sleep Duration (hours)"].apply(lambda x: isinstance(x, (int, float)))].unique()
    print(f"Non-numeric values in Sleep Duration column: {non_numeric}")
    # Replace strings numbers as actual numbers and use NaN where conversion isn't success
    df["Sleep Duration (hours)"] = df["Sleep Duration (hours)"].apply(pd.to_numeric, errors="coerce")
    # Check if there are some NaN now
    has_nan = df["Sleep Duration (hours)"].isna().any()
    print(f"Sleep Duration column has NaN values: {has_nan}")
    if has_nan:
        nan_count = df["Sleep Duration (hours)"].isna().sum()
        print(f"Number of NaN values: {nan_count}\n")
        # Drop these records with NaN
        df = df.dropna()


    ### User ID column
    ids_unique = df["User ID"].is_unique
    print(f"All IDs are unique : {ids_unique}")
    if not ids_unique:
        num_unique = df["User ID"].nunique()
        print(f"Number of unique values in User ID column: {num_unique}\n")
        # Replace the column with unique ID values
        df["User ID"] = range(len(df))
        ids_unique = df["User ID"].is_unique


    ### Summary of data clearing
    print("***SUMMARY OF DATA CLEARING***\n")
    print(f"Original number of records: {original_length}")
    print(f"Number of records after preprocessing: {len(df)}")
    is_numeric = df.map(lambda x: pd.to_numeric(x, errors='coerce')).notna().all().all()
    print(f"All values are numeric: {is_numeric}")
    print(f"All IDs are unique : {ids_unique}")

    df["User ID"] = df["User ID"].astype("Int64") 
    df.to_csv("data/smartwatch_cleared.csv", index=False)
