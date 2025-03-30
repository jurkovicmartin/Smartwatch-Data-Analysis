import pandas as pd

# Measurements to perform
STATISTICAL_MEASURES = [
        "count", "min", "max", "mean",
        # 25th percentile
        lambda x: x.quantile(0.25),
        "median",
        # 75th percentile
        lambda x: x.quantile(0.75),
        "var", "std", "skew", "kurt"
    ]
# Names to display
STATISTICAL_MEASURES_NAMES = [
        "Count", "Minimum", "Maximum", "Mean", "25th percentile", "Median",
        "75th percentile", "Variance", "Standard Deviation", "Skewness", "ExcessKurtosis"
    ]


def statistical_analysis(df: pd.DataFrame, drop: list[str] = None) -> pd.DataFrame:
    """Performs statistical measurements on dataframe.

    Args:
        df (pd.DataFrame): input dataframe
        drop (list[str], optional): features to drop (not include in the analysis). Defaults to None.

    Returns:
        pd.DataFrame: measurements
    """
    if drop:
        measurements = df.drop(columns=drop).agg(STATISTICAL_MEASURES)
    else:
        measurements = df.agg(STATISTICAL_MEASURES)
    # Rename indexes with more readable names
    measurements.index = STATISTICAL_MEASURES_NAMES
    return measurements