import pandas as pd

def make_hashable(x):
    if isinstance(x, list):
        return tuple(make_hashable(i) for i in x)
    if isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    if isinstance(x, set):
        return tuple(sorted(make_hashable(i) for i in x))
    return x


def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {}

    for col in df.columns:
        series = df[col]
        total = len(series)

        has_unhashable = series.apply(lambda x: isinstance(x, (list, dict, set))).any()

        if has_unhashable:
            unique = int(series.apply(make_hashable).nunique())
            column_type = "nested"
        else:
            unique = int(series.nunique())
            column_type = "scalar"

        unique_ratio = unique / total if total else 0.0

        profile[col] = {
            "dtype": str(series.dtype),
            "column_type": column_type,
            "count": total,
            "null_perc": float(series.isna().mean()),
            "unique": unique,
            "unique_ratio": float(unique_ratio),
            "sample": series.dropna().head(5).tolist()
        }

    return profile