from ydata_profiling import ProfileReport
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    report = ProfileReport(df, minimal=True, progress_bar=False)
    desc = report.get_description()

    profile = {}

    for col in df.columns:
        col_desc = desc.variables[col]
        series = df[col]
        total = len(series)

        has_unhashable = series.apply(lambda x: isinstance(x, (list, dict, set))).any()

        if has_unhashable:
            unique = int(series.apply(make_hashable).nunique())
            column_type = "nested"
        else:
            unique = int(series.nunique())
            column_type = get_column_type(col_desc)

        null_perc = float(series.isna().mean())
        unique_ratio = unique / total if total else 0.0

        profile[col] = {
            "dtype": str(series.dtype),
            "column_type": column_type,
            "count": total,
            "null_perc": null_perc,
            "unique": unique,
            "unique_ratio": unique_ratio,
            "mean": col_desc.get("mean") if column_type == "numeric" else None,
            "std": col_desc.get("std") if column_type == "numeric" else None,
            "skewness": col_desc.get("skewness") if column_type == "numeric" else None,
            "kurtosis": col_desc.get("kurtosis") if column_type == "numeric" else None,
            "n_zeros": col_desc.get("n_zeros") if column_type == "numeric" else None,
            "sample": safe_sample(series),
        }

    return profile


def get_column_type(col_desc: dict) -> str:
    vtype = str(col_desc.get("type", "")).lower()
    if "datetime" in vtype:
        return "datetime"
    if "numeric" in vtype or "real" in vtype:
        return "numeric"
    if "boolean" in vtype or "bool" in vtype:
        return "boolean"
    if "categorical" in vtype or "text" in vtype:
        return "categorical"
    if "unsupported" in vtype:
        return "nested"
    return "scalar"


def make_hashable(x):
    if isinstance(x, list):
        return tuple(make_hashable(i) for i in x)
    if isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    if isinstance(x, set):
        return tuple(sorted(make_hashable(i) for i in x))
    return x


def safe_sample(series: pd.Series) -> list:
    sample = series.dropna().head(5).tolist()
    return [str(s) if not isinstance(s, (int, float, bool)) else s for s in sample]