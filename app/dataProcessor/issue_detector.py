import great_expectations as gx
import pandas as pd


def detect_issues(profile: dict, df: pd.DataFrame) -> dict:
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("pandas_source")
    asset = datasource.add_dataframe_asset("dataset")
    batch = asset.add_batch_definition_whole_dataframe("batch").get_batch(
        batch_parameters={"dataframe": df}
    )

    issues = {}

    for col, stats in profile.items():
        col_issues = []

        if stats["null_perc"] > 0.3:
            result = batch.expect_column_values_to_not_be_null(column=col)
            if not result.success:
                col_issues.append("high_missing")

        if stats["unique"] == 1:
            col_issues.append("constant_column")
        elif stats["unique"] > 1 and stats["unique_ratio"] < 0.01:
            col_issues.append("near_constant")

        if stats["unique_ratio"] > 0.9 and stats["column_type"] not in ("datetime", "numeric"):
            col_issues.append("high_cardinality")

        if stats["unique_ratio"] >= 0.999 and stats["column_type"] not in ("datetime", "numeric"):
            col_issues.append("id_like_column")

        if stats["column_type"] == "nested":
            col_issues.append("nested_data")

        if stats["column_type"] == "numeric":
            if stats.get("skewness") and abs(stats["skewness"]) > 2:
                col_issues.append("high_skewness")
            if stats.get("n_zeros") and stats["count"] and stats["n_zeros"] / stats["count"] > 0.5:
                col_issues.append("high_zeros")

        if col_issues:
            issues[col] = col_issues

    return issues