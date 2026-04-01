import pandas as pd


def apply_actions(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_df = df.copy()
    cleaned_df = df.copy()

    for issue in report["issues"]:
        col = issue["column"]
        action = issue["analysis"]["recommended_option"]

        if col not in cleaned_df.columns:
            continue

        if action == "drop_column":
            cleaned_df.drop(columns=[col], inplace=True)

        elif action == "fill_mean":
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())

        elif action == "fill_median":
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

        elif action == "fill_mode":
            mode = cleaned_df[col].mode()
            if not mode.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode[0])

        elif action == "drop_rows":
            cleaned_df.dropna(subset=[col], inplace=True)

        elif action == "encode":
            cleaned_df[col] = pd.factorize(cleaned_df[col])[0]

        elif action == "group_categories":
            top = cleaned_df[col].value_counts().nlargest(20).index
            cleaned_df[col] = cleaned_df[col].where(
                cleaned_df[col].isin(top), other="other"
            )

        elif action == "leave":
            pass

    return cleaned_df, original_df