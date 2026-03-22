import pandas as pd

def apply_action(df: pd.DataFrame, col: str, decision: dict) -> pd.DataFrame:
    action = decision["recommended_action"]

    if action == "impute":
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    elif action == "drop":
        df.drop(columns=[col], inplace=True)

    elif action == "convert":
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    elif action == "flag":
        # no modification
        pass

    elif action == "ignore":
        pass

    return df