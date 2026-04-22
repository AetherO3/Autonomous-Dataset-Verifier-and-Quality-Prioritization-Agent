import pandas as pd
from app.logger import log_operation
from app.core.config import CONFIDENCE_THRESHOLD, FLAG_THRESHOLD

ID_ISSUES = {"id_like_column", "constant_column"}


def is_relation_sensitive(col, issues, relations):
    # ID-like and constant columns should always be droppable
    # regardless of what the relation analyser found
    if set(issues) & ID_ISSUES:
        return False
    for r in relations or []:
        if col in [r["col_a"], r["col_b"]]:
            if r["relation"] == "correlated":
                return True
    return False


def apply_actions(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    original_df = df.copy()
    cleaned_df = df.copy()
    flagged = []

    for issue in report["issues"]:
        col = issue["column"]
        action = issue["analysis"]["recommended_option"]
        confidence = issue["analysis"].get("confidence", 0)
        col_issues = issue.get("issues", [])

        if col not in cleaned_df.columns:
            continue

        if confidence < FLAG_THRESHOLD:
            log_operation(col, "skipped", cleaned_df[col], None)
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            flagged.append({
                "column": col,
                "action": action,
                "confidence": confidence,
                "reason": issue["analysis"].get("explanation", ""),
            })
            log_operation(col, "flagged", cleaned_df[col], None)
            continue

        before = cleaned_df[col].copy()

        if action == "drop_column":
            if is_relation_sensitive(col, col_issues, report.get("relations")):
                flagged.append({
                    "column": col,
                    "action": action,
                    "confidence": confidence,
                    "reason": "Drop blocked: correlated feature",
                })
                log_operation(col, "blocked_drop_relation", before, None)
                continue

            cleaned_df.drop(columns=[col], inplace=True)
            log_operation(col, action, before, None)
            continue

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
            cleaned_df[col] = cleaned_df[col].where(cleaned_df[col].isin(top), other="other")
        elif action == "leave":
            pass

        log_operation(col, action, before, cleaned_df[col])

    return cleaned_df, original_df, flagged