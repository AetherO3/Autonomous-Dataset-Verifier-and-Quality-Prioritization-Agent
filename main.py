from app.data.loader import load_dataset
from app.data.profiler import profile_dataframe
from app.data.issue_detector import detect_issues

from app.core.llm_engine import generate_llm_decision
from app.core.decision_layer import validate_decision

from app.actions import apply_action
from app.logger import log_operation


def run_pipeline(dataset_name: str):
    df = load_dataset(dataset_name)

    profile = profile_dataframe(df)
    issues = detect_issues(profile)

    for col in df.columns:
        column_profile = profile[col]
        column_issues = issues.get(col, [])

        decision = generate_llm_decision(column_profile, column_issues)
        validated_decision = validate_decision(decision)

        before = df[col].copy()

        df = apply_action(df, col, validated_decision)

        after = df[col] if col in df.columns else None

        log_operation(
            col=col,
            action=validated_decision["recommended_action"],
            before=before,
            after=after
        )

    return df


if __name__ == "__main__":
    # DATASET = "karkavelrajaj/amazon-sales-dataset"
    DATASET = "data/amazon.csv"

    final_df = run_pipeline(DATASET)

    print("\nFinal dataset preview:")
    print(final_df.head())