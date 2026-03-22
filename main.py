from app.core.llm_interpreter import interpret_issue
from app.core.recommender import recommend_actions
from app.data.issue_detector import detect_issues
from app.data.profiler import profile_dataframe
from app.core.report import generate_report
from app.data.loader import load_dataset
from app.core.ranker import rank_issues


def run_pipeline(dataset_name: str):
    df = load_dataset(dataset_name)

    profile = profile_dataframe(df)
    issues = detect_issues(profile)

    all_issues = []

    for col in df.columns:
        column_profile = profile[col]
        column_issues = issues.get(col, [])

        if not column_issues:
            continue

        # Step 1: rule-based options
        options = recommend_actions(column_profile, column_issues)

        # Step 2: LLM explanation (NO decisions executed)
        analysis = interpret_issue(column_profile, column_issues, options)

        all_issues.append({
            "column": col,
            "issues": column_issues,
            "profile": column_profile,
            "options": options,
            "analysis": analysis
        })

    # Step 3: prioritize
    ranked = rank_issues(all_issues)

    # Step 4: generate report
    report = generate_report(ranked)

    return report