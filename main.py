from app.core.llm_interpreter import interpret_issue
from app.core.recommender import recommend_actions
from app.data.issue_detector import detect_issues
from app.data.profiler import profile_dataframe
from app.core.report import generate_report
from app.data.loader import load_dataset
from app.core.ranker import rank_issues
from openai import OpenAI


def run_pipeline(dataset_name: str):
    df = load_dataset(dataset_name)
    client = OpenAI(api_key="key")

    profile = profile_dataframe(df)

    issues = detect_issues(profile)

    all_issues = []

    for col in df.columns:
        column_profile = profile[col]
        column_issues = issues.get(col, [])

        if not column_issues:
            continue

        options = recommend_actions(column_profile, column_issues)

        analysis = interpret_issue(client, column_profile, column_issues, options)

        all_issues.append({
            "column": col,
            "issues": column_issues,
            "profile": column_profile,
            "options": options,
            "analysis": analysis
        })

    ranked = rank_issues(all_issues)

    report = generate_report(ranked)

    return report

run_pipeline("data/amazon.csv")