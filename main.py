from app.vision.quality_checker import check_image_text_consistency
from app.dataProcessor.issue_detector import detect_issues
from app.core.relation_analyzer import analyze_relations
from app.dataProcessor.profiler import profile_dataframe
from app.core.llm_interpreter import interpret_issue
from app.core.recommender import recommend_actions
from app.dataProcessor.loader import load_dataset
from app.core.config import RATE_LIMIT_SLEEP
from app.core.report import generate_report
from app.core.applier import apply_actions
from app.core.ranker import rank_issues
from app.logger import get_logs
import google.genai as genai
import creds
import time


def run_pipeline(dataset_name: str):
    df = load_dataset(dataset_name)
    client = genai.Client(api_key = creds.gemini_key)

    profile = profile_dataframe(df)
    relations = analyze_relations(client, profile)
    issues = detect_issues(profile, df)

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
            "analysis": analysis,
        })

        time.sleep(RATE_LIMIT_SLEEP)

    ranked = rank_issues(all_issues)
    report = generate_report(ranked, relations)

    cleaned_df, original_df, flagged = apply_actions(df, report)

    vision_results = check_image_text_consistency(df)
    
    if vision_results:
        print("\nVision Quality Check")
        for v in vision_results:
            flag = "Flagged" if v["flagged"] else "Okay"
            print(f"  {v['image_column']} <-> {v['text_column']} | score: {v['semantic_match_score']} | {flag}")

    for entry in get_logs():
        print(f"[LOG] {entry['column']} | {entry['action']} | {entry['timestamp']}")

    cleaned_df, original_df, flagged = apply_actions(df, report)

    if flagged:
        print("\nFlagged : low confidence")
        for f in flagged:
            print(f"  {f['column']} | {f['action']} | confidence: {f['confidence']:.2f}")

    return report, cleaned_df, original_df


report, cleaned_df, original_df = run_pipeline("data/amazon.csv")

for idx, r in enumerate(report["issues"]):
    print(f"\n{idx + 1} - Column: {r['column']}")
    print(f"   Issues: {r['issues']}")
    print(f"   Action: {r['analysis']['recommended_option']} (confidence: {r['analysis']['confidence']})")

print(f"\nOriginal shape: {original_df.shape}")
print(f"Cleaned shape:  {cleaned_df.shape}")