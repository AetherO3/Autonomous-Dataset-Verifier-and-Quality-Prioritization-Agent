from app.core.llm_interpreter import interpret_issues_batch, safe_fallback
from app.core.recommender import recommend_actions
from app.dataProcessor.issue_detector import detect_issues
from app.dataProcessor.profiler import profile_dataframe
from app.core.report import generate_report
from app.dataProcessor.loader import load_dataset
from app.core.ranker import rank_issues
from app.core.applier import apply_actions
from app.core.relation_analyzer import analyze_relations
from app.logger import get_logs
import google.genai as genai
import time
import creds
import os
import json

RELATIONS_CACHE = "data/relations_cache.json"


def run_pipeline(dataset_name: str):
    df = load_dataset(dataset_name)
    client = genai.Client(api_key = creds.gemini_key)

    profile = profile_dataframe(df)

    # relations = analyze_relations(client, profile)
    if os.path.exists(RELATIONS_CACHE):
        with open(RELATIONS_CACHE) as f:
            relations = json.load(f)
    else:
        relations = analyze_relations(client, profile)
        with open(RELATIONS_CACHE, "w") as f:
            json.dump(relations, f)

    time.sleep(65)

    issues = detect_issues(profile, df)

    issues_input = []
    for col in df.columns:
        column_profile = profile[col]
        column_issues = issues.get(col, [])
        if not column_issues:
            continue
        options = recommend_actions(column_profile, column_issues)
        issues_input.append({
            "column": col,
            "profile": column_profile,
            "issues": column_issues,
            "options": options,
        })

    batch_results = interpret_issues_batch(client, issues_input)

    all_issues = []
    for item in issues_input:
        col = item["column"]
        analysis = batch_results.get(
            col, safe_fallback(item["profile"], item["issues"], item["options"])
        )
        all_issues.append({
            "column": col,
            "issues": item["issues"],
            "profile": item["profile"],
            "options": item["options"],
            "analysis": analysis,
        })

    ranked = rank_issues(all_issues)
    report = generate_report(ranked, relations)
    cleaned_df, original_df, flagged = apply_actions(df, report)

    if relations:
        print("\n Semantic Relations ")
        for r in relations:
            print(f"  [{r['relation']}] {r['col_a']} <-> {r['col_b']} | {r['suggestion']} (conf: {r['confidence']})")

    print("\n Column Issues ")
    for idx, r in enumerate(report["issues"]):
        conf = r['analysis']['confidence']
        action = r['analysis']['recommended_option']
        status = "APPLY" if conf >= 0.7 else "FLAG" if conf >= 0.5 else "SKIP"
        print(f"{idx+1} - Column: {r['column']}")
        print(f"   Issues: {r['issues']}")
        print(f"   Action: {action} (confidence: {conf}) → {status}")

    if flagged:
        print("\n Flagged (low confidence) ")
        for f in flagged:
            print(f"  {f['column']} | {f['action']} | confidence: {f['confidence']:.2f}")

    print(f"\nOriginal shape: {original_df.shape}")
    print(f"Cleaned shape:  {cleaned_df.shape}")

    print("\n Logs ")
    for entry in get_logs():
        print(f"  [{entry['column']}] {entry['action']} @ {entry['timestamp']}")

    return report, cleaned_df, original_df, flagged


if __name__ == "__main__":
    report, cleaned_df, original_df, flagged = run_pipeline("data/amazon.csv")

os.makedirs("data", exist_ok = True)

cleaned_df.to_csv("data/cleaned.csv", index=False)

cleaning_log = get_logs()
with open("data/cleaning_log.json", "w") as f:
    json.dump(cleaning_log, f, indent=2)

print("\nSaved: data/Cleaned.csv")
print(f"\nSaved: data/Cleaning Log.json")