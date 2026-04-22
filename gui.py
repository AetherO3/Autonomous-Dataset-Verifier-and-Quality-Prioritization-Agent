import streamlit as st
import pandas as pd
import json
import os
import tempfile
import google.genai as genai
import creds
from dotenv import load_dotenv
from app.core.llm_interpreter import interpret_issues_batch, safe_fallback
from app.core.recommender import recommend_actions
from app.dataProcessor.issue_detector import detect_issues
from app.dataProcessor.profiler import profile_dataframe
from app.core.report import generate_report
from app.dataProcessor.loader import load_dataset
from app.core.ranker import rank_issues
from app.core.applier import apply_actions
from app.core.relation_analyzer import analyze_relations
from app.logger import get_logs, log_store

st.set_page_config(page_title = "Dataset Verifier", layout = "wide")
st.title("Dataset Verifier")

RELATIONS_CACHE = "data/relations_cache.json"
SUPPORTED = ["csv", "json", "jsonl", "parquet", "xlsx", "xls", "tsv"]

uploaded = st.file_uploader("Upload a dataset", type = SUPPORTED)

if uploaded:
    suffix = uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete = False, suffix = f".{suffix}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    df = load_dataset(tmp_path)
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width = True)

    if st.button("Run"):
        api_key = creds.gemini_key
        # client = genai.Client(api_key = api_key)
        # api_key = os.getenv("gemini_key")
        client = genai.Client(api_key = api_key)
        log_store.clear()

        profile = profile_dataframe(df)

        if os.path.exists(RELATIONS_CACHE):
            with open(RELATIONS_CACHE) as f:
                relations = json.load(f)
        else:
            relations = analyze_relations(client, profile)
            os.makedirs("data", exist_ok = True)
            with open(RELATIONS_CACHE, "w") as f:
                json.dump(relations, f, indent = 2)

        issues = detect_issues(profile, df)

        issues_input = []
        for col in df.columns:
            col_issues = issues.get(col, [])
            if not col_issues:
                continue

            col_relations = [r for r in relations if r["col_a"] == col or r["col_b"] == col]
            options = recommend_actions(profile[col], col_issues, col_relations)
            issues_input.append({
                "column": col,
                "profile": profile[col],
                "issues": col_issues,
                "options": options,
            })

        batch_results = interpret_issues_batch(client, issues_input, relations)

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

        if relations:
            st.subheader("Semantic Relations")
            st.dataframe(pd.DataFrame([{
                "col_a": r["col_a"],
                "col_b": r["col_b"],
                "relation": r["relation"],
                "suggestion": r["suggestion"],
                "confidence": r["confidence"],
            } for r in relations]), use_container_width = True)

        st.subheader("Column Issues")
        user_decisions = {}
        for r in report["issues"]:
            col = r["column"]
            action = r["analysis"]["recommended_option"]
            conf = r["analysis"]["confidence"]

            with st.expander(f"{col} — {action} (conf: {conf})"):
                st.write(f"issues: {', '.join(r['issues'])}")
                st.write(f"risk: {r['analysis'].get('risk', 'medium')}")
                st.write(r["analysis"].get("explanation", ""))
                user_decisions[col] = st.checkbox(
                    "apply",
                    value = False,
                    key = f"approve_{col}"
                )

        if st.button("Apply"):
            for r in report["issues"]:
                if not user_decisions.get(r["column"], False):
                    r["analysis"]["confidence"] = 0.0

            cleaned_df, original_df, flagged = apply_actions(df, report)

            st.write(f"original: {original_df.shape} → cleaned: {cleaned_df.shape}")
            st.dataframe(cleaned_df.head(10), use_container_width = True)

            logs = get_logs()
            if logs:
                st.subheader("Log")
                st.dataframe(pd.DataFrame([{
                    "column": l["column"],
                    "action": l["action"],
                    "timestamp": l["timestamp"],
                } for l in logs]), use_container_width = True)

            st.download_button(
                "Download cleaned CSV",
                data = cleaned_df.to_csv(index = False),
                file_name = "cleaned.csv",
                mime = "text/csv"
            )
            st.download_button(
                "Download log",
                data = json.dumps(logs, indent = 2),
                file_name = "cleaning_log.json",
                mime = "application/json"
            )