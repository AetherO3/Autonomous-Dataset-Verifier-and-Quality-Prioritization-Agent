import re
import time
import json
import pandas as pd
import google.genai as genai
from google.genai import types
from typing import Dict, List
from app.core.config import MODEL_NAME


SYSTEM_PROMPT = """You are a data quality advisor.

Return ONLY a valid JSON array. No explanations outside JSON.

You are given a list of column issues. For each, return a decision object.

Each column may include a "relations" field describing links with other columns.

Allowed actions:
["fill_mean", "fill_median", "fill_mode", "drop_rows", "drop_column", "leave", "encode", "group_categories"]

STRICT DECISION POLICY:

ID-LIKE COLUMNS (issue = "id_like_column"):
- ALWAYS recommend "drop_column" with confidence >= 0.9
- Ignore any relations entries for this column
- ID columns like product_id, img_link, product_link, user_id, review_id
  provide no analytical signal and must be dropped

CONSTANT COLUMNS (issue = "constant_column"):
- ALWAYS recommend "drop_column" with confidence >= 0.9

For all other columns:
- Default action is "leave" unless there is a strong reason to change

- Use "drop_column" ONLY if ALL of the following are true:
  1. The column is clearly useless (near-constant), AND
  2. It provides no meaningful signal, AND
  3. Confidence is at least 0.85

- If "relations" contains:
  - "correlated" -> DO NOT drop the column (applies to feature columns only)
  - "constraint" -> DO NOT drop the column

- Use:
  - "fill_*" for missing values
  - "encode" / "group_categories" for categorical issues
  - "leave" if unsure

CONFIDENCE RULES:
- Only assign confidence > 0.8 if the decision is very safe
- If unsure, keep confidence <= 0.7

Output format:
[
  {
    "column": str,
    "explanation": str,
    "risk": "low" | "medium" | "high",
    "recommended_option": str,
    "confidence": float (0-1)
  }
]
"""


def build_payload(column_profile: dict, issues: List[str], options: List[str]) -> dict:
    sample = column_profile.get("sample", [])[:3]
    sample = [str(s) if isinstance(s, (pd.Timestamp, bool)) else s for s in sample]
    return {
        "dtype": column_profile.get("dtype"),
        "column_type": column_profile.get("column_type"),
        "null_perc": column_profile.get("null_perc"),
        "unique_ratio": column_profile.get("unique_ratio"),
        "issues": issues,
        "options": options,
        "sample": sample,
    }


def safe_fallback(column_profile: dict, issues: List[str], options: List[str]) -> dict:
    if not options:
        return {
            "explanation": "No options available",
            "risk": "low",
            "recommended_option": None,
            "confidence": 0.0,
        }

    if "id_like_column" in issues:
        action = "drop_column" if "drop_column" in options else "leave"
        return {
            "explanation": "ID-like column dropped by fallback",
            "risk": "low",
            "recommended_option": action,
            "confidence": 0.9,
        }

    if "constant_column" in issues and "drop_column" in options:
        action = "drop_column"
    elif "high_missing" in issues:
        action = next(
            (o for o in ["fill_median", "fill_mean", "fill_mode", "leave"] if o in options),
            options[0]
        )
    elif "nested_data" in issues:
        action = "leave" if "leave" in options else options[0]
    elif "high_cardinality" in issues:
        action = next(
            (o for o in ["encode", "group_categories", "leave"] if o in options),
            options[0]
        )
    elif "near_constant" in issues:
        action = "drop_column" if "drop_column" in options else "leave"
    else:
        action = options[0]

    return {
        "explanation": "Fallback decision applied",
        "risk": "medium",
        "recommended_option": action,
        "confidence": 0.5,
    }


def parse_retry_delay(e: Exception) -> float:
    try:
        match = re.search(r"retryDelay.*?(\d+)s", str(e))
        return float(match.group(1)) + 1 if match else 60.0
    except Exception:
        return 60.0


def interpret_issues_batch(client, issues_list: List[dict], relations: list) -> Dict[str, dict]:
    payload = []
    for item in issues_list:
        entry = build_payload(item["profile"], item["issues"], item["options"])
        entry["column"] = item["column"]
        entry["relations"] = [
            r for r in relations
            if r["col_a"] == item["column"] or r["col_b"] == item["column"]
        ]
        payload.append(entry)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"{SYSTEM_PROMPT}\n\n{json.dumps(payload)}",
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            results = json.loads(response.text.strip())

            if not isinstance(results, list):
                raise ValueError("Response is not a list")

            validated = {}
            for r in results:
                col = r.get("column")
                options = next(
                    (i["options"] for i in issues_list if i["column"] == col), []
                )
                if not col or not isinstance(r.get("recommended_option"), str):
                    continue
                if r["recommended_option"] not in options:
                    r["recommended_option"] = options[0] if options else "leave"
                if r.get("risk") not in {"low", "medium", "high"}:
                    continue
                r["confidence"] = max(0.0, min(1.0, float(r.get("confidence", 0.5))))
                validated[col] = r

            return validated

        except Exception as e:
            if "429" in str(e):
                delay = parse_retry_delay(e)
                print(f"Rate limited. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"LLM batch error: {e}")
                break

    return {
        item["column"]: safe_fallback(item["profile"], item["issues"], item["options"])
        for item in issues_list
    }