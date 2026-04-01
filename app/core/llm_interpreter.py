import json
import pandas as pd
import google.genai as genai
from google.genai import types
from typing import Dict, List


SYSTEM_PROMPT = """
You are a data quality advisor.

Return ONLY a valid JSON object. No explanations outside JSON.

You are given:
- column profile (dtype, null %, uniqueness, type)
- detected issues
- allowed options

Rules:
- Prefer safe, reversible actions
- Avoid destructive actions unless strongly justified
- Respect column_type (nested data should not use numeric imputations)

Allowed actions:
["fill_mean", "fill_median", "fill_mode", "drop_rows", "drop_column", "leave", "encode", "group_categories"]

Output format:
{
  "explanation": str,
  "risk": "low" | "medium" | "high",
  "recommended_option": str,
  "confidence": float (0-1)
}
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

    if "constant_column" in issues and "drop_column" in options:
        action = "drop_column"
    elif "high_missing" in issues:
        for opt in ["fill_median", "fill_mean", "fill_mode", "leave"]:
            if opt in options:
                action = opt
                break
        else:
            action = options[0]
    elif "nested_data" in issues:
        action = "leave" if "leave" in options else options[0]
    elif "high_cardinality" in issues:
        for opt in ["encode", "group_categories", "leave"]:
            if opt in options:
                action = opt
                break
        else:
            action = options[0]
    elif "id_like_column" in issues or "near_constant" in issues:
        action = "drop_column" if "drop_column" in options else "leave"
    else:
        action = options[0]

    return {
        "explanation": "Fallback decision applied",
        "risk": "medium",
        "recommended_option": action,
        "confidence": 0.5,
    }


def interpret_issue(client, column_profile: dict, issues: List[str], options: List[str]) -> Dict:
    payload = build_payload(column_profile, issues, options)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\n{json.dumps(payload)}",
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        text = response.text.strip()
        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError

        required_keys = {"explanation", "risk", "recommended_option", "confidence"}
        if not required_keys.issubset(result.keys()):
            raise ValueError

        if not isinstance(result["recommended_option"], str) or result["recommended_option"] not in options:
            raise ValueError

        if result["risk"] not in {"low", "medium", "high"}:
            raise ValueError

        result["confidence"] = float(result["confidence"])
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        return result

    except Exception:
        return safe_fallback(column_profile, issues, options)