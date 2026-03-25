import json
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
["fill_mean", "fill_median", "fill_mode", "drop_rows", "drop_column", "leave"]

Output format:
{
  "explanation": str,
  "risk": "low" | "medium" | "high",
  "recommended_option": str,
  "confidence": float (0-1)
}
"""


def build_payload(column_profile: dict, issues: List[str], options: List[str]) -> dict:
    return {
        "dtype": column_profile.get("dtype"),
        "column_type": column_profile.get("column_type"),
        "null_perc": column_profile.get("null_perc"),
        "unique_ratio": column_profile.get("unique_ratio"),
        "issues": issues,
        "options": options,
        "sample": column_profile.get("sample", [])[:3]
    }


def safe_fallback(column_profile: dict, issues: List[str], options: List[str]) -> dict:
    if not options:
        return {
            "explanation": "No options available",
            "risk": "low",
            "recommended_option": None,
            "confidence": 0.0
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
    else:
        action = options[0]

    return {
        "explanation": "Fallback decision applied",
        "risk": "medium",
        "recommended_option": action,
        "confidence": 0.5
    }


def interpret_issue(client, column_profile: dict, issues: List[str], options: List[str]) -> Dict:
    payload = build_payload(column_profile, issues, options)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)}
            ],
            temperature=0.2
        )

        text = response.choices[0].message.content.strip()

        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError("Invalid response format")

        required_keys = {"explanation", "risk", "recommended_option", "confidence"}
        if not required_keys.issubset(result.keys()):
            raise ValueError("Missing keys")

        if result["recommended_option"] not in options:
            raise ValueError("Invalid option selected")

        if result["risk"] not in {"low", "medium", "high"}:
            raise ValueError("Invalid risk level")

        result["confidence"] = float(result["confidence"])
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        return result

    except Exception:
        return safe_fallback(column_profile, issues, options)