import json
import time
import re
from typing import Dict, List
from app.core.config import MODEL_NAME


RELATION_PROMPT = """
You are a data quality expert analyzing semantic relationships between dataset columns.

Given column profiles, identify meaningful relationships between column pairs.

Return ONLY a valid JSON array. No explanations outside JSON.
If no meaningful relations found, return [].

Relationship types:
- "redundant": columns likely contain the same or derivable information
- "inconsistent": columns that should logically agree but may not
- "correlated": strongly related, one may be droppable
- "constraint": one column constrains valid values of another

Output format:
[
  {
    "col_a": str,
    "col_b": str,
    "relation": "redundant" | "inconsistent" | "correlated" | "constraint",
    "explanation": str,
    "suggestion": "drop_col_a" | "drop_col_b" | "flag_for_review" | "add_constraint_check",
    "confidence": float (0-1)
  }
]
"""


def build_relation_payload(profile: dict) -> list:
    return [
        {
            "column": col,
            "dtype": stats["dtype"],
            "column_type": stats["column_type"],
            "unique_ratio": stats["unique_ratio"],
            "null_perc": stats["null_perc"],
            "sample": stats["sample"][:3],
        }
        for col, stats in profile.items()
    ]


def parse_retry_delay(e: Exception) -> float:
    try:
        match = re.search(r"retryDelay.*?(\d+)s", str(e))
        return float(match.group(1)) + 1 if match else 60.0
    except Exception:
        return 60.0


def analyze_relations(client, profile: dict) -> List[Dict]:
    payload = build_relation_payload(profile)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"{RELATION_PROMPT}\n\n{json.dumps(payload)}",
            )

            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            result = json.loads(text)
            if not isinstance(result, list):
                return []

            valid = []
            for r in result:
                if not {"col_a", "col_b", "relation", "suggestion", "confidence"}.issubset(r.keys()):
                    continue
                if r["relation"] not in {"redundant", "inconsistent", "correlated", "constraint"}:
                    continue
                if r["suggestion"] not in {"drop_col_a", "drop_col_b", "flag_for_review", "add_constraint_check"}:
                    continue
                r["confidence"] = max(0.0, min(1.0, float(r["confidence"])))
                valid.append(r)

            return valid

        except Exception as e:
            if "429" in str(e):
                delay = parse_retry_delay(e)
                print(f"Relation analyzer rate limited. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Relation analyzer error: {e}")
                return []

    return []