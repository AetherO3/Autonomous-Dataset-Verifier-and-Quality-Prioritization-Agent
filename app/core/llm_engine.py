from app.core.config import ALLOWED_ACTIONS, ALLOWED_SEVERITY


def generate_llm_decision(column_profile: dict, issues: list) -> dict:

    # Default decision.
    decision = {    "issue_type": "none",
        "severity": "low",
        "recommended_action": "ignore",
        "confidence_score": 0.9,
        "explanation": "No significant issues detected"
    }

    if "high_missing" in issues:
        decision.update({
            "issue_type": "missing_values",
            "severity": "high",
            "recommended_action": "impute",
            "confidence_score": 0.85,
            "explanation": "High percentage of missing values"
        })

    elif "constant_column" in issues:
        decision.update({
            "issue_type": "constant",
            "severity": "medium",
            "recommended_action": "drop",
            "confidence_score": 0.8,
            "explanation": "Column has constant value"
        })

    elif "high_cardinality" in issues:
        decision.update({
            "issue_type": "cardinality",
            "severity": "medium",
            "recommended_action": "flag",
            "confidence_score": 0.75,
            "explanation": "Too many unique values"
        })

    return decision