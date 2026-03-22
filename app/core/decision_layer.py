from app.core.config import CONFIDENCE_THRESHOLD

def validate_decision(decision: dict) -> dict:
    # Risk-aware validation layer.

    if decision["confidence_score"] < CONFIDENCE_THRESHOLD:
        decision["recommended_action"] = "flag"

    if decision["severity"] == "high" and decision["confidence_score"] < 0.8:
        decision["recommended_action"] = "flag"

    return decision