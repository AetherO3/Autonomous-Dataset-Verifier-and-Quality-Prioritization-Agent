def interpret_issue(column_profile: dict, issues: list, options: list) -> dict:
    
    explanation = "No major issues"

    if "high_missing" in issues:
        explanation = "Column has a high percentage of missing values"

    elif "constant_column" in issues:
        explanation = "Column has the same value everywhere"

    elif "high_cardinality" in issues:
        explanation = "Column has too many unique values"

    return {
        "explanation": explanation,
        "risk": "medium",
        "recommended_option": options[0] if options else None,
        "confidence": 0.7
    }