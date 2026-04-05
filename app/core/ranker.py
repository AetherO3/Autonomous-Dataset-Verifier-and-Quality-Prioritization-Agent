SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

def rank_issues(issues: list) -> list:
    return sorted(
        issues,
        key=lambda x: (
            SEVERITY_WEIGHT.get(x["analysis"].get("risk", "low"), 1),
            x["profile"].get("null_perc", 0),
            x["analysis"].get("confidence", 0),
        ),
        reverse=True,
    )