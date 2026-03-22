def rank_issues(issues: list) -> list:
    return sorted(
        issues,
        key=lambda x: x["profile"].get("null_perc", 0),
        reverse=True
    )