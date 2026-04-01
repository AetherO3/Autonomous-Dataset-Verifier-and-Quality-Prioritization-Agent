def detect_issues(profile: dict) -> dict:
    issues = {}

    for col, stats in profile.items():
        col_issues = []

        if stats["null_perc"] > 0.3:
            col_issues.append("high_missing")

        if stats["unique"] == 1:
            col_issues.append("constant_column")

        if stats["unique_ratio"] > 0.9 and "datetime" not in stats["dtype"]:
            col_issues.append("high_cardinality")

        if stats["unique"] > 1 and stats["unique_ratio"] < 0.01:
            col_issues.append("near_constant")

        if stats["unique_ratio"] == 1.0 and "datetime" not in stats["dtype"]:
            col_issues.append("id_like_column")

        if stats.get("column_type") == "nested":
            col_issues.append("nested_data")

        if col_issues:
            issues[col] = col_issues

    return issues