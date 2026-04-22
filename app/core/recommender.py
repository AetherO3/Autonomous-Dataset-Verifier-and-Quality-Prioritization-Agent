def recommend_actions(column_profile: dict, issues: list, relations: list) -> list:
    options = set()
    column_type = column_profile.get("column_type", "scalar")
    relations = relations or []

    is_correlated = any(r["relation"] == "correlated" for r in relations)
    has_constraint = any(r["relation"] == "constraint" for r in relations)

    if "id_like_column" in issues:
        return ["drop_column", "leave"]

    if "high_missing" in issues:
        options.update(["fill_mean", "fill_median", "fill_mode", "drop_rows", "leave"])

    if "constant_column" in issues:
        return ["drop_column", "leave"]
    
    if "high_cardinality" in issues:
        options.update(["encode", "group_categories", "leave"])

    if "near_constant" in issues:
        options.add("leave")
        if column_profile.get("unique_ratio", 1) < 0.001:
            options.add("drop_column")

    if "nested_data" in issues:
        return ["leave"]

    if "high_zeros" in issues:
        options.add("leave")

    if "high_skewness" in issues:
        options.add("leave")

    if not options:
        options.add("leave")

    if is_correlated:
        options.discard("drop_column")

    if has_constraint:
        options.discard("drop_column")

    return sorted(options)