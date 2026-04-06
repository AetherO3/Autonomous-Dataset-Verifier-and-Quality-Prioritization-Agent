def recommend_actions(column_profile: dict, issues: list) -> list:
    options = set()
    column_type = column_profile.get("column_type", "scalar")

    if "id_like_column" in issues:
        options.update(["drop_column", "leave"])
        return sorted(options)

    if "high_missing" in issues:
        options.update(["fill_mean", "fill_median", "fill_mode", "drop_rows", "leave"])

    if "constant_column" in issues:
        options.update(["drop_column", "leave"])

    if "high_cardinality" in issues:
        options.update(["encode", "group_categories", "leave"])

    if "near_constant" in issues:
        options.update(["drop_column", "leave"])

    if "nested_data" in issues:
        options.update(["leave"])

    if not options:
        options.add("leave")

    if column_type == "nested":
        options = {"leave"} if "leave" in options else options

    if "high_skewness" in issues:
        options.update(["leave"])

    if "high_zeros" in issues:
        options.update(["drop_column", "leave"])

    return sorted(options)