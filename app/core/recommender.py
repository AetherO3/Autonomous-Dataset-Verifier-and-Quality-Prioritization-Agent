def recommend_actions(column_profile: dict, issues: list) -> list:

    options = []

    if "high_missing" in issues:
        options = ["fill_mean", "fill_median", "drop_rows", "leave"]

    elif "constant_column" in issues:
        options = ["drop_column", "keep"]

    elif "high_cardinality" in issues:
        options = ["group_categories", "encode", "leave"]

    return options