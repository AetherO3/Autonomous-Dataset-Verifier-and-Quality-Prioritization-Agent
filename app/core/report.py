def generate_report(issues: list, relations: list ) -> dict:
    return {
        "total_issues": len(issues),
        "issues": issues,
        "relations": relations or [],
    }