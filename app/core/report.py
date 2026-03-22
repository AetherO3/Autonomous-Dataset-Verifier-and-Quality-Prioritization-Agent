def generate_report(issues: list) -> dict:
    return {
        "total_issues": len(issues),
        "issues": issues
    }