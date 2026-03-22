import time

log_store = []


def log_operation(col, action, before, after):
    log_store.append({
        "column": col,
        "action": action,
        "before_sample": str(before.head().tolist()) if before is not None else None,
        "after_sample": str(after.head().tolist()) if after is not None else None,
        "timestamp": time.time()
    })


def get_logs():
    return log_store