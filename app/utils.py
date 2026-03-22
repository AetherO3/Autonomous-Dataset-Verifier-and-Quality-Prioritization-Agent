def safe_get(d: dict, key, default=None):
    return d[key] if key in d else default