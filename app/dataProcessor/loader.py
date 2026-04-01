import os
import pandas as pd
import kagglehub

def load_dataset(source: str):
    path = source if os.path.exists(source) else kagglehub.dataset_download(source)

    if os.path.isfile(path):
        target = path
    else:
        files = [os.path.join(root, f)  for root, _, fs in os.walk(path)    for f in fs
            if f.lower().endswith((".csv", ".json", ".jsonl", ".parquet", ".xlsx", ".tsv"))]

        if not files:
            raise FileNotFoundError("No valid dataset found")

        target = max(files, key=os.path.getsize)

    ext = target.split('.')[-1].lower()

    loaders = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "jsonl": lambda f: pd.read_json(f, lines=True),
        "parquet": pd.read_parquet,
        "xlsx": pd.read_excel,
        "xls": pd.read_excel,
        "tsv": lambda f: pd.read_csv(f, sep="\t"),
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}")

    return loaders[ext](target)
