from app.vision.image_utils import load_image_from_url
from app.vision.clip_model import load_clip_model
from app.core.config import DEVICE
from urllib.parse import urlparse
from typing import List, Dict
import pandas as pd
import torch
import random


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def is_valid_image_url(x: str) -> bool:
    if not isinstance(x, str):
        return False

    try:
        parsed = urlparse(x)
        if parsed.scheme not in ("http", "https"):
            return False

        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in VALID_EXTENSIONS)
    except Exception:
        return False


def find_image_url_columns(df, sample_size: int = 20) -> List[str]:
    image_cols = []

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue

        sample = sample.sample(min(sample_size, len(sample)), random_state=42)

        valid_ratio = sample.apply(is_valid_image_url).mean()

        if valid_ratio > 0.6:
            image_cols.append(col)

    return image_cols


def get_text_columns(df, exclude: List[str]) -> List[str]:
    text_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if df[col].dtype == object:
            sample = df[col].dropna().head(20)

            if sample.empty:
                continue

            # ensure majority are strings
            if sample.apply(lambda x: isinstance(x, str)).mean() > 0.7:
                text_cols.append(col)

    return text_cols


def check_image_text_consistency(df, model_name: str = "openai/clip-vit-base-patch32", sample_size: int = 20, batch_size: int = 8,) -> List[Dict]:

    model, processor = load_clip_model(model_name, DEVICE)
    model.eval()

    image_cols = find_image_url_columns(df)
    if not image_cols:
        return []

    text_cols = get_text_columns(df, exclude=image_cols)
    if not text_cols:
        return []

    results = []

    for img_col in image_cols:
        base_sample = df[[img_col] + text_cols].dropna(subset=[img_col])

        if base_sample.empty:
            continue

        base_sample = base_sample.sample(
            min(sample_size, len(base_sample)), random_state=42
        )

        for text_col in text_cols:
            pairs = []

            for _, row in base_sample.iterrows():
                try:
                    url = row[img_col]
                    text = str(row[text_col])

                    if not is_valid_image_url(url):
                        continue

                    image = load_image_from_url(url)
                    pairs.append((image, text))

                except Exception as e:
                    print("Image Quality Checker Failed at reading image or text.")
                    continue

            if not pairs:
                continue

            scores = []

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                images, texts = zip(*batch)

                inputs = processor(
                    text=list(texts),
                    images=list(images),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)

                    # CLIP similarity (correct way)
                    logits = outputs.logits_per_image  # shape: (N, N)
                    probs = logits.softmax(dim=1)

                    # take diagonal (matching pairs)
                    batch_scores = probs.diag().cpu().tolist()

                    scores.extend(batch_scores)

            if scores:
                avg_score = sum(scores) / len(scores)

                results.append({
                    "image_column": img_col,
                    "text_column": text_col,
                    "semantic_match_score": round(avg_score, 4),
                    "flagged": avg_score < 0.25,
                    "samples_used": len(scores),
                })

    return results