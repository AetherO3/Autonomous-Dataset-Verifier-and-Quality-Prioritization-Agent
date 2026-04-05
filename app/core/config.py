import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "gemini-2.5-flash"
CONFIDENCE_THRESHOLD = 0.7
RATE_LIMIT_SLEEP = 8

ALLOWED_ACTIONS = ["impute", "drop", "convert", "flag", "ignore"]
ALLOWED_SEVERITY = ["low", "medium", "high"]