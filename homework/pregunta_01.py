import os
import glob
import zipfile

import pandas as pd


def pregunta_01():
    """Genera train_dataset.csv y test_dataset.csv a partir de input.zip."""

    with zipfile.ZipFile("files/input.zip", "r") as zip_ref:
        zip_ref.extractall("files/")

    os.makedirs("files/output", exist_ok=True)

    def build_dataset(split):
        phrases = []
        targets = []
        for sentiment in ["negative", "positive", "neutral"]:
            pattern = os.path.join("files/input", split, sentiment, "*.txt")
            for filepath in glob.glob(pattern):
                with open(filepath, "r", encoding="utf-8") as f:
                    phrases.append(f.read().strip())
                targets.append(sentiment)
        return pd.DataFrame({"phrase": phrases, "target": targets})

    train_dataset = build_dataset("train")
    test_dataset = build_dataset("test")

    train_dataset.to_csv("files/output/train_dataset.csv", index=False)
    test_dataset.to_csv("files/output/test_dataset.csv", index=False)