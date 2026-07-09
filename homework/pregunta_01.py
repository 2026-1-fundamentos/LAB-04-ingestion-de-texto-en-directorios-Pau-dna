# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""


from pathlib import Path

import pandas as pd

BASE_DIR = Path("files")
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_DIR = INPUT_DIR / "train"
TEST_DIR = INPUT_DIR / "test"


def pregunta_01():
    """Genera los archivos train_dataset.csv y test_dataset.csv en files/output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = build_dataset_rows(TRAIN_DIR)
    test_rows = build_dataset_rows(TEST_DIR)

    train_path = OUTPUT_DIR / "train_dataset.csv"
    test_path = OUTPUT_DIR / "test_dataset.csv"

    save_dataset(train_rows, train_path)
    save_dataset(test_rows, test_path)


def build_dataset_rows(split_dir):
    """Construye las filas de un dataset a partir de un directorio de split."""
    if not split_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {split_dir}")

    rows: list[dict[str, str]] = []

    for sentiment_dir in sorted(split_dir.iterdir()):
        if not sentiment_dir.is_dir():
            continue

        sentiment = sentiment_dir.name
        text_files = sorted(sentiment_dir.glob("*.txt"))

        for text_file in text_files:
            phrase = read_phrase(text_file)
            rows.append({"phrase": phrase, "target": sentiment})

    return rows


def read_phrase(text_path):
    """Lee el texto de un archivo .txt y devuelve la frase limpia."""
    content = text_path.read_text(encoding="utf-8", errors="replace")
    return content.strip()


def save_dataset(rows: list[dict[str, str]], output_path: Path) -> None:
    """Guarda un listado de filas en un CSV con columnas phrase y target."""
    df = pd.DataFrame(rows, columns=["phrase", "target"])
    df.to_csv(output_path, index=False)
