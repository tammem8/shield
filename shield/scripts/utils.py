import csv
from pathlib import Path

from shield.models import DatasetRecord

DATA_DIR = Path(__file__).parent.parent / "data"


def load_dataset() -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(DatasetRecord(text=row["text"], label=int(row["label"])))
    return records
