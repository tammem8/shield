import csv
from pathlib import Path

from shield.models.evaluation import DatasetRecord

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PREPROCESSED_DIR = Path(__file__).parent.parent / "data" / "preprocessed"


def _parse_label(value) -> int:
    return {"true": 1, "false": 0, "1": 1, "0": 0}[str(value).strip().lower()]


def _read_csv(path: Path) -> list[DatasetRecord]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            DatasetRecord(
                text=row["text"],
                label=_parse_label(row["label"]),
                language=row.get("language") or None,
            )
            for row in reader
        ]


def load_raw(filename: str, n: int | None = None) -> list[DatasetRecord]:
    records = _read_csv(RAW_DIR / filename)
    return records[:n] if n is not None else records


def load_preprocessed(filename: str, n: int | None = None) -> list[DatasetRecord]:
    records = _read_csv(PREPROCESSED_DIR / filename)
    return records[:n] if n is not None else records


def save_preprocessed(records: list[DatasetRecord], filename: str) -> Path:
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PREPROCESSED_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "language"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "text": record.text,
                    "label": record.label,
                    "language": record.language,
                }
            )
    return path
