import click
from rich.console import Console
from rich.table import Table

from shield.client import augment_dataset
from shield.scripts.evaluator import Evaluator
from shield.scripts.reporter import save_results
from shield.scripts.utils import load_preprocessed, load_raw, save_preprocessed

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("filename")
@click.option(
    "--augment",
    "-a",
    is_flag=True,
    default=False,
    help="Translate fr/en/de records into the other two supported languages.",
)
@click.option(
    "--langdetect",
    "-ld",
    is_flag=True,
    default=False,
    help="Detect language for each record.",
)
@click.option("-n", default=None, type=int, help="Limit to the first N rows.")
def preprocess(filename: str, augment: bool, langdetect: bool, n: int | None) -> None:
    """Detect language and optionally augment a raw CSV file.

    FILENAME is the CSV file to process inside data/raw/.
    The result is written to data/preprocessed/ with the same name.
    """
    records = load_raw(filename, n=n)
    print(f"Loaded {len(records)} records from data/raw/{filename}")

    records = augment_dataset(records, detect_lang=langdetect, translate=augment)

    action = (
        "Augmented"
        if augment
        else "Detected language for"
        if langdetect
        else "Preprocessed"
    )
    print(f"{action} {len(records)} records")

    output_path = save_preprocessed(records, filename)
    print(f"Saved to {output_path}")


@main.command()
@click.argument("filename")
@click.option("-n", default=None, type=int, help="Limit to the first N rows.")
@click.option(
    "-t",
    "--threshold",
    default=None,
    type=click.FloatRange(0.0, 1.0),
    help="Score threshold to determine predicted class (0.00–1.00).",
)
def analyze(filename: str, n: int | None, threshold: float | None) -> None:
    """Evaluate the shield API against a preprocessed CSV file.

    FILENAME is the CSV file to analyze inside data/preprocessed/.
    """
    records = load_preprocessed(filename, n=n)
    print(f"Loaded {len(records)} records from data/preprocessed/{filename}\n")

    results, metrics = Evaluator(records).run(threshold=threshold)

    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan", min_width=20)
    table.add_column("Value", style="bold green", justify="right")

    table.add_row("Total samples", str(metrics.total))
    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1 Score", f"{metrics.f1:.4f}")
    table.add_section()
    table.add_row("True Positives (TP)", str(metrics.tp))
    table.add_row("False Positives (FP)", str(metrics.fp))
    table.add_row("False Negatives (FN)", str(metrics.fn))
    table.add_row("True Negatives (TN)", str(metrics.tn))

    console.print(table)

    output_dir = save_results(results, metrics, filename)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
