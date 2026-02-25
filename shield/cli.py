import asyncio  # noqa: F401 — used via asyncio.run

import click
from rich.console import Console
from rich.table import Table

from shield.models.evaluator import Evaluator
from shield.scripts.utils import load_dataset

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
def analyze() -> None:
    """Evaluate the shield API against all datasets in the data folder."""
    records = load_dataset()
    console.print(f"Loaded [bold]{len(records)}[/bold] records\n")

    metrics = asyncio.run(Evaluator(records).run())

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


if __name__ == "__main__":
    main()
