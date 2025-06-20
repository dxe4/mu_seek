from pathlib import Path

import polars as pl

from rich.console import Console
from rich.progress import track


current_dir = Path.cwd()
BASE_DIR = current_dir / "data"
console = Console()


def convert_all_csv_to_parquet(base_dir: Path):
    csv_dir = base_dir / "csv-dir"
    parquet_dir = base_dir / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        console.print("[bold red]No CSV files found in the directory.[/bold red]")
        return


    for csv_file in track(csv_files, description="[cyan]Converting CSVs to Parquet..."):
        parquet_file = parquet_dir / (csv_file.stem + ".parquet")
        try:
            console.print(f"[green]✓ Saved:[/green] {parquet_file}")
            pl.scan_csv(csv_file, ignore_errors=True).sink_parquet(
                parquet_dir / (csv_file.stem + ".parquet"),
                compression="zstd",
                row_group_size=70_000
            )
        except Exception as e:
            console.print(f"[red]✗ Failed:[/red] {csv_file.name} — {e}")


if __name__ == "__main__":
    convert_all_csv_to_parquet(BASE_DIR)
