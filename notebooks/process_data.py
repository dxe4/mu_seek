# %%
from pathlib import Path

import polars as pl

from plotnine import (
    theme,
    theme_gray,
    theme_set,
)
from rich.console import Console


INPUT_DIR = Path.cwd().parent / "data/parquet"
OUTPUT_DIR = Path.cwd().parent / "data/processed"

# %%


def dedup_release(release_df: pl.DataFrame) -> pl.DataFrame:
    zero_master = release_df.filter(pl.col("master_id") == 0)
    non_zero_master = release_df.filter(pl.col("master_id") != 0)

    processed_non_zero = non_zero_master.sort("released").unique(
        subset=["master_id"], keep="first"
    )

    relese_dedup = pl.concat([zero_master, processed_non_zero]).sort("id")
    return relese_dedup


def get_releases_with_style_and_genre(
    filter_release_type: bool=False,
    filter_main_artists: bool=False,
    remove_style_genre_outliers: bool=False
):
    console = Console()
    releases = pl.read_parquet(
        INPUT_DIR / "release.parquet",
        columns=["id", "master_id", "released"],
    )
    releases = dedup_release(releases)
    releases = releases.select("id").rename({"id": "release_id"})

    if filter_release_type:
        rf = pl.read_parquet(
            INPUT_DIR / "release_format.parquet", columns=["release_id", "descriptions"]
        )
        rf = (
            rf.filter(
                    pl.col("descriptions").str.to_lowercase().str.contains("album")
                    | pl.col("descriptions").str.to_lowercase().str.contains('12"')
                    | pl.col("descriptions").str.to_lowercase().str.contains("lp")
            )
            .select("release_id")
            .unique()
        )

        releases = releases.join(rf, left_on="id", right_on="release_id", how="inner")


    genre = pl.read_parquet(
        INPUT_DIR / "release_style.parquet", columns=["release_id", "style"]
    )
    style = pl.read_parquet(
        INPUT_DIR / "release_genre.parquet", columns=["release_id", "genre"]
    )
    releases = releases.join(genre, on=["release_id"], how="inner").join(
        style, on=["release_id"], how="inner"
    )

    release_artist = pl.read_parquet(
        INPUT_DIR / "release_artist.parquet",
        columns=["release_id", "artist_id", "extra", "position"],
    )
    if filter_main_artists:
        release_artist = release_artist.filter(
            (pl.col("extra") == 0) & (pl.col("position") == 1) & (pl.col("artist_id") != 194)
        )

    release_artist = release_artist.select(
        "release_id",
        "artist_id",
    )
    releases = releases.join(release_artist, on="release_id", how="inner")

    if remove_style_genre_outliers:
        style_genre_artist_pairs = (
            releases.select(["style", "genre", "artist_id"])
            .drop_nulls()
            .with_columns(
                (pl.col("style") + " " + pl.col("genre")).alias("style_genre")
            )
            .group_by("artist_id")
            .agg(pl.len().alias("style_genre_count"))
        )
        style_genre_artist_pairs = style_genre_artist_pairs.filter(
            pl.col("style_genre_count") < style_genre_artist_pairs["style_genre_count"].quantile(0.99)
        )

        releases = releases.join(style_genre_artist_pairs.select("artist_id"), on="artist_id", how="inner")

    release_label = pl.read_parquet(INPUT_DIR / "release_label.parquet", columns=["release_id", "label_name"])
    releases = releases.join(release_label, on="release_id", how="inner")
    return releases


# %%

release_df = get_releases_with_style_and_genre()

# %%
release_df.write_parquet(OUTPUT_DIR / "release_artist_style_genre_labels_filtered.parquet")
# %%
# %%
sane_theme = theme_gray() + theme(figure_size=(5, 3))
theme_set(sane_theme)

# %%
release_df.shape[0] / 1e6
