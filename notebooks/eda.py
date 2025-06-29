# %%
from pathlib import Path

import polars as pl


INPUT_DIR = Path.cwd().parent / "data/parquet"
OUTPUT_DIR = Path.cwd().parent / "data/processed"
release_df = pl.read_parquet(OUTPUT_DIR / "release_artist_style_genre_labels_filtered.parquet")

# %%

release_df = release_df.drop_nulls()


# %%
release_df.head()

# %%
df_prepared = release_df.with_columns([
    pl.format("{} | {}", pl.col("genre"), pl.col("style")).alias("feature_name"),
    pl.lit(1, dtype=pl.UInt8).alias("value_col")
])

df_unique = df_prepared.unique(
    subset=["release_id", "feature_name"],
    keep="first"
)

features_df = df_unique.pivot(
    on="feature_name",
    index="release_id",
    values="value_col"
).fill_null(0)
# %%
features_df.head()
