# %%
from pathlib import Path

import hdbscan  # Make sure to install it: pip install hdbscan
import polars as pl

from plotnine import *
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


INPUT_DIR = Path.cwd().parent / "data/parquet"
OUTPUT_DIR = Path.cwd().parent / "data/processed"
release_df = pl.read_parquet(OUTPUT_DIR / "release_artist_style_genre_labels_filtered.parquet")
bpm_sound = pl.read_parquet(OUTPUT_DIR / "bpm_and_sound_neutrality.parquet")
# %%
release_df = release_df.filter(~pl.col("label_name").str.starts_with("Not On Label"))
release_df = release_df.join(
    release_df.select("release_id", "label_name").unique().group_by("label_name").len().filter(
        pl.col("len") > 10.0  # 90 perceintile
    ).select("label_name"),
    on=["label_name"],
    how="inner"
)

# %%
release_df = release_df.group_by("release_id").agg(
    pl.first("style"),
    pl.first("genre"),
    pl.first("label_name"),
    pl.col("artist_id").unique().implode().alias("artists")
)

# %%
release_df.head()
artist_count = release_df.explode("artists") \
    .group_by(["style", "genre"]) \
    .agg(pl.col("artists").n_unique().alias("artist_count")) \
    .sort("artist_count", descending=True)

style_genre_count = release_df.group_by(["label_name", "style", "genre"]) \
    .agg(pl.count().alias("count")) \
    .sort("label_name") \
    .sort("count", descending=True)
# %%


def label_clusters(style_genre_count):
    df_with_feature = style_genre_count.with_columns(
        (pl.col("style") + "|" + pl.col("genre")).alias("feature")
    )

    # Pivot the table
    label_vectors_df = df_with_feature.pivot(
        values="count",
        index="label_name",
        columns="feature",
        aggregate_function="sum"
    ).fill_null(0)

    # Separate labels from the feature data
    label_names = label_vectors_df["label_name"]
    feature_matrix = label_vectors_df.drop("label_name")
    # %%
    sparse_feature_matrix = csr_matrix(feature_matrix.to_numpy())
    sparse_feature_matrix_normalized = normalize(sparse_feature_matrix, norm='l2', axis=1)
    N_COMPONENTS = 50
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    reduced_matrix = svd.fit_transform(sparse_feature_matrix_normalized)

    MIN_CLUSTER_SIZE = 5

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric='euclidean',
        core_dist_n_jobs=-1,
        allow_single_cluster=True
    )

    clusterer.fit(reduced_matrix)

    cluster_labels = clusterer.labels_

    result_df = pl.DataFrame({
        "label_name": label_names,
        "cluster": cluster_labels
    })

    print("\nClustering complete. Result:")
    print(result_df.sort("cluster"))

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    print(f"\nFound {num_clusters} clusters and {num_noise} noise points.")
    # %%
    result_df.filter(pl.col("cluster") > -1)
    return result_df


label_clusters = label_clusters(style_genre_count)
