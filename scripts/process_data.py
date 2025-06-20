from pathlib import Path

import polars as pl


INPUT_DIR = Path(__file__).parent.parent / "data/parquet"
OUTPUT_DIR = Path(__file__).parent.parent / "data/processed"


def add_artists_to_releases(relases: pl.DataFrame):
    release_artist = pl.read_parquet(
        INPUT_DIR / "release_artist.parquet", columns=["release_id", "artist_id"]
    )

    release_counts = release_artist.group_by("release_id").agg(
        pl.count().alias("artists_in_album")
    )
    filtered_releases = release_counts.filter(pl.col("artists_in_album") <= 3)

    filtered_release_artist = release_artist.join(
        filtered_releases, on="release_id", how="inner"
    )

    relases = relases.join(
        filtered_release_artist,
        left_on="id",
        right_on="release_id",
        how="inner",
    )
    return relases


def dedup_releases(releases: pl.DataFrame):
    zero_master = releases.filter(pl.col("master_id") == 0)
    non_zero_master = releases.filter(pl.col("master_id") != 0)

    processed_non_zero = non_zero_master.sort("released").unique(
        subset=["master_id"], keep="first"
    )

    relese_dedup = pl.concat([zero_master, processed_non_zero]).sort("id")
    return relese_dedup


def add_genres_to_releases(releases: pl.DataFrame):
    release_genre = pl.read_parquet(INPUT_DIR / "release_genre.parquet")
    genres_per_release = release_genre.group_by("release_id").agg(
        pl.col("genre").alias("genres")
    )

    releases = releases.join(
        genres_per_release,
        left_on="id",
        right_on="release_id",
        how="left",
    )

    releases = releases.with_columns(pl.col("genres").fill_null(pl.lit([])))
    return releases


def add_styles_to_releases(releases: pl.DataFrame):
    release_style = pl.read_parquet(INPUT_DIR / "release_style.parquet")
    styles_per_release = release_style.group_by("release_id").agg(
        pl.col("style").alias("styles")
    )

    releases = releases.join(
        styles_per_release,
        left_on="id",
        right_on="release_id",
        how="left",
    )
    releases = releases.with_columns(pl.col("styles").fill_null(pl.lit([])))
    return releases


def add_release_video_on_releases():
    release_ids = pl.read_parquet(
        OUTPUT_DIR / "releases_w_style_genre_artist_and_video.parquet", columns=["id"]
    ).with_columns(pl.col("id").alias("release_id"))
    release_video = pl.read_parquet(
        INPUT_DIR / "release_video.parquet",
        columns=["release_id", "duration", "title", "uri"],
    )
    release_video = release_video.join(release_ids, on="release_id", how="inner")
    grouped = release_video.group_by("release_id").agg(
        pl.col("duration").alias("durations"),
        pl.col("title").alias("titles"),
        pl.col("uri").alias("uris"),
    )
    video_links_release = (
        grouped.with_columns(
            pl.struct(["durations", "titles", "uris"])
            .map_elements(
                lambda x: [
                    {"duration": d, "uri": u, "title": t}
                    for d, u, t in zip(
                        x["durations"], x["titles"], x["uris"], strict=False
                    )
                ]
            )
            .alias("release_video_info")
        )
        .select(["release_id", "release_video_info"])
        .with_columns(pl.col("release_video_info").fill_null(pl.lit([])))
    )
    video_links_release.write_parquet(OUTPUT_DIR / "relese_video_grouped.parquet")


def add_genres_styles_and_artits_on_releases():
    release_original = pl.read_parquet(
        INPUT_DIR / "release.parquet",
        columns=["id", "title", "released", "country", "master_id"],
    )
    releases = dedup_releases(release_original)

    releases = add_genres_to_releases(releases)
    releases = add_styles_to_releases(releases)
    releases = add_artists_to_releases(releases)

    releases.write_parquet(
        OUTPUT_DIR / "releases_w_style_genre_artist_and_video.parquet"
    )


def add_urls_to_artists():
    artist = pl.read_parquet(INPUT_DIR / "artist.parquet")
    artist_url = pl.read_parquet(INPUT_DIR / "artist_url.parquet")

    urls_per_artist = artist_url.group_by("artist_id").agg(pl.col("url").alias("urls"))

    artist_with_urls = artist.join(
        urls_per_artist, left_on="id", right_on="artist_id", how="left"
    )
    artist_with_urls.write_parquet(OUTPUT_DIR / "artists_with_urls.parquet")


def create_track_durations():
    tracks = pl.read_parquet(
        INPUT_DIR / "release_track.parquet", columns=["duration", "release_id"]
    )
    release_ids = pl.read_parquet(
        OUTPUT_DIR / "releases_w_style_genre_artist_and_video.parquet", columns=["id"]
    ).with_columns(pl.col("id").alias("release_id"))
    tracks = tracks.join(release_ids, on="release_id", how="inner")

    tracks = tracks.with_columns(
        pl.col("duration").str.replace_all(r"\s+", "").alias("duration")
    )

    tracks = tracks.filter(
        (pl.col("duration") != "")
        & (pl.col("duration").str.contains(":"))
        & (pl.col("duration").str.contains(r"^\d+:\d+$"))
        & (~pl.col("duration").str.contains(r"[۰۱۲۳۴۵۶۷۸۹]"))
        & (~pl.col("duration").str.contains(r"[０-９]"))
    )
    tracks = (
        tracks.with_columns(
            pl.col("duration")
            .str.split(":")
            .list.get(0)
            .cast(pl.Int64)
            .alias("minutes"),
            pl.col("duration")
            .str.split(":")
            .list.get(1)
            .cast(pl.Int64)
            .alias("seconds"),
        )
        .with_columns(
            (pl.col("minutes") * 60 + pl.col("seconds")).alias("duration_seconds")
        )
        .group_by("release_id")
        .agg(pl.col("duration_seconds").sum().alias("total_duration_seconds"))
    )
    track_durations = tracks.group_by("release_id", maintain_order=True).mean()
    track_durations.write_parquet(OUTPUT_DIR / "tracks_with_duration.parquet")


def add_track_duration_and_label_on_releases():
    tracks_with_duration = pl.read_parquet(OUTPUT_DIR / "tracks_with_duration.parquet")
    releases = pl.read_parquet(
        OUTPUT_DIR / "releases_w_style_genre_artist_and_video.parquet"
    )

    release = releases.join(
        tracks_with_duration, left_on="id", right_on="release_id", how="left"
    )
    release_label = pl.read_parquet(
        INPUT_DIR / "release_label.parquet", columns=["release_id", "label_name"]
    ).unique()
    release_label = release_label.group_by("release_id").agg(
        pl.col("label_name").alias("label_name"),
        pl.col("label_name").count().alias("label_count"),
    )
    release = release.join(
        release_label, left_on="id", right_on="release_id", how="left"
    )
    release = release.with_columns(pl.col("label_name").fill_null(pl.lit([])))
    release = release.with_columns(pl.col("label_count").fill_null(0))
    release.write_parquet(
        OUTPUT_DIR / "releases_w_style_genre_artist_video_track_and_label.parquet"
    )


def create_master_agg():
    releases = pl.read_parquet(
        OUTPUT_DIR / "releases_w_style_genre_artist_video_track_and_label.parquet"
    )
    artist_master_agg = releases.group_by("artist_id").agg(
        pl.col("genres").flatten().alias("genres"),
        pl.col("styles").flatten().alias("styles"),
        pl.col("label_name").flatten().alias("label_name"),
        pl.col("total_duration_seconds").mean().alias("duration_mean"),
        pl.col("total_duration_seconds").median().alias("duration_median"),
        pl.col("total_duration_seconds").sum().alias("duration_sum"),
        pl.count().alias("album_master_count"),
    )
    list_columns = ["genres", "styles", "label_name"]

    artist_master_agg = artist_master_agg.with_columns(
        pl.col(col).list.eval(pl.element().drop_nulls()).alias(col)
        for col in list_columns
    )
    artist_master_agg.write_parquet(OUTPUT_DIR / "artist_release_agg.parquet")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    add_genres_styles_and_artits_on_releases()
    add_urls_to_artists()
    add_release_video_on_releases()
    create_track_durations()
    add_track_duration_and_label_on_releases()
    create_master_agg()
