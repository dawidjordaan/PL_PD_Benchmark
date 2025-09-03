from __future__ import annotations

from typing import List, Tuple

import polars as pl


def process_data(file_paths: List[str]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process NYC TLC Yellow Taxi Parquet files using Polars (lazy API where possible).

    Returns:
    (location_summary_df, active_trips_df)
    """

    if not file_paths:
        raise ValueError("process_data received an empty list of file paths")

    # Lazy scan and concatenate all files
    lazy_scans = [pl.scan_parquet(path) for path in file_paths]
    lf = pl.concat(lazy_scans)

    # Ensure datetime types and filter invalid values
    lf = lf.with_columns(
        [
            pl.col("tpep_pickup_datetime").cast(pl.Datetime).alias("tpep_pickup_datetime"),
            pl.col("tpep_dropoff_datetime").cast(pl.Datetime).alias("tpep_dropoff_datetime"),
        ]
    ).drop_nulls(["tpep_pickup_datetime", "tpep_dropoff_datetime"])  # drop rows with invalid datetimes

    # Cleaning & filtering
    lf = lf.filter(pl.col("trip_distance") > 0)
    lf = lf.filter(pl.col("passenger_count").fill_null(0) > 0)

    # Feature engineering: trip duration in minutes
    lf = lf.with_columns(
        (
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds()
            / 60.0
        ).alias("trip_duration_minutes")
    )

    # Filter duration between 1 and 960 minutes
    lf = lf.filter((pl.col("trip_duration_minutes") >= 1.0) & (pl.col("trip_duration_minutes") <= 960.0))

    # Aggregation #1: Location-based summary
    location_summary_lf = (
        lf.group_by("PULocationID")
        .agg(
            avg_trip_distance=pl.col("trip_distance").mean(),
            avg_trip_duration=pl.col("trip_duration_minutes").mean(),
            trip_count=pl.len(),
        )
        .sort("trip_count", descending=True)
    )

    # Aggregation #2: Active trip time series via events
    pickups_lf = lf.select(
        [
            pl.col("tpep_pickup_datetime").alias("event_time"),
            pl.lit(1).alias("change"),
        ]
    )
    dropoffs_lf = lf.select(
        [
            pl.col("tpep_dropoff_datetime").alias("event_time"),
            pl.lit(-1).alias("change"),
        ]
    )
    events_lf = pl.concat([pickups_lf, dropoffs_lf]).sort("event_time")
    active_trips_lf = events_lf.with_columns(
        pl.col("change").cum_sum().alias("active_trips")
    ).select(["event_time", "active_trips"])  # retain only required columns

    # Collect to eager DataFrames
    location_summary_df = location_summary_lf.collect()
    active_trips_df = active_trips_lf.collect()

    return location_summary_df, active_trips_df


