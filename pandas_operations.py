from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def process_data(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process NYC TLC Yellow Taxi Parquet files using Pandas.

    Steps:
    - Load and concatenate
    - Ensure datetime types
    - Clean/filter rows
    - Feature engineer trip_duration_minutes
    - Aggregation #1: Location-based summary
    - Aggregation #2: Active trip time series via event cumsum

    Returns:
    (location_summary_df, active_trips_df)
    """

    if not file_paths:
        raise ValueError("process_data received an empty list of file paths")

    dataframes: List[pd.DataFrame] = []
    for path in file_paths:
        df_part = pd.read_parquet(path, engine="pyarrow")
        dataframes.append(df_part)

    full_df = pd.concat(dataframes, ignore_index=True)

    # Ensure datetime types
    full_df["tpep_pickup_datetime"] = pd.to_datetime(
        full_df["tpep_pickup_datetime"], errors="coerce"
    )
    full_df["tpep_dropoff_datetime"] = pd.to_datetime(
        full_df["tpep_dropoff_datetime"], errors="coerce"
    )

    # Drop rows with invalid datetimes
    full_df = full_df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])  # type: ignore[assignment]

    # Cleaning & filtering
    full_df = full_df[full_df["trip_distance"] > 0]
    full_df = full_df[full_df["passenger_count"].fillna(0) > 0]

    # Feature engineering: trip duration in minutes
    full_df["trip_duration_minutes"] = (
        (full_df["tpep_dropoff_datetime"] - full_df["tpep_pickup_datetime"]).dt.total_seconds()
        / 60.0
    )

    # Filter duration between 1 and 960 minutes (inclusive bounds)
    full_df = full_df[
        (full_df["trip_duration_minutes"] >= 1.0)
        & (full_df["trip_duration_minutes"] <= 960.0)
    ]

    # Aggregation #1: Location-based summary
    location_summary_df = (
        full_df.groupby("PULocationID", as_index=False)
        .agg(
            avg_trip_distance=("trip_distance", "mean"),
            avg_trip_duration=("trip_duration_minutes", "mean"),
            trip_count=("PULocationID", "size"),
        )
        .sort_values("trip_count", ascending=False)
        .reset_index(drop=True)
    )

    # Aggregation #2: Active trip time series
    pickups = (
        full_df[["tpep_pickup_datetime"]]
        .rename(columns={"tpep_pickup_datetime": "event_time"})
        .assign(change=1)
    )
    dropoffs = (
        full_df[["tpep_dropoff_datetime"]]
        .rename(columns={"tpep_dropoff_datetime": "event_time"})
        .assign(change=-1)
    )

    events = pd.concat([pickups, dropoffs], ignore_index=True)
    # Deterministic order at identical timestamps: apply pickups (+1) before dropoffs (-1)
    events = events.sort_values(["event_time", "change"], ascending=[True, False])
    events["active_trips"] = events["change"].cumsum()
    active_trips_df = events[["event_time", "active_trips"]].reset_index(drop=True)

    return location_summary_df, active_trips_df


