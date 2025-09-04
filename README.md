## Pandas vs Polars: NYC Yellow Taxi Benchmark

This repository benchmarks Pandas and Polars on NYC TLC Yellow Taxi trip records stored as Parquet files. It measures runtime, CPU usage, and peak memory while producing identical analytics with both libraries.

### What this benchmark does

- **Input data**: Parquet files under `Data/` (e.g., `yellow_tripdata_2025-01.parquet` … `2025-07.parquet`). You can download monthly Parquet files from the NYC TLC site: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- **Workloads (performed identically in Pandas and Polars)**
  - **Load & concatenate** all Parquet files.
  - **Type hygiene**: cast `tpep_pickup_datetime` and `tpep_dropoff_datetime` to datetimes and drop rows with invalid datetimes.
  - **Row filtering**:
    - `trip_distance` > 0
    - `passenger_count` > 0 (treat nulls as 0)
  - **Feature engineering**: compute `trip_duration_minutes = (dropoff − pickup) / 60` seconds.
  - **Duration filter**: keep trips with 1 ≤ `trip_duration_minutes` ≤ 960.
  - **Aggregation 1 (location summary)**: group by `PULocationID` and compute:
    - `avg_trip_distance`
    - `avg_trip_duration`
    - `trip_count`
  - **Aggregation 2 (active trips time series)**:
    - Build an event stream: pickups `+1`, dropoffs `-1`.
    - Sort by `event_time`, then `change` (pickups before dropoffs at same timestamp).
    - Cumulative sum to produce `active_trips` over time.
- **Result validation**: The outputs from Pandas and Polars are canonicalized and compared for exact equality; if they differ, lightweight diffs are printed to help diagnose mismatches.

### Metrics collected

- **Time (s)**: wall-clock execution time per library.
- **Avg CPU (%/core)**: process CPU sampled every 10 ms and normalized per logical core in the report.
- **Peak memory (MB)**: peak RSS of the process during the run.
- **System info**: OS, CPU cores and frequencies, total/available memory.
- **Disk IO (MB/s)**: sequential read speed from a sample data file and sequential write speed to `Data/` via a 32 MB temporary file.

### Requirements

- Python 3.10+
- Platform with sufficient RAM for the chosen dataset

Python packages are listed in `requirements.txt`:

```
pandas>=2.2.0
polars>=1.5.0
pyarrow>=16.0.0
psutil>=5.9.0
```

### Setup

Prepare a virtual environment and install dependencies.

- Windows (PowerShell):

```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

- Linux/macOS (bash/zsh):

```
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Place your Parquet files in the `Data/` directory at the repository root. Filenames must end with `.parquet`.

### Run

From the project root:

```
python benchmark_runner.py
```

You should see a report similar to:

```
------------------------------------------------------------------
| Benchmark Report: NYC Taxi Data (Jan 2025 - Jul 2025)          |
------------------------------------------------------------------
| Library | Time Taken (s) | Peak Memory (MB) | Avg CPU (%/core) |
------------------------------------------------------------------
| Pandas  | 12.34          | 1234.56          | 78.90            |
| Polars  |  5.67          |  890.12          | 65.43            |
------------------------------------------------------------------

System Info:
- OS: Linux 6.8.0-79-generic (x86_64)
- CPU: 8 physical / 16 logical cores
- CPU Freq (cur/min/max MHz): 2800.0/800.0/4200.0
- Memory Total: 32.0 GB
- Memory Available: 24.5 GB
- Disk IO MB/s: read=1200.1, write=850.6

Outputs identical: True
```

If the outputs are not identical, the script prints small sample diffs for the location summary and the active trips time series to help pinpoint differences.

### Files of interest

- `benchmark_runner.py`: Orchestrates discovery of Parquet files, runs the Pandas and Polars pipelines with monitoring, prints the report, and compares results.
- `pandas_operations.py`: Pandas implementation of the workload described above.
- `polars_operations.py`: Polars implementation (uses the lazy API where possible).
- `Data/`: Place `.parquet` files here.

### Customizing

- To benchmark different months or datasets, add/remove `.parquet` files under `Data/`. The runner automatically picks up all `*.parquet` files in that directory.
- The disk IO test writes a temporary file to `Data/`; ensure the directory is writable if you want those numbers.

### Troubleshooting

- If you see errors about missing Parquet/Arrow support, ensure `pyarrow` is installed (it is included in `requirements.txt`).
- Large datasets may require substantial RAM. Close other applications or use a machine with more memory.
- On Windows, if PowerShell execution policy prevents activating the venv, you can either run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, or use `python -m venv .venv` then `& .\.venv\Scripts\python.exe benchmark_runner.py` without activation.

### License

This project is licensed under the MIT License. See `LICENSE` for details.


