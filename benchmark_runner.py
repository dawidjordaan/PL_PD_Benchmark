from __future__ import annotations

import os
import time
import glob
import threading
from typing import Callable, Dict, List, Tuple

import polars as pl
import platform

import psutil

from pandas_operations import process_data as pandas_process
from polars_operations import process_data as polars_process


def discover_parquet_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No .parquet files found in: {data_dir}")
    return sorted(paths)


class Monitor:
    def __init__(self, pid: int, sample_interval_s: float = 0.01) -> None:
        self.process = psutil.Process(pid)
        self.sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: List[float] = []
        self.mem_samples_mb: List[float] = []

    def start(self) -> None:
        # Initialize CPU percent measurement baseline
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                cpu = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                self.cpu_samples.append(cpu)
                self.mem_samples_mb.append(mem_mb)
            except Exception:
                # If the process terminates or fails to sample, break
                break
            time.sleep(self.sample_interval_s)

    def summarize(self) -> Tuple[float, float]:
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        peak_mem = max(self.mem_samples_mb) if self.mem_samples_mb else 0.0
        return avg_cpu, peak_mem


def run_benchmark(processing_function: Callable[[List[str]], Tuple[object, object]], file_paths: List[str]) -> Dict[str, float | Tuple[object, object]]:
    monitor = Monitor(os.getpid(), sample_interval_s=0.01)
    monitor.start()
    start_time = time.perf_counter()
    try:
        result = processing_function(file_paths)
    finally:
        end_time = time.perf_counter()
        monitor.stop()

    total_time_s = end_time - start_time
    avg_cpu, peak_mem_mb = monitor.summarize()

    return {
        "time_s": total_time_s,
        "avg_cpu_percent": avg_cpu,
        "peak_mem_mb": peak_mem_mb,
        "result": result,
    }


def _get_system_info() -> Dict[str, object]:
    # CPU info
    try:
        cpu_freq = psutil.cpu_freq()
    except Exception:
        cpu_freq = None
    svmem = psutil.virtual_memory()
    sys_info: Dict[str, object] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "cpu_freq_max_mhz": round(cpu_freq.max, 2) if cpu_freq else None,
        "cpu_freq_min_mhz": round(cpu_freq.min, 2) if cpu_freq else None,
        "cpu_freq_current_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
        "memory_total_gb": round(svmem.total / (1024 ** 3), 2),
        "memory_available_gb": round(svmem.available / (1024 ** 3), 2),
    }
    return sys_info


def _measure_disk_speeds(data_dir: str, sample_read_path: str | None) -> Dict[str, float | None]:
    # Sequential read speed from a sample file (if available)
    read_mbps: float | None = None
    if sample_read_path and os.path.isfile(sample_read_path):
        total_bytes = 0
        start = time.perf_counter()
        try:
            with open(sample_read_path, "rb", buffering=0) as f:
                while True:
                    chunk = f.read(16 * 1024 * 1024)  # 16MB
                    if not chunk:
                        break
                    total_bytes += len(chunk)
        except Exception:
            total_bytes = 0
        end = time.perf_counter()
        if end > start and total_bytes > 0:
            read_mbps = (total_bytes / (1024 * 1024)) / (end - start)

    # Sequential write speed to data dir (temp file)
    write_mbps: float | None = None
    tmp_path = os.path.join(data_dir, ".disk_io_test.tmp")
    total_to_write = 32 * 1024 * 1024  # 32MB
    block = os.urandom(4 * 1024 * 1024)  # 4MB block
    written = 0
    start_w = time.perf_counter()
    try:
        with open(tmp_path, "wb", buffering=0) as f:
            while written < total_to_write:
                n = min(len(block), total_to_write - written)
                f.write(block[:n])
                written += n
            f.flush()
            os.fsync(f.fileno())
        end_w = time.perf_counter()
        if end_w > start_w and written > 0:
            write_mbps = (written / (1024 * 1024)) / (end_w - start_w)
    except Exception:
        write_mbps = None
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return {"disk_read_MBps": read_mbps, "disk_write_MBps": write_mbps}


def print_report(pd_metrics: Dict[str, float], pl_metrics: Dict[str, float], system_info: Dict[str, object], disk_info: Dict[str, float | None]) -> None:
    # Normalize CPU usage by number of logical CPUs so values are per-core percentages
    num_cpus = psutil.cpu_count() or 1
    pd_cpu_norm = pd_metrics["avg_cpu_percent"] / num_cpus
    pl_cpu_norm = pl_metrics["avg_cpu_percent"] / num_cpus

    header = (
        "\n"
        "------------------------------------------------------------------\n"
        "| Benchmark Report: NYC Taxi Data (Jan 2025 - Jul 2025)          |\n"
        "------------------------------------------------------------------\n"
        "| Library | Time Taken (s) | Peak Memory (MB) | Avg CPU (%/core) |\n"
        "------------------------------------------------------------------\n"
    )
    row_pd = f"| Pandas  | {pd_metrics['time_s']:.2f}          | {pd_metrics['peak_mem_mb']:.2f}           | {pd_cpu_norm:.2f}           |\n"
    row_pl = f"| Polars  | {pl_metrics['time_s']:.2f}          | {pl_metrics['peak_mem_mb']:.2f}           | {pl_cpu_norm:.2f}           |\n"
    footer = "------------------------------------------------------------------\n"
    report = header + row_pd + row_pl + footer

    # Append system information section
    sys_lines = [
        "System Info:",
        f"- OS: {system_info.get('system')} {system_info.get('release')} ({system_info.get('machine')})",
        f"- CPU: {system_info.get('physical_cores')} physical / {system_info.get('logical_cores')} logical cores",
    ]
    if system_info.get("cpu_freq_current_mhz") is not None:
        sys_lines.append(
            f"- CPU Freq (cur/min/max MHz): {system_info.get('cpu_freq_current_mhz')}/{system_info.get('cpu_freq_min_mhz')}/{system_info.get('cpu_freq_max_mhz')}"
        )
    sys_lines.extend(
        [
            f"- Memory Total: {system_info.get('memory_total_gb')} GB",
            f"- Memory Available: {system_info.get('memory_available_gb')} GB",
        ]
    )

    # Disk IO speeds
    rd = disk_info.get("disk_read_MBps")
    wr = disk_info.get("disk_write_MBps")
    io_line = "- Disk IO MB/s: "
    parts = []
    parts.append(f"read={rd:.1f}" if isinstance(rd, float) and rd > 0 else "read=n/a")
    parts.append(f"write={wr:.1f}" if isinstance(wr, float) and wr > 0 else "write=n/a")
    sys_lines.append(io_line + ", ".join(parts))

    print(report + "\n" + "\n".join(sys_lines) + "\n")


def _to_polars(df_obj: object) -> pl.DataFrame:
    # Pass through if already a Polars DataFrame
    if isinstance(df_obj, pl.DataFrame):
        return df_obj
    # Convert from pandas if available
    to_pandas = getattr(df_obj, "to_pandas", None)
    if callable(to_pandas):
        import pandas as _pd  # local import to avoid global dependency
        pd_df = to_pandas()
        return pl.from_pandas(pd_df)
    # Best-effort construction
    try:
        return pl.DataFrame(df_obj)
    except Exception:
        # If construction fails, try via dict conversion
        as_dict = getattr(df_obj, "to_dict", None)
        if callable(as_dict):
            return pl.from_dicts(as_dict())
        raise TypeError("Unsupported result type for Polars conversion")


def _canonicalize_location_summary_pl(df_obj: object) -> pl.DataFrame:
    df = _to_polars(df_obj)
    expected_cols = ["PULocationID", "avg_trip_distance", "avg_trip_duration", "trip_count"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in location summary: {missing}")
    df = df.select(expected_cols)
    df = df.with_columns(
        [
            pl.col("PULocationID").cast(pl.Int64),
            pl.col("trip_count").cast(pl.Int64),
            pl.col("avg_trip_distance").cast(pl.Float64).round(6),
            pl.col("avg_trip_duration").cast(pl.Float64).round(6),
        ]
    )
    df = df.sort("PULocationID")
    return df


def _canonicalize_active_trips_pl(df_obj: object) -> pl.DataFrame:
    df = _to_polars(df_obj)
    expected_cols = ["event_time", "active_trips"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in active trips: {missing}")
    df = df.select(expected_cols)
    df = df.with_columns(
        [
            pl.col("event_time").cast(pl.Datetime(time_unit="ns")),
            pl.col("active_trips").cast(pl.Int64),
        ]
    )
    df = df.sort("event_time")
    return df


def compare_results(pandas_result: Tuple[object, object], polars_result: Tuple[object, object]) -> bool:
    pd_loc, pd_active = pandas_result
    pl_loc, pl_active = polars_result

    # Canonicalize both to Polars with consistent dtypes, rounding, and sort order
    loc_left = _canonicalize_location_summary_pl(pd_loc)
    loc_right = _canonicalize_location_summary_pl(pl_loc)

    active_left = _canonicalize_active_trips_pl(pd_active)
    active_right = _canonicalize_active_trips_pl(pl_active)

    # Compare shapes first
    if loc_left.shape != loc_right.shape or active_left.shape != active_right.shape:
        return False

    return bool(loc_left.equals(loc_right) and active_left.equals(active_right))


def _diff_report_pl(left: pl.DataFrame, right: pl.DataFrame, key_cols: list[str] | None = None, value_cols: list[str] | None = None, title: str = "") -> None:
    print("\n--- Diff:" + (f" {title}" if title else ""))
    # Schema differences
    left_schema = {name: dtype for name, dtype in zip(left.columns, left.dtypes)}
    right_schema = {name: dtype for name, dtype in zip(right.columns, right.dtypes)}
    if left_schema != right_schema:
        print("Schema differs:")
        print("left:", left_schema)
        print("right:", right_schema)
    else:
        print("Schema: identical")

    # Row count
    print(f"Row counts -> left: {left.height}, right: {right.height}")

    # Decide keys and values
    cols = left.columns
    if key_cols is None:
        # default: use all columns not likely to be float metrics as key for location summary, or event_time for time series
        if "PULocationID" in cols:
            key_cols = ["PULocationID"]
        elif "event_time" in cols:
            key_cols = ["event_time"]
        else:
            key_cols = cols[:1]
    if value_cols is None:
        value_cols = [c for c in cols if c not in key_cols]

    # Anti-joins to find extra/missing rows
    missing_in_right = left.join(right.select(key_cols), on=key_cols, how="anti")
    missing_in_left = right.join(left.select(key_cols), on=key_cols, how="anti")
    if missing_in_right.height > 0:
        print(f"Rows present in left but missing in right: {missing_in_right.height}")
        print(missing_in_right.head(5))
    if missing_in_left.height > 0:
        print(f"Rows present in right but missing in left: {missing_in_left.height}")
        print(missing_in_left.head(5))

    # Compare values for overlapping keys
    overlap = left.join(right, on=key_cols, how="inner", suffix="_r")
    diffs = []
    for col in value_cols:
        left_col = col
        right_col = f"{col}_r"
        if left_col not in overlap.columns or right_col not in overlap.columns:
            continue
        # Try numeric comparison with tolerance; fallback to exact inequality
        try:
            delta = (overlap[left_col] - overlap[right_col]).abs()
            mask = delta > 1e-6
        except Exception:
            mask = overlap[left_col] != overlap[right_col]
        diff_rows = overlap.filter(mask).select(key_cols + [left_col, right_col]).head(5)
        if diff_rows.height > 0:
            diffs.append((col, diff_rows))

    if diffs:
        print("Value differences (showing up to 5 per column):")
        for col, sample in diffs:
            print(f"- Column: {col}")
            print(sample)
    else:
        print("No value differences on overlapping keys.")


def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    file_paths = discover_parquet_files(data_dir)
    sample_read_path = file_paths[0] if file_paths else None

    system_info = _get_system_info()
    disk_info = _measure_disk_speeds(data_dir, sample_read_path)

    pandas_metrics = run_benchmark(pandas_process, file_paths)
    polars_metrics = run_benchmark(polars_process, file_paths)

    print_report(pandas_metrics, polars_metrics, system_info, disk_info)

    # Verify outputs equality between Pandas and Polars
    outputs_match = compare_results(pandas_metrics["result"], polars_metrics["result"])
    print(f"Outputs identical: {outputs_match}")
    if not outputs_match:
        # Produce lightweight diffs to help identify mismatches
        pd_loc, pd_active = pandas_metrics["result"]
        pl_loc, pl_active = polars_metrics["result"]
        loc_left = _canonicalize_location_summary_pl(pd_loc)
        loc_right = _canonicalize_location_summary_pl(pl_loc)
        _diff_report_pl(loc_left, loc_right, key_cols=["PULocationID"], title="Location Summary")

        act_left = _canonicalize_active_trips_pl(pd_active)
        act_right = _canonicalize_active_trips_pl(pl_active)
        _diff_report_pl(act_left, act_right, key_cols=["event_time"], title="Active Trips")


if __name__ == "__main__":
    main()


