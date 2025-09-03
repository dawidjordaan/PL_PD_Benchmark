from __future__ import annotations

import os
import time
import glob
import threading
from typing import Callable, Dict, List, Tuple

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


def print_report(pd_metrics: Dict[str, float], pl_metrics: Dict[str, float]) -> None:
    header = (
        "\n"
        "------------------------------------------------------------------\n"
        "| Benchmark Report: NYC Taxi Data (Jan 2025 - Jul 2025)          |\n"
        "------------------------------------------------------------------\n"
        "| Library | Time Taken (s) | Peak Memory (MB) | Avg CPU (%)    |\n"
        "------------------------------------------------------------------\n"
    )
    row_pd = f"| Pandas  | {pd_metrics['time_s']:.2f}          | {pd_metrics['peak_mem_mb']:.2f}           | {pd_metrics['avg_cpu_percent']:.2f}         |\n"
    row_pl = f"| Polars  | {pl_metrics['time_s']:.2f}          | {pl_metrics['peak_mem_mb']:.2f}           | {pl_metrics['avg_cpu_percent']:.2f}         |\n"
    footer = "------------------------------------------------------------------\n"
    print(header + row_pd + row_pl + footer)


def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    file_paths = discover_parquet_files(data_dir)

    pandas_metrics = run_benchmark(pandas_process, file_paths)
    polars_metrics = run_benchmark(polars_process, file_paths)

    print_report(pandas_metrics, polars_metrics)


if __name__ == "__main__":
    main()


