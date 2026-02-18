"""Persist streaming predictions to cumulative Parquet and rolling CSV.

Two output files: a cumulative Parquet (full history for analysis) and
a rolling CSV (most recent N rows for live dashboard polling).
"""

from pathlib import Path

import pandas as pd


class StreamWriter:
    """Appends prediction DataFrames to parquet and maintains a rolling CSV.

    Args:
        output_dir:      directory for output files
        parquet_file:    filename for the cumulative parquet file
        latest_csv:      filename for the rolling CSV
        max_latest_rows: maximum rows to keep in the rolling CSV
    """

    def __init__(
        self,
        output_dir: str = "outputs/streaming",
        parquet_file: str = "stream_predictions.parquet",
        latest_csv: str = "stream_latest.csv",
        max_latest_rows: int = 2000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.output_dir / parquet_file
        self.csv_path = self.output_dir / latest_csv
        self.max_latest_rows = max_latest_rows
        self._buffer: list[pd.DataFrame] = []
        self._total_written = 0

    def write(self, df: pd.DataFrame) -> None:
        """Append a batch of predictions to both output files.

        For the parquet file: reads existing data, concatenates the new batch,
        and overwrites. This is simple but O(n) in file size — for very large
        deployments you'd use append-mode parquet or a database instead.

        For the CSV file: keeps only the last `max_latest_rows` rows.

        Args:
            df: DataFrame with columns like sensor_id, timestamp,
                actual_speed_mph, predicted_speed_mph, residual,
                horizon_step, horizon_min, latitude, longitude
        """
        self._buffer.append(df)
        self._total_written += len(df)

        # Append to cumulative parquet (read-concat-overwrite pattern)
        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined.to_parquet(self.parquet_path, index=False)

        # Update rolling CSV — only keep the most recent rows
        if len(combined) > self.max_latest_rows:
            latest = combined.tail(self.max_latest_rows)
        else:
            latest = combined

        latest.to_csv(self.csv_path, index=False)

    @property
    def total_written(self) -> int:
        """Total number of prediction rows written across all batches."""
        return self._total_written

    def reset(self) -> None:
        """Clear all output files for a fresh simulation run."""
        if self.parquet_path.exists():
            self.parquet_path.unlink()
        if self.csv_path.exists():
            self.csv_path.unlink()
        self._buffer.clear()
        self._total_written = 0
