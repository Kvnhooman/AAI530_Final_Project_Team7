"""Download METR-LA raw data files from Google Drive via gdown.

Downloads the four METR-LA files (speed HDF5, sensor distances, locations,
and IDs) from Google Drive if they aren't already present locally.
"""

import os
from pathlib import Path

import gdown

# Each file's Google Drive ID. These are stable public links from the
# original DCRNN repository (https://github.com/liyaguang/DCRNN).
_FILES = {
    "metr-la.h5": "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC",
    "distances_la_2012.csv": "1PL05Mfl81qJHMO7VHbE5wCnaDJSAkVtq",
    "graph_sensor_locations.csv": "1BMKX2dKWm06JI1bBNFKToqWdXsHXuWfP",
    "graph_sensor_ids.txt": "1DhBE-OhIpuprbDOD09fU7agKyYfsjuSw",
}


def download_metr_la(raw_dir: str = "data/raw", quiet: bool = False) -> None:
    """Download the four METR-LA raw files if they don't already exist.

    We skip files that are already on disk so this function is safe to call
    repeatedly (e.g., at the top of a notebook or in a Makefile target).

    Args:
        raw_dir: destination directory for the downloaded files.
        quiet:   if True, suppresses gdown's per-file progress bars.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    for filename, file_id in _FILES.items():
        dest = raw_path / filename
        if dest.exists():
            if not quiet:
                print(f"  [skip] {filename} already exists")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        if not quiet:
            print(f"  [download] {filename} ...")
        gdown.download(url, str(dest), quiet=quiet)

    print(f"All METR-LA files ready in {raw_path.resolve()}")


if __name__ == "__main__":
    download_metr_la()
