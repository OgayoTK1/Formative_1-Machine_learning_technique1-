"""
src/data/download_dataverse.py

Resumable downloader for the Milan telecommunications activity dataset
(Harvard Dataverse, DOI 10.7910/DVN/EGZHFV).

Verified against one file (sms-call-internet-mi-2013-11-01.txt, file_id 2674255):
the POST to /api/access/datafile/{id} with a guestbookResponse payload returns
a JSON envelope {"status": "OK", "data": {"signedUrl": ...}}, not the file
bytes directly. The actual file must be fetched with a second GET against
that signedUrl.

This module:
- reads the file listing already saved at configs/dataverse_raw_metadata.json
  (or re-queries the API if that file is missing)
- skips files that already exist locally with the correct size
- downloads missing/incomplete files via the two-step (POST -> signedUrl -> GET) flow
- logs every attempt (success/failure, bytes, elapsed time) to a CSV
- is safe to re-run after an interruption: completed files are skipped
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import requests

DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
PERSISTENT_ID = "doi:10.7910/DVN/EGZHFV"

GUESTBOOK_INFO = {
    "name": "Ogayo Andrew",
    "email": "a.ogayo@alustudent.com",
    "institution": "African Leadership University",
    "position": "Student",
}


def get_file_listing(metadata_path: Path) -> list[dict]:
    """Load the file listing from the saved metadata JSON."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    files = metadata["data"]["latestVersion"]["files"]
    listing = []
    for entry in files:
        df = entry["dataFile"]
        listing.append(
            {
                "file_id": df.get("id"),
                "filename": df.get("filename"),
                "expected_size": df.get("filesize"),
            }
        )
    return listing


def already_downloaded(out_path: Path, expected_size: int) -> bool:
    """A file counts as done only if it exists AND matches the expected byte size."""
    if not out_path.exists():
        return False
    return out_path.stat().st_size == expected_size


def download_one_file(file_id: int, filename: str, out_dir: Path, expected_size: int) -> dict:
    """Download a single file via the POST -> signedUrl -> GET flow. Returns a log record."""
    out_path = out_dir / filename

    if already_downloaded(out_path, expected_size):
        return {
            "filename": filename,
            "file_id": file_id,
            "status": "skipped_already_complete",
            "bytes_downloaded": out_path.stat().st_size,
            "elapsed_seconds": 0.0,
            "error": "",
        }

    access_url = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"
    start = time.perf_counter()

    try:
        envelope_response = requests.post(
            access_url, json={"guestbookResponse": GUESTBOOK_INFO}, timeout=60
        )
        envelope_response.raise_for_status()
        envelope = envelope_response.json()
        signed_url = envelope["data"]["signedUrl"]

        file_response = requests.get(signed_url, stream=True, timeout=300)
        file_response.raise_for_status()

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        tmp_path.rename(out_path)

        elapsed = time.perf_counter() - start
        actual_size = out_path.stat().st_size

        if actual_size != expected_size:
            return {
                "filename": filename,
                "file_id": file_id,
                "status": "size_mismatch",
                "bytes_downloaded": actual_size,
                "elapsed_seconds": round(elapsed, 2),
                "error": f"expected {expected_size}, got {actual_size}",
            }

        return {
            "filename": filename,
            "file_id": file_id,
            "status": "success",
            "bytes_downloaded": actual_size,
            "elapsed_seconds": round(elapsed, 2),
            "error": "",
        }

    except Exception as exc:  # noqa: BLE001 - we want to log and continue, not crash the batch
        elapsed = time.perf_counter() - start
        return {
            "filename": filename,
            "file_id": file_id,
            "status": "failed",
            "bytes_downloaded": 0,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(exc),
        }


def download_all(listing: list[dict], out_dir: Path, log_path: Path) -> list[dict]:
    """Download every file in the listing, logging each attempt. Resumable across reruns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_records = []

    for i, item in enumerate(listing, start=1):
        print(f"[{i}/{len(listing)}] {item['filename']} ... ", end="", flush=True)
        record = download_one_file(
            item["file_id"], item["filename"], out_dir, item["expected_size"]
        )
        print(record["status"])
        log_records.append(record)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_records[0].keys())
        writer.writeheader()
        writer.writerows(log_records)

    n_success = sum(1 for r in log_records if r["status"] in ("success", "skipped_already_complete"))
    n_failed = len(log_records) - n_success
    print(f"\nDone. {n_success}/{len(log_records)} files OK, {n_failed} failed.")
    print(f"Log written to: {log_path}")

    return log_records
