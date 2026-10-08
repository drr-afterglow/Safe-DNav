"""Atomic per-episode JSON and append-only CSV reporting."""

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable


def result_basename(scenario_name: str, run_id: str = "") -> str:
    return scenario_name if not run_id else f"{scenario_name}__{run_id}"


def write_trajectory(output_directory: str, scenario_name: str,
                     rows: Iterable[Dict[str, object]], run_id: str = "") -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/f"{result_basename(scenario_name, run_id)}__trajectory.csv"
    rows = list(rows)
    if not rows:
        raise ValueError("trajectory must contain at least one sample")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_episode_result(output_directory: str, scenario_name: str,
                         result: Dict[str, object], run_id: str = "") -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"scenario": scenario_name, "run_id": run_id, **result}
    result_name = result_basename(scenario_name, run_id)
    destination = directory/f"{result_name}.json"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result_name}.", suffix=".tmp", dir=str(directory)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    summary = directory/"scenario_results.csv"
    exists = summary.exists()
    with summary.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(payload.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(payload)
    return destination
