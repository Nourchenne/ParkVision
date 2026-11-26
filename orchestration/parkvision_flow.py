"""Prefect flow that orchestrates the ParkVision DVC data pipeline.

Usage:
    prefect server start                    # terminal 1, laisser tourner
    python orchestration/parkvision_flow.py # terminal 2
"""

import subprocess
import sys
from pathlib import Path

from prefect import flow, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dvc_command(stage: str) -> list[str]:
    """Build the `dvc repro` command for one stage of dvc.yaml."""
    return [sys.executable, "-m", "dvc", "repro", stage]


def run_stage(stage: str) -> None:
    """Run one DVC stage from the project root (DVC skips it if up to date)."""
    subprocess.run(dvc_command(stage), cwd=PROJECT_ROOT, check=True)


@task(name="prepare_split", retries=1)
def run_prepare_split() -> None:
    run_stage("prepare_split")


@task(name="prepare_yolo", retries=1)
def run_prepare_yolo() -> None:
    run_stage("prepare_yolo")


@flow(name="parkvision-pipeline")
def parkvision_pipeline() -> None:
    run_prepare_split()
    run_prepare_yolo()


if __name__ == "__main__":
    parkvision_pipeline()
