"""Shared test fixtures and helpers."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest


def write_profile_csv(
    path: Path,
    num_days: int = 14,
    step_minutes: int = 15,
    scale: float = 1.0,
    header: tuple[str, str] | None = ("timestamp", "consumption_kw"),
    tz_suffix: str = "",
) -> Path:
    """Write a synthetic profile with a morning bump and an evening peak around 18:00-21:00."""
    start = datetime(2024, 1, 1)
    steps = num_days * 24 * 60 // step_minutes
    rows: list[list[str]] = [list(header)] if header else []
    for step in range(steps):
        dt = start + timedelta(minutes=step_minutes * step)
        hour = dt.hour + dt.minute / 60
        base = 0.5
        if 7 <= hour <= 9:
            base = 1.8
        elif 18 <= hour <= 21:
            base = 3.5
        power = (base + 0.1 * np.sin(step)) * scale
        rows.append([dt.strftime("%Y-%m-%d %H:%M:%S") + tz_suffix, f"{power:.3f}"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


@pytest.fixture
def profile_csv(tmp_path: Path) -> Path:
    return write_profile_csv(tmp_path / "profile.csv")
