"""Loader for config/params.yaml with light validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Read the parameter file and check the handful of values that can silently
    wreck a run if they are wrong (fee signs, path counts, an issue age outside the
    mortality table)."""
    path = Path(path) if path is not None else paths.CONFIG / "params.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    contract = cfg["contract"]
    if contract["premium"] <= 0:
        raise ValueError("premium must be positive")
    if not 0 < contract["gawa_pct"] < 1:
        raise ValueError(f"gawa_pct outside (0, 1): {contract['gawa_pct']}")
    for fee in ("rider_charge_pct", "base_contract_charge", "fund_expense"):
        if contract[fee] < 0:
            raise ValueError(f"{fee} must not be negative")
    if not 0 <= contract["issue_age"] <= 120:
        raise ValueError(f"issue_age outside the mortality table: {contract['issue_age']}")

    mix = contract["sex_mix"]
    total = mix["male"] + mix["female"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"sex_mix must sum to 1, got {total}")

    sim = cfg["simulation"]
    if sim["n_paths"] < 1000:
        raise ValueError("n_paths below 1000 leaves too much Monte Carlo noise")
    if sim["antithetic"] and sim["n_paths"] % 2:
        raise ValueError("antithetic sampling needs an even n_paths")
    if sim["max_age"] <= contract["issue_age"]:
        raise ValueError("max_age must exceed issue_age")

    return cfg
