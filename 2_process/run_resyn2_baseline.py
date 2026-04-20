#!/usr/bin/env python3
"""Run ABC resyn2 baseline on a set of AIG circuits and save LUT6/Level stats.

This is intended for fair comparison against HybridSYN:
- same input circuits
- same ABC binary
- same 6-LUT mapping step: if -K 6
- same extracted metrics: LUT6 count and mapped level
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ABC_BINARY = SCRIPT_DIR / "abc" / "abc"
DEFAULT_ABC_RC = SCRIPT_DIR / "abc" / "abc.rc"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def parse_stats(text: str) -> dict[str, float | None]:
    nd_matches = re.findall(r"\bnd\s*=\s*(\d+)", text)
    lev_matches = re.findall(r"\blev\s*=\s*(\d+)", text)
    return {
        "lut6_count": float(nd_matches[-1]) if nd_matches else None,
        "lut6_lev": float(lev_matches[-1]) if lev_matches else None,
    }


def run_resyn2_on_circuit(abc_binary: Path, abc_rc: Path, circuit_file: Path, timeout_s: int) -> dict[str, Any]:
    script = f"source {abc_rc}; read_aiger {circuit_file}; strash; resyn2; if -K 6; print_stats"
    proc = subprocess.run(
        [str(abc_binary), "-c", script],
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    stats = parse_stats(text)
    return {
        "circuit": circuit_file.name,
        "circuit_path": str(circuit_file),
        "returncode": proc.returncode,
        "success": proc.returncode == 0 and stats["lut6_count"] is not None and stats["lut6_lev"] is not None,
        "lut6_count": stats["lut6_count"],
        "lut6_lev": stats["lut6_lev"],
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABC resyn2 baseline on AIG circuits")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="../1_inputs/EPFL_benchmarks/arithmetic/*.aig",
        help="Glob of AIG circuits to evaluate",
    )
    parser.add_argument(
        "--abc-binary",
        type=str,
        default=str(DEFAULT_ABC_BINARY),
        help="Path to ABC binary",
    )
    parser.add_argument(
        "--abc-rc",
        type=str,
        default=str(DEFAULT_ABC_RC),
        help="Path to abc.rc so resyn2 aliases are available",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="../3_outputs/resyn2_baseline.csv",
        help="CSV path for baseline results",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=60,
        help="Timeout per circuit in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    abc_binary = resolve_path(args.abc_binary)
    if not abc_binary.exists():
        raise FileNotFoundError(f"ABC binary not found: {abc_binary}")
    abc_rc = resolve_path(args.abc_rc)
    if not abc_rc.exists():
        raise FileNotFoundError(f"ABC rc file not found: {abc_rc}")

    raw_pattern = Path(args.input_glob)
    if raw_pattern.is_absolute():
        pattern = str(raw_pattern)
    else:
        pattern = str((SCRIPT_DIR / raw_pattern).resolve())

    matches = sorted(glob.glob(pattern))
    if not matches and any(ch in args.input_glob for ch in "*?["):
        matches = sorted(glob.glob(args.input_glob))

    circuit_files = [Path(item).resolve() for item in matches if Path(item).is_file()]

    circuit_files = [c.resolve() for c in circuit_files if c.is_file()]
    if not circuit_files:
        raise FileNotFoundError(f"No circuits matched: {args.input_glob}")

    rows: list[dict[str, Any]] = []
    for circuit_file in circuit_files:
        result = run_resyn2_on_circuit(abc_binary, abc_rc, circuit_file, args.timeout_s)
        rows.append(result)
        status = "OK" if result["success"] else "FAIL"
        print(
            f"[{status}] {circuit_file.name} | LUT6={result['lut6_count']} | Level={result['lut6_lev']}"
        )

    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "circuit",
            "circuit_path",
            "returncode",
            "success",
            "lut6_count",
            "lut6_lev",
            "stdout_tail",
            "stderr_tail",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(1 for row in rows if row["success"])
    print(f"Saved CSV: {output_csv}")
    print(f"Completed: {success_count}/{len(rows)} successful")


if __name__ == "__main__":
    main()
