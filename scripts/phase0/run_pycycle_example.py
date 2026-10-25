"""Phase 0: run pyCycle's `simple_turbojet.py` example *unmodified* and save its output.

The example (vendored verbatim in third_party/) is a ~11.8 klbf (~52 kN) full-size
single-spool turbojet -- NOT a micro turbojet. It is only a check that pyCycle + OpenMDAO
+ numpy work together in our pinned environment.

Usage:  uv run python scripts/phase0/run_pycycle_example.py
Output: validation/phase0/pycycle_simple_turbojet.txt
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "third_party" / "pycycle_examples" / "simple_turbojet.py"
OUT = ROOT / "validation" / "phase0" / "pycycle_simple_turbojet.txt"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Run in its own directory so OpenMDAO report files don't litter the repo.
    work = ROOT / "runs" / "phase0_pycycle"
    work.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE)], cwd=work, capture_output=True, text=True, check=False
    )
    dt = time.time() - t0
    OUT.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")

    print(
        f"exit code {proc.returncode}, wall time {dt:.1f} s, full output -> {OUT.relative_to(ROOT)}"
    )
    # Print the one-line performance summary of each operating point.
    lines = proc.stdout.splitlines()
    for i, line in enumerate(lines):
        if "POINT:" in line:
            print(line.strip())
        if "Mach      Alt" in line and i + 1 < len(lines):
            print("   ", line.strip())
            print("   ", lines[i + 1].strip())
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
