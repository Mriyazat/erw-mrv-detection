"""Write a sha256 manifest of the parquet caches for Mac<->Rorqual integrity.

Usage: python scripts/hpc/make_manifest.py [--check]
  (no args) -> writes outputs/cache/MANIFEST.sha256
  --check   -> verifies current files against the manifest, exits 1 on mismatch
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR

MANIFEST = CACHE_DIR / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(CACHE_DIR.glob("*.parquet"))
    check = "--check" in sys.argv

    if check:
        if not MANIFEST.exists():
            print("No manifest to check against."); sys.exit(1)
        expected = {}
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                expected[name] = digest
        ok = True
        for f in files:
            got = sha256(f)
            exp = expected.get(f.name)
            status = "OK " if got == exp else "MISMATCH"
            if got != exp:
                ok = False
            print(f"  {status} {f.name}")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    lines = [f"{sha256(f)}  {f.name}" for f in files]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {MANIFEST} ({len(files)} files)")
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()
