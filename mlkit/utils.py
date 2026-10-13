"""Small utilities: seeding, terminal tables, learning-curve rendering, timing."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np

# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------

_seed_counter = 0


def set_seed(seed: int) -> np.random.Generator:
    """Seed python + numpy global RNGs and return a fresh Generator."""
    global _seed_counter
    _seed_counter += 1
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed + 1000 * _seed_counter)


# --------------------------------------------------------------------------
# terminal formatting
# --------------------------------------------------------------------------

RULE = "─"


def section(title: str) -> None:
    print(f"\n\x1b[1m{title}\x1b[0m")
    print(RULE * max(40, len(title)))


def kv(pairs: dict, indent: int = 2) -> None:
    """Print aligned key/value pairs."""
    if not pairs:
        return
    width = max(len(str(k)) for k in pairs)
    for k, v in pairs.items():
        print(" " * indent + f"{str(k).ljust(width)} : {v}")


def table(headers: list[str], rows: list[list], indent: int = 0) -> None:
    """Print a simple bordered-less table."""
    rows = [[_cell(c) for c in r] for r in rows]
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    pad = " " * indent
    print(pad + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print(pad + "  ".join(RULE * widths[i] for i in range(len(headers))))
    for r in rows:
        print(pad + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def _cell(v) -> str:
    if isinstance(v, float):
        if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.4e}"
        return f"{v:.4f}"
    return str(v)


SPARK = "▁▂▃▄▅▆▇█"


def curve(values, width: int = 50) -> str:
    """Render a 1-D array as an ASCII sparkline (for loss curves)."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return ""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ""
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    # resample to `width` points
    idx = np.linspace(0, v.size - 1, min(width, v.size)).astype(int)
    return "".join(SPARK[int(round((v[i] - lo) / span * (len(SPARK) - 1)))] for i in idx)


def series(values, width: int = 50) -> str:
    """First value -> last value plus sparkline."""
    v = list(values)
    if not v:
        return ""
    return f"{v[0]:.4f} → {v[-1]:.4f}  {curve(v, width)}"


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


@contextmanager
def timer(label: str = "", fmt: str = "{label}: {ms:.1f} ms"):
    t0 = time.perf_counter()
    out = {}
    yield out
    dt = time.perf_counter() - t0
    out["seconds"] = dt
    print(fmt.format(label=label, ms=dt * 1000, s=dt))


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def fmt_num(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)