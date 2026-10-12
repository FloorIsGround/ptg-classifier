from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        _ = end - start
