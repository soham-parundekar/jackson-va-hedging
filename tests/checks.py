"""Assertion helpers, so the tests run with nothing installed beyond the project itself.

A research repository whose tests need an extra package is a repository whose tests do
not get run. These two helpers cover everything the suite needs and behave the way the
equivalents in pytest do, so the same test files also run under pytest if it happens to
be available.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class Approx:
    """Tolerant comparison for floats and arrays: ``value == approx(target, rel=1e-6)``."""

    def __init__(self, expected: Any, rel: float | None = None, abs: float | None = None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _tolerance(self, expected: float) -> float:
        rel = 1e-6 if (self.rel is None and self.abs is None) else (self.rel or 0.0)
        return max(abs(rel * expected), self.abs or 0.0)

    def __eq__(self, actual: Any) -> bool:
        expected = self.expected
        if isinstance(expected, (list, tuple, np.ndarray)) or isinstance(
            actual, (list, tuple, np.ndarray)
        ):
            actual_arr = np.asarray(actual, dtype=float)
            expected_arr = np.asarray(expected, dtype=float)
            if actual_arr.shape != expected_arr.shape:
                return False
            tolerances = np.array([self._tolerance(float(v)) for v in expected_arr.ravel()])
            return bool(
                np.all(np.abs(actual_arr.ravel() - expected_arr.ravel()) <= tolerances)
            )
        if math.isnan(float(expected)) or math.isnan(float(actual)):
            return math.isnan(float(expected)) and math.isnan(float(actual))
        return abs(float(actual) - float(expected)) <= self._tolerance(float(expected))

    def __repr__(self) -> str:
        bounds = []
        if self.rel is not None:
            bounds.append(f"rel={self.rel}")
        if self.abs is not None:
            bounds.append(f"abs={self.abs}")
        return f"approx({self.expected!r}{', ' if bounds else ''}{', '.join(bounds)})"


def approx(expected: Any, rel: float | None = None, abs: float | None = None) -> Approx:
    return Approx(expected, rel=rel, abs=abs)


class raises:
    """Context manager asserting that the block raises the given exception type."""

    def __init__(self, expected_exception, match: str | None = None):
        self.expected_exception = expected_exception
        self.match = match
        self.value: BaseException | None = None

    def __enter__(self) -> "raises":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            raise AssertionError(f"expected {self.expected_exception.__name__}, nothing raised")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match is not None and self.match not in str(exc_value):
            raise AssertionError(
                f"expected message containing {self.match!r}, got {str(exc_value)!r}"
            )
        self.value = exc_value
        return True
