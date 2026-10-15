"""Signal-conditioning filters for the gaze stream.

Two filters are stacked in the runtime loop:

1. :class:`TemporalMedian` removes isolated single-frame landmark spikes.
   It runs *first* so that One-Euro's velocity estimate is not disturbed by
   outliers (a spike would otherwise look like a huge saccade and open the
   cutoff right when we want smoothing).
2. :class:`OneEuroFilter` does the actual smoothing, adaptively: heavy at
   rest, light during fast motion.

Neither depends on numpy, OpenCV or a clock source — the caller supplies
timestamps, which keeps both deterministic under test.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median

#: Smallest timestep the filter will consider. Guards against division by
#: zero and against a clock that does not advance between two frames.
MIN_DT = 1e-6


class OneEuroFilter:
    """Adaptive low-pass filter: smooth when slow, snappy when fast.

    Implements Casiez, Roussel & Vogel (CHI 2012). The cutoff frequency is
    raised in proportion to the estimated signal velocity, so slow motion
    (fixation tremor) is filtered hard while fast motion (saccades) passes
    through with minimal lag.

    Args:
        min_cutoff: Cutoff in Hz applied when the signal is stationary.
            Lower values mean a steadier cursor at rest.
        beta: Velocity coefficient. Higher values make the filter relax
            faster during motion, trading jitter for responsiveness.
        d_cutoff: Cutoff in Hz for the internal velocity estimator.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    def reset(self) -> None:
        """Forget all history; the next sample is passed through verbatim."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        """Filter sample ``x`` observed at time ``t`` (seconds)."""
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return float(x)
        # A non-advancing or backwards clock would otherwise produce a
        # negative dt and an alpha outside [0, 1], which flips the filter
        # into an amplifier.
        dt = max(t - self.t_prev, MIN_DT)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class TemporalMedian:
    """Sliding-window median — kills isolated landmark outliers.

    A median over an odd window rejects any spike shorter than half the
    window without the lag a mean of the same length would add. Backed by a
    ``deque`` and :func:`statistics.median`, which is ~20x cheaper per call
    than ``numpy.median`` at the window sizes used here (3-7 samples).
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.values: deque[float] = deque(maxlen=window)

    def update(self, x: float) -> float:
        """Append ``x`` and return the median of the current window."""
        self.values.append(float(x))
        return float(median(self.values))

    def reset(self) -> None:
        """Empty the window."""
        self.values.clear()

    def __len__(self) -> int:
        return len(self.values)


def make_axis_filters(
    min_cutoff: float,
    beta: float,
    vertical_cutoff_scale: float = 0.6,
    vertical_beta_scale: float = 0.8,
) -> tuple[OneEuroFilter, OneEuroFilter]:
    """Build the (x, y) filter pair for one smoothing preset.

    The vertical channel is filtered harder: iris travel up/down spans
    roughly half the range of left/right travel, so the same absolute
    landmark noise is proportionally larger on Y.
    """
    return (
        OneEuroFilter(min_cutoff, beta),
        OneEuroFilter(min_cutoff * vertical_cutoff_scale,
                      beta * vertical_beta_scale),
    )
