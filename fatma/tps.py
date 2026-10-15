"""Thin-plate-spline interpolation in 2D, on numpy alone.

A TPS surface passes exactly through every control point and is the
minimum-bending-energy interpolant between them, which is what we want for
gaze calibration: the mapping is exact at the points the user actually
looked at, and smooth (no creases, no overshoot) everywhere in between.

Reference: Bookstein, "Principal Warps" (IEEE PAMI, 1989). The interpolant
for one output axis is

    f(p) = a0 + a1*x + a2*y + sum_i w_i * phi(||p - p_i||)

with the thin-plate kernel phi(r) = r^2 * log(r).
"""

from __future__ import annotations

import numpy as np

#: Ridge term added to the kernel diagonal. Large enough to keep the solve
#: stable when two calibration points land almost on top of each other,
#: small enough that the interpolation stays visually exact.
DEFAULT_REGULARIZATION = 1e-9

#: Control points closer than this (in input units) are treated as
#: duplicates — a calibration where the user did not actually move their
#: gaze between targets.
MIN_POINT_SEPARATION = 1e-4


class TPSFitError(ValueError):
    """Raised when a TPS system cannot be solved from the given points."""


def _phi_from_r2(r2: np.ndarray) -> np.ndarray:
    """Thin-plate kernel evaluated from *squared* distances.

    Uses r^2*log(r) == 0.5 * r^2 * log(r^2), which skips a square root and
    keeps the r -> 0 limit (phi -> 0) exact instead of hitting log(0).
    """
    out = np.zeros_like(r2)
    mask = r2 > 1e-24
    out[mask] = 0.5 * r2[mask] * np.log(r2[mask])
    return out


class TPSMapping:
    """A fitted 2D -> 2D thin-plate spline.

    Construct from control-point correspondences, or restore a previously
    fitted mapping with :meth:`from_dict`.

    Args:
        src: ``(n, 2)`` input points.
        tx: ``(n,)`` target values for the first output axis.
        ty: ``(n,)`` target values for the second output axis.
        regularization: Ridge term on the kernel diagonal.

    Raises:
        TPSFitError: if fewer than three points are given, if any two
            points coincide, or if the linear system is singular.
    """

    __slots__ = ("src", "cx", "cy")

    def __init__(
        self,
        src=None,
        tx=None,
        ty=None,
        regularization: float = DEFAULT_REGULARIZATION,
    ) -> None:
        if src is None:
            # Uninitialised shell, filled in by from_dict().
            return

        src = np.asarray(src, dtype=float)
        if src.ndim != 2 or src.shape[1] != 2:
            raise TPSFitError(f"src must have shape (n, 2), got {src.shape}")
        n = len(src)
        if n < 3:
            raise TPSFitError(
                f"need at least 3 control points to fit a TPS, got {n}"
            )
        tx = np.asarray(tx, dtype=float)
        ty = np.asarray(ty, dtype=float)
        if tx.shape != (n,) or ty.shape != (n,):
            raise TPSFitError(
                f"tx/ty must each have shape ({n},), "
                f"got {tx.shape} and {ty.shape}"
            )
        if not (np.all(np.isfinite(src)) and np.all(np.isfinite(tx))
                and np.all(np.isfinite(ty))):
            raise TPSFitError("control points contain NaN or infinity")

        diff = src[:, None, :] - src[None, :, :]
        r2 = (diff ** 2).sum(axis=2)
        # Any off-diagonal near-zero distance means two targets produced the
        # same gaze reading — the user did not move between them, and the
        # resulting system has no unique solution. Masking the diagonal with
        # fill_diagonal rather than adding `eye(n) * inf` matters: the latter
        # puts 0 * inf = NaN in every off-diagonal cell, and the comparison
        # below then silently never fires.
        off_diagonal = r2.copy()
        np.fill_diagonal(off_diagonal, np.inf)
        if off_diagonal.min() < MIN_POINT_SEPARATION ** 2:
            raise TPSFitError(
                "two or more control points coincide; "
                "the calibration did not capture distinct gaze directions"
            )

        K = _phi_from_r2(r2)
        # Ridge belongs on the kernel block only. Adding it to the affine
        # constraint rows (as a plain `L += eye(n+3) * eps` would) perturbs
        # the exactness of the affine part for no stability benefit.
        K[np.diag_indices(n)] += regularization
        P = np.hstack([np.ones((n, 1)), src])          # (n, 3): [1, x, y]
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        # The affine constraint rows are zero-padded on the right-hand side.
        Yx = np.concatenate([tx, np.zeros(3)])
        Yy = np.concatenate([ty, np.zeros(3)])
        try:
            self.cx = np.linalg.solve(L, Yx)
            self.cy = np.linalg.solve(L, Yy)
        except np.linalg.LinAlgError as exc:
            raise TPSFitError(
                f"TPS system is singular ({exc}); control points are likely "
                "collinear"
            ) from exc
        if not (np.all(np.isfinite(self.cx)) and np.all(np.isfinite(self.cy))):
            raise TPSFitError("TPS solution diverged to NaN/infinity")
        self.src = src

    @property
    def n_points(self) -> int:
        """Number of control points backing this mapping."""
        return len(self.src)

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        """Evaluate the mapping at a single point."""
        dx = self.src[:, 0] - x
        dy = self.src[:, 1] - y
        phi = _phi_from_r2(dx * dx + dy * dy)
        n = len(phi)
        # f = phi . w + a0 + a1*x + a2*y, with the affine terms living in
        # the last three coefficients.
        cx, cy = self.cx, self.cy
        fx = float(phi @ cx[:n] + cx[n] + cx[n + 1] * x + cx[n + 2] * y)
        fy = float(phi @ cy[:n] + cy[n] + cy[n + 1] * x + cy[n + 2] * y)
        return fx, fy

    def map_many(self, pts) -> np.ndarray:
        """Evaluate at many points at once. ``pts`` is ``(m, 2)``."""
        pts = np.asarray(pts, dtype=float)
        diff = pts[:, None, :] - self.src[None, :, :]
        phi = _phi_from_r2((diff ** 2).sum(axis=2))          # (m, n)
        feats = np.hstack([phi, np.ones((len(pts), 1)), pts])  # (m, n+3)
        return np.column_stack([feats @ self.cx, feats @ self.cy])

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "kind": "tps",
            "src": self.src.tolist(),
            "cx": self.cx.tolist(),
            "cy": self.cy.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TPSMapping":
        """Restore a mapping produced by :meth:`to_dict`.

        Raises:
            TPSFitError: if the dict is missing keys or the coefficient
                vectors do not match the control-point count.
        """
        try:
            src = np.array(d["src"], dtype=float)
            cx = np.array(d["cx"], dtype=float)
            cy = np.array(d["cy"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise TPSFitError(f"malformed TPS payload: {exc}") from exc
        if src.ndim != 2 or src.shape[1] != 2:
            raise TPSFitError(f"src must have shape (n, 2), got {src.shape}")
        expected = len(src) + 3
        if cx.shape != (expected,) or cy.shape != (expected,):
            raise TPSFitError(
                f"coefficient length mismatch: expected {expected}, got "
                f"cx={cx.shape} cy={cy.shape}"
            )
        obj = cls()
        obj.src = src
        obj.cx = cx
        obj.cy = cy
        return obj
