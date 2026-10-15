"""Thin-plate-spline fitting, evaluation and serialisation."""

import json
from pathlib import Path

import numpy as np
import pytest

from fatma.tps import TPSFitError, TPSMapping

GRID = [
    (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
    (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
    (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
]


def fit_identity():
    src = np.array(GRID)
    return TPSMapping(src, src[:, 0], src[:, 1])


class TestFitting:
    def test_interpolates_exactly_at_control_points(self):
        """Exactness at the control points is the defining TPS property —
        the cursor must land on a target the user actually calibrated."""
        src = np.array(GRID)
        tx = np.array([p[0] * 2 + 1 for p in GRID])
        ty = np.array([p[1] * 3 - 2 for p in GRID])
        tps = TPSMapping(src, tx, ty)
        for (x, y), ex, ey in zip(GRID, tx, ty):
            assert tps(x, y) == pytest.approx((ex, ey), abs=1e-6)

    def test_reproduces_an_affine_map_everywhere(self):
        """A TPS fitted to affine data must stay affine between the points,
        with a zero bending-energy term."""
        src = np.array(GRID)
        tx = 2.0 * src[:, 0] + 0.5 * src[:, 1] + 1.0
        ty = -1.0 * src[:, 0] + 3.0 * src[:, 1] - 0.5
        tps = TPSMapping(src, tx, ty)
        for x, y in [(0.23, 0.71), (0.9, 0.1), (0.5, 0.5)]:
            assert tps(x, y) == pytest.approx(
                (2.0 * x + 0.5 * y + 1.0, -1.0 * x + 3.0 * y - 0.5), abs=1e-6
            )

    def test_is_smooth_between_control_points(self):
        """No creases: small input steps give small output steps."""
        tps = fit_identity()
        xs = np.linspace(0.05, 0.95, 40)
        outs = np.array([tps(x, 0.5)[0] for x in xs])
        steps = np.diff(outs)
        assert np.all(steps > 0)
        assert steps.max() / steps.min() < 1.5

    def test_rejects_too_few_points(self):
        with pytest.raises(TPSFitError, match="at least 3"):
            TPSMapping(np.array([[0.0, 0.0], [1.0, 1.0]]), [0, 1], [0, 1])

    def test_rejects_coincident_points(self):
        """Nine identical readings mean the user never moved their gaze."""
        src = np.zeros((9, 2))
        with pytest.raises(TPSFitError, match="coincide"):
            TPSMapping(src, np.arange(9.0), np.arange(9.0))

    def test_rejects_non_finite_input(self):
        src = np.array(GRID, dtype=float)
        src[0, 0] = np.nan
        with pytest.raises(TPSFitError, match="NaN"):
            TPSMapping(src, src[:, 0], src[:, 1])

    def test_rejects_mismatched_target_lengths(self):
        src = np.array(GRID)
        with pytest.raises(TPSFitError, match="shape"):
            TPSMapping(src, src[:, 0][:5], src[:, 1])

    def test_rejects_wrong_src_shape(self):
        with pytest.raises(TPSFitError, match="shape"):
            TPSMapping(np.zeros((9, 3)), np.arange(9.0), np.arange(9.0))

    def test_collinear_points_are_rejected_not_silently_wrong(self):
        src = np.array([[float(i), 0.0] for i in range(5)])
        with pytest.raises(TPSFitError):
            TPSMapping(src, np.arange(5.0), np.zeros(5))


class TestEvaluation:
    def test_map_many_matches_single_point_calls(self):
        tps = fit_identity()
        pts = [(0.1, 0.2), (0.45, 0.8), (0.99, 0.01)]
        batch = tps.map_many(pts)
        for (x, y), row in zip(pts, batch):
            assert tps(x, y) == pytest.approx(tuple(row), abs=1e-9)

    def test_kernel_is_zero_at_zero_distance(self):
        """phi(0) must be 0, not log(0) = -inf."""
        tps = fit_identity()
        value = tps(*GRID[4])
        assert np.all(np.isfinite(value))

    def test_n_points_reports_the_control_count(self):
        assert fit_identity().n_points == 9


class TestSerialisation:
    def test_roundtrip_preserves_the_mapping(self):
        tps = fit_identity()
        restored = TPSMapping.from_dict(json.loads(json.dumps(tps.to_dict())))
        for x, y in [(0.0, 0.0), (0.33, 0.66), (1.0, 1.0)]:
            assert restored(x, y) == pytest.approx(tps(x, y), abs=1e-12)

    def test_rejects_coefficient_length_mismatch(self):
        d = fit_identity().to_dict()
        d["cx"] = d["cx"][:-1]
        with pytest.raises(TPSFitError, match="length mismatch"):
            TPSMapping.from_dict(d)

    def test_rejects_missing_keys(self):
        with pytest.raises(TPSFitError, match="malformed"):
            TPSMapping.from_dict({"kind": "tps", "src": [[0, 0]]})

    def test_rejects_wrong_src_shape(self):
        d = fit_identity().to_dict()
        d["src"] = [[0.0, 0.0, 0.0]] * 9
        with pytest.raises(TPSFitError, match="shape"):
            TPSMapping.from_dict(d)

    def test_loads_the_calibration_shipped_with_the_project(self):
        """The refactor must not invalidate a user's existing calibration."""
        path = Path(__file__).resolve().parent.parent / "calibration.json"
        if not path.exists():
            pytest.skip("no calibration.json in the working tree")
        data = json.loads(path.read_text())
        tps = TPSMapping.from_dict(data["tps"])
        # Evaluating at the midpoint of the recorded gaze range must land
        # somewhere on the screen, not at infinity.
        nx = sum(data["nx_range"]) / 2
        ny = sum(data["ny_range"]) / 2
        sx, sy = tps(nx, ny)
        assert np.isfinite(sx) and np.isfinite(sy)
        assert -0.5 < sx < 1.5 and -0.5 < sy < 1.5
