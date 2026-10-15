"""One-Euro and temporal-median behaviour."""

import math
import statistics

import pytest

from fatma.filters import OneEuroFilter, TemporalMedian, make_axis_filters


def feed(filt, values, dt=1 / 30):
    """Run a sequence through a filter on a regular clock."""
    return [filt(v, i * dt) for i, v in enumerate(values)]


class TestOneEuroFilter:
    def test_first_sample_passes_through_unchanged(self):
        assert OneEuroFilter()(4.2, 0.0) == 4.2

    def test_converges_to_a_constant_input(self):
        out = feed(OneEuroFilter(min_cutoff=1.0, beta=0.0), [5.0] * 60)
        assert out[-1] == pytest.approx(5.0, abs=1e-9)

    def test_output_lies_between_previous_output_and_new_input(self):
        """The filter is a convex blend, so it can never overshoot."""
        filt = OneEuroFilter(min_cutoff=0.3, beta=0.0)
        filt(0.0, 0.0)
        prev = 0.0
        for i in range(1, 30):
            out = filt(1.0, i / 30)
            assert prev <= out <= 1.0
            prev = out

    def test_smooths_noise(self):
        noise = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        out = feed(OneEuroFilter(min_cutoff=0.3, beta=0.0), noise)
        assert statistics.pstdev(out[1:]) < statistics.pstdev(noise[1:])

    def test_higher_beta_tracks_a_step_faster(self):
        """The adaptive cutoff is the whole point: fast motion must not lag."""
        step = [0.0] * 5 + [1.0] * 15
        slow = feed(OneEuroFilter(min_cutoff=0.3, beta=0.0), step)
        fast = feed(OneEuroFilter(min_cutoff=0.3, beta=5.0), step)
        assert fast[-1] > slow[-1]

    def test_survives_a_clock_that_does_not_advance(self):
        """Two frames inside the same timestamp must not divide by zero.

        With no elapsed time the filter holds its previous estimate, which
        is the consistent reading of the model — what it must not do is
        produce a NaN or an unbounded value.
        """
        filt = OneEuroFilter()
        filt(1.0, 10.0)
        out = filt(2.0, 10.0)
        assert math.isfinite(out)
        assert 1.0 <= out <= 2.0

    def test_survives_a_clock_that_goes_backwards(self):
        filt = OneEuroFilter()
        filt(1.0, 10.0)
        out = filt(2.0, 9.0)
        assert math.isfinite(out)
        assert 1.0 <= out <= 2.0

    def test_reset_restores_passthrough(self):
        filt = OneEuroFilter(min_cutoff=0.3)
        feed(filt, [0.0] * 20)
        filt.reset()
        assert filt(9.0, 100.0) == 9.0

    @pytest.mark.parametrize("kwargs", [
        {"min_cutoff": 0.0}, {"min_cutoff": -1.0},
        {"d_cutoff": 0.0}, {"beta": -0.1},
    ])
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            OneEuroFilter(**kwargs)


class TestTemporalMedian:
    def test_single_frame_spike_is_rejected(self):
        """The reason this filter exists: one bad landmark frame."""
        med = TemporalMedian(window=5)
        for _ in range(5):
            med.update(0.0)
        assert med.update(100.0) == 0.0

    def test_a_real_move_gets_through_once_it_fills_the_window(self):
        med = TemporalMedian(window=5)
        for _ in range(5):
            med.update(0.0)
        outs = [med.update(1.0) for _ in range(5)]
        assert outs[-1] == 1.0

    def test_window_never_exceeds_its_size(self):
        med = TemporalMedian(window=3)
        for i in range(10):
            med.update(float(i))
        assert len(med) == 3

    def test_partial_window_uses_what_is_there(self):
        med = TemporalMedian(window=5)
        assert med.update(2.0) == 2.0
        assert med.update(4.0) == 3.0

    def test_reset_empties_the_window(self):
        med = TemporalMedian(window=3)
        med.update(1.0)
        med.reset()
        assert len(med) == 0
        assert med.update(7.0) == 7.0

    def test_window_of_one_is_a_passthrough(self):
        med = TemporalMedian(window=1)
        assert med.update(3.0) == 3.0
        assert med.update(99.0) == 99.0

    def test_rejects_invalid_window(self):
        with pytest.raises(ValueError):
            TemporalMedian(window=0)


class TestMakeAxisFilters:
    def test_vertical_channel_is_filtered_harder(self):
        """Y carries less range than X, so it gets a lower cutoff."""
        fx, fy = make_axis_filters(1.0, 0.5, 0.6, 0.8)
        assert fy.min_cutoff == pytest.approx(0.6)
        assert fy.beta == pytest.approx(0.4)
        assert fx.min_cutoff == pytest.approx(1.0)
        assert fx.beta == pytest.approx(0.5)

    def test_filters_are_independent_objects(self):
        fx, fy = make_axis_filters(1.0, 0.5)
        fx(1.0, 0.0)
        assert fy.x_prev is None
