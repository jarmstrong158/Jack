"""Tests for Jack's pure logic.

Jack is the validated single-facility predecessor that Clark's headline
generalization claim is benchmarked against, so its numbers are load-bearing
for two repos. It had no tests at all. These cover the parts that are
deterministic enough to assert on: action encoding, state normalization, and
the invariants of order arrival. The RL training loop itself is not unit
testable in any useful way and is deliberately not faked here.
"""
import numpy as np
import pytest

from volt_sim.config import NUM_TASKS, NUM_WORKERS, TASKS, TASK_TO_IDX
from volt_sim.agent.actions import (
    ACTION_HEAD_SIZE, NUM_ACTION_HEADS, decode_actions,
)
from volt_sim.agent.state import (
    FULL_STATE_SIZE, RunningStats, normalize_state, validate_state,
)
from volt_sim.env.order_arrival import (
    generate_arrival_schedule, is_high_volume_day,
)


# --------------------------------------------------------------------------- #
# action space
# --------------------------------------------------------------------------- #
class TestActionEncoding:
    def test_head_size_is_two_actions_per_task(self):
        """Every task is reachable with and without hustle, and nothing else is.

        This pins the number the docstrings disagreed about: the module header
        said 14 per head while decode_actions' docstring and the ACTION_HEAD_SIZE
        comment said 12, left over from when there were six tasks and before
        cycle_count was added. 14 is correct.
        """
        assert NUM_TASKS == len(TASKS)
        assert ACTION_HEAD_SIZE == NUM_TASKS * 2
        assert NUM_ACTION_HEADS == NUM_WORKERS

    def test_decode_assigns_every_worker_exactly_once(self):
        out = decode_actions([0] * NUM_WORKERS)
        assert [w for w, _, _ in out] == list(range(NUM_WORKERS))

    def test_low_half_is_no_hustle_high_half_is_hustle(self):
        """action < NUM_TASKS means no hustle; >= means hustle, same task."""
        for task in range(NUM_TASKS):
            (_, plain_task, plain_hustle), = decode_actions([task])
            (_, hustle_task, hustle_flag), = decode_actions([task + NUM_TASKS])
            assert (plain_task, plain_hustle) == (task, False)
            assert (hustle_task, hustle_flag) == (task, True)

    def test_every_action_in_range_decodes_to_a_real_task(self):
        for action in range(ACTION_HEAD_SIZE):
            (_, task, _), = decode_actions([action])
            assert 0 <= task < NUM_TASKS
            assert TASKS[task] in TASK_TO_IDX

    def test_task_index_round_trips_through_the_name_table(self):
        for name, idx in TASK_TO_IDX.items():
            assert TASKS[idx] == name


# --------------------------------------------------------------------------- #
# state normalization
# --------------------------------------------------------------------------- #
class TestStateValidation:
    def test_correct_shape_and_finite_passes(self):
        assert validate_state(np.zeros(FULL_STATE_SIZE, dtype=np.float32))

    def test_wrong_shape_fails(self):
        assert not validate_state(np.zeros(FULL_STATE_SIZE + 1, dtype=np.float32))

    def test_nan_fails(self):
        """A NaN reaching the policy poisons every downstream gradient, so this
        is the one check that must not be merely advisory."""
        s = np.zeros(FULL_STATE_SIZE, dtype=np.float32)
        s[0] = np.nan
        assert not validate_state(s)


class TestNormalizeState:
    def test_zero_variance_does_not_divide_by_zero(self):
        """Constant features are real (flags that never flip during a warmup),
        and an unguarded divide would emit inf/NaN into the policy."""
        state = np.ones(4, dtype=np.float32)
        out = normalize_state(state, np.ones(4), np.zeros(4))
        assert np.all(np.isfinite(out))

    def test_output_is_clipped_both_directions(self):
        state = np.array([1e6, -1e6], dtype=np.float32)
        out = normalize_state(state, np.zeros(2), np.ones(2), clip=10.0)
        assert out.max() <= 10.0 and out.min() >= -10.0

    def test_centres_on_the_mean(self):
        state = np.array([5.0, 5.0], dtype=np.float32)
        out = normalize_state(state, np.array([5.0, 5.0]), np.ones(2))
        assert np.allclose(out, 0.0)


class TestRunningStats:
    def test_welford_matches_numpy_on_the_same_sample(self):
        """Welford is used because it is numerically stable online, but it still
        has to agree with the batch computation it stands in for."""
        rng = np.random.default_rng(0)
        xs = rng.normal(size=(200, 3)).astype(np.float32)
        rs = RunningStats(3)
        for x in xs:
            rs.update(x)
        assert np.allclose(rs.mean, xs.mean(axis=0), atol=1e-4)
        assert np.allclose(rs.var, xs.var(axis=0, ddof=1), atol=1e-3)

    def test_single_sample_keeps_unit_variance(self):
        """With n == 1 the sample variance is undefined; it must stay at the
        initial 1.0 rather than becoming 0 and blowing up the divisor."""
        rs = RunningStats(2)
        rs.update(np.array([3.0, 4.0], dtype=np.float32))
        assert np.allclose(rs.var, 1.0)
        assert np.all(np.isfinite(rs.normalize(np.array([3.0, 4.0]))))


# --------------------------------------------------------------------------- #
# order arrival
# --------------------------------------------------------------------------- #
class TestOrderArrival:
    @pytest.mark.parametrize("total", [1, 37, 250, 1000])
    @pytest.mark.parametrize("high_volume", [False, True])
    def test_every_order_is_scheduled_exactly_once(self, total, high_volume):
        """The schedule is a partition of the day's orders. Dropping or
        duplicating here would silently change the workload the agent is
        graded on, which is exactly the kind of drift a completion-rate
        headline cannot survive."""
        sched = generate_arrival_schedule(total, high_volume, eod_hour=17.0)
        assert sum(sched.values()) == total

    @pytest.mark.parametrize("high_volume", [False, True])
    def test_nothing_arrives_after_the_arrival_cutoff(self, high_volume):
        """Orders must land before OT starts, or the agent is scored on work it
        could not have begun."""
        from volt_sim.env.order_arrival import ORDER_ARRIVAL_END_HOUR
        sched = generate_arrival_schedule(500, high_volume, eod_hour=17.0)
        arriving = [h for h, n in sched.items() if n > 0]
        assert arriving, "schedule placed no orders at all"
        assert max(arriving) <= ORDER_ARRIVAL_END_HOUR

    def test_no_negative_counts(self):
        sched = generate_arrival_schedule(300, False, eod_hour=17.0)
        assert all(n >= 0 for n in sched.values())

    def test_zero_orders_is_a_valid_empty_day(self):
        sched = generate_arrival_schedule(0, False, eod_hour=17.0)
        assert sum(sched.values()) == 0


class TestHighVolumeDay:
    def test_threshold_is_inclusive_at_the_percentile(self):
        from volt_sim.env.order_arrival import HIGH_VOLUME_PERCENTILE
        lo, hi = 100, 200
        threshold = lo + (hi - lo) * HIGH_VOLUME_PERCENTILE
        assert is_high_volume_day(int(threshold) + 1, (lo, hi))
        assert not is_high_volume_day(int(threshold) - 1, (lo, hi))

    def test_bottom_of_range_is_never_high_volume(self):
        assert not is_high_volume_day(100, (100, 200))

    def test_top_of_range_always_is(self):
        assert is_high_volume_day(200, (100, 200))
