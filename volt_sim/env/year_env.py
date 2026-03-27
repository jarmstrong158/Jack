"""
Year-level environment wrapper.
One episode = one full year (~260 work days).
Manages carryover between days: restock level, management backlog, cycle counts.
Delegates intra-day simulation to WarehouseEnv.
"""
import random
import calendar
import numpy as np
from typing import Optional

from volt_sim.config import (
    ORDER_VOLUME_RANGES, MONTH_TO_SEASON, SEASONS,
    WEEKLY_VOLUME_CURVE, MANAGEMENT_MIN_DAILY_HOURS, MARCUS_MANAGEMENT_HOURS_REQUIRED,
    MANAGEMENT_BACKLOG_WEEK_THRESHOLD, MANAGEMENT_BACKLOG_WEEKLY_PENALTY,
    CYCLE_COUNTS_PER_WEEK, CYCLE_COUNT_HOURS,
    CYCLE_COUNT_SLIP_PENALTY_1, CYCLE_COUNT_SLIP_PENALTY_2,
    CYCLE_COUNT_MONTH_MISS_PENALTY,
    RESTOCK_STARTING_LEVEL, NUM_WORKERS, NUM_TASKS,
    HUSTLE_WEEKLY_THRESHOLDS,
    REWARDS,
)
from volt_sim.env.warehouse_env import WarehouseEnv


class YearEnv:
    """
    Wraps WarehouseEnv to run a full calendar year.
    The agent still makes 15-minute decisions — same action space.
    Year-level state (backlog, cycle counts, etc.) is appended to the state vector.
    """

    # Extra state variables beyond the daily env
    YEAR_STATE_SIZE = 7  # day_of_year, week_of_year, mgmt_backlog, cycle_counts_done_this_week,
                         # cycle_counts_overdue, month_progress, is_monday

    def __init__(self):
        self.day_env = WarehouseEnv()

        # Year-level state
        self.year: int = 0
        self.work_days: list[dict] = []  # pre-generated schedule
        self.current_day_idx: int = 0
        self.total_work_days: int = 0

        # Carryover mechanics
        self.restock_level: float = RESTOCK_STARTING_LEVEL
        self.management_backlog: float = 0.0  # accumulated missed mgmt hours
        self.cycle_counts_done_this_week: int = 0
        self.cycle_counts_overdue: int = 0  # from previous weeks

        # Weekly hustle tracking — resets every Monday
        self.weekly_hustle_hours: dict[int, float] = {i: 0.0 for i in range(NUM_WORKERS)}
        self.hustle_exhausted: dict[int, bool] = {i: False for i in range(NUM_WORKERS)}

        # Tracking
        self.year_reward: float = 0.0
        self.year_reward_breakdown: dict = {}
        self.daily_grades: list[str] = []
        self.daily_summaries: list[dict] = []
        self.is_year_done: bool = False

        # Week tracking
        self.current_week: int = 0

    def reset(self) -> np.ndarray:
        """Start a new year. Returns initial state vector."""
        self.year = random.randint(2024, 2026)
        self.work_days = self._generate_year_schedule()
        self.total_work_days = len(self.work_days)
        self.current_day_idx = 0

        # Reset carryover
        self.restock_level = RESTOCK_STARTING_LEVEL
        self.management_backlog = 0.0
        self.mgmt_carryover_hours = 0.0
        self.cycle_counts_done_this_week = 0
        self.cycle_counts_overdue = 0
        self.current_week = 0

        # Reset weekly hustle tracking
        self.weekly_hustle_hours = {i: 0.0 for i in range(NUM_WORKERS)}
        self.hustle_exhausted = {i: False for i in range(NUM_WORKERS)}

        # Reset tracking
        self.year_reward = 0.0
        self.year_reward_breakdown = {}
        self.daily_grades = []
        self.daily_summaries = []
        self.is_year_done = False

        # Start first day
        return self._start_new_day()

    def step(self, actions: list[tuple[int, int]]) -> tuple[np.ndarray, float, bool, dict]:
        """
        Same interface as WarehouseEnv.step().
        Returns (state, reward, done, info).
        When a day ends, automatically starts the next day.
        """
        state, reward, day_done, info = self.day_env.step(actions)

        if day_done:
            # End-of-day processing
            day_reward = self._process_end_of_day()
            reward += day_reward

            self.year_reward += self.day_env.total_reward + day_reward
            self.current_day_idx += 1

            if self.current_day_idx >= self.total_work_days:
                # Year is over
                year_end_reward = self._process_end_of_year()
                reward += year_end_reward
                self.is_year_done = True
                info["year_done"] = True
                info["year_summary"] = self._get_year_summary()
                return self._get_year_state(), reward, True, info

            # Start next day — returns already-augmented state
            state = self._start_new_day()
            info["new_day"] = True
            info["day_number"] = self.current_day_idx + 1
            return state, reward, False, info

        return self._augment_state(state), reward, False, info

    def _generate_year_schedule(self) -> list[dict]:
        """Generate all ~260 work days for the year with volumes."""
        schedule = []

        for month in range(1, 13):
            month_name = calendar.month_name[month]
            season = MONTH_TO_SEASON[month]
            vol_range = ORDER_VOLUME_RANGES[month_name]
            vol_lo, vol_hi = vol_range
            vol_spread = vol_hi - vol_lo

            # Get work days in this month (Mon-Fri)
            cal = calendar.Calendar()
            for day, dow in cal.itermonthdays2(self.year, month):
                if day == 0:
                    continue  # padding from other months
                if dow > 4:
                    continue  # skip weekends

                # Apply weekly volume curve
                pct_lo, pct_hi = WEEKLY_VOLUME_CURVE[dow]
                day_vol_lo = vol_lo + int(vol_spread * pct_lo)
                day_vol_hi = vol_lo + int(vol_spread * pct_hi)
                daily_volume = random.randint(day_vol_lo, max(day_vol_lo, day_vol_hi))

                schedule.append({
                    "year": self.year,
                    "month": month,
                    "month_name": month_name,
                    "day": day,
                    "day_of_week": dow,
                    "season": season,
                    "volume": daily_volume,
                    "vol_range": vol_range,
                })

        return schedule

    def _start_new_day(self) -> np.ndarray:
        """Initialize the daily env for the current day with carryover state."""
        day_info = self.work_days[self.current_day_idx]

        # Check for new week (Monday)
        if day_info["day_of_week"] == 0:
            self._process_new_week()

        # Reset daily env with today's parameters
        state = self.day_env.reset(
            force_month=day_info["month"],
            force_dow=day_info["day_of_week"],
            force_volume=day_info["volume"],
        )

        # Apply carryover: restock level from previous day
        self.day_env.restock_level = self.restock_level

        # Pass backlog to day env so action mask knows if management cap should lift
        self.day_env._mgmt_backlog = self.management_backlog

        # Apply management carryover: if yesterday's min wasn't met,
        # Marcus/Nolan must catch up before doing anything else.
        # This burns their available hours at day start.
        if self.mgmt_carryover_hours > 0:
            for w in self.day_env.episode.workers:
                if w.worker_id in (0, 1):  # Marcus or Nolan
                    catch_up = min(self.mgmt_carryover_hours, 1.0)  # split between them
                    w.management_hours += catch_up
                    w.hours_worked += catch_up
                    self.mgmt_carryover_hours -= catch_up
                    if self.mgmt_carryover_hours <= 0:
                        break

        # Inject weekly hustle state so each worker knows their weekly pressure
        for w in self.day_env.episode.workers:
            w.weekly_hustle_hours = self.weekly_hustle_hours.get(w.worker_id, 0.0)
            w.is_hustle_exhausted = self.hustle_exhausted.get(w.worker_id, False)

        return self._augment_state(state)

    def _process_end_of_day(self) -> float:
        """Handle end-of-day carryover and penalties. Returns extra reward."""
        reward = 0.0

        # Carry restock level to next day
        self.restock_level = self.day_env.restock_level

        # Management backlog — track against 4h daily target
        # Under 4h: shortfall adds to backlog. Over 4h: surplus burns backlog down.
        # When both Marcus + Nolan are absent, Felix's hours count for this calculation.
        total_mgmt = self.day_env._get_effective_management_hours()
        delta = MARCUS_MANAGEMENT_HOURS_REQUIRED - total_mgmt  # positive = shortfall, negative = surplus
        self.management_backlog = max(0.0, self.management_backlog + delta)

        # If under 1.5h minimum, next day loses hours to catch up
        min_shortfall = max(0.0, MANAGEMENT_MIN_DAILY_HOURS - total_mgmt)
        self.mgmt_carryover_hours = min_shortfall  # hours deducted from start of next day

        # Daily escalating backlog penalty — the more it piles up, the louder it gets
        if self.management_backlog > 0:
            # Quadratic scaling: 5h backlog = -2.5, 10h = -10, 20h = -40, 50h = -250
            penalty = -0.1 * (self.management_backlog ** 2) / 10.0
            reward += penalty
            self._add_year_reward("management_backlog_daily", penalty)

        # Accumulate weekly hustle hours and check for exhaustion
        for w in self.day_env.episode.workers:
            if w.hustle_hours_today > 0:
                self.weekly_hustle_hours[w.worker_id] = (
                    self.weekly_hustle_hours.get(w.worker_id, 0.0) + w.hustle_hours_today
                )
                threshold = HUSTLE_WEEKLY_THRESHOLDS.get(w.worker_id, 16.0)
                if (not self.hustle_exhausted.get(w.worker_id, False) and
                        self.weekly_hustle_hours[w.worker_id] >= threshold):
                    self.hustle_exhausted[w.worker_id] = True

        # Record daily summary
        summary = self.day_env.get_episode_summary(management_backlog=self.management_backlog)
        self.daily_grades.append(summary["footer"]["grade"])
        self.daily_summaries.append(summary)

        return reward

    def _process_new_week(self) -> float:
        """Called on Monday — check previous week's cycle counts and backlog."""
        reward = 0.0

        if self.current_week > 0:
            # Check cycle count compliance for previous week
            missed = max(0, CYCLE_COUNTS_PER_WEEK - self.cycle_counts_done_this_week)
            if missed > 0:
                self.cycle_counts_overdue += missed

                if self.cycle_counts_overdue <= CYCLE_COUNTS_PER_WEEK:
                    # Slipped 1 week
                    reward += CYCLE_COUNT_SLIP_PENALTY_1 * missed
                else:
                    # Slipped 2+ weeks
                    reward += CYCLE_COUNT_SLIP_PENALTY_2 * missed

                self._add_year_reward("cycle_count_slip", reward)

            # Management backlog crossing week threshold
            if self.management_backlog > MANAGEMENT_BACKLOG_WEEK_THRESHOLD:
                penalty = MANAGEMENT_BACKLOG_WEEKLY_PENALTY
                self._add_year_reward("management_backlog_weekly", penalty)
                reward += penalty

        # Reset weekly counters — new week, fresh hustle slate
        self.cycle_counts_done_this_week = 0
        self.weekly_hustle_hours = {i: 0.0 for i in range(NUM_WORKERS)}
        self.hustle_exhausted = {i: False for i in range(NUM_WORKERS)}
        self.current_week += 1

        self.year_reward += reward
        return reward

    def _process_end_of_year(self) -> float:
        """Final year-end scoring."""
        reward = 0.0

        # Check final week's cycle counts
        missed = max(0, CYCLE_COUNTS_PER_WEEK - self.cycle_counts_done_this_week)
        self.cycle_counts_overdue += missed

        # Monthly cycle count compliance
        # Each month should have ~8 cycle counts (2/week × 4 weeks)
        total_expected = self.current_week * CYCLE_COUNTS_PER_WEEK
        total_done = total_expected - self.cycle_counts_overdue
        if self.cycle_counts_overdue > 4:
            reward += CYCLE_COUNT_MONTH_MISS_PENALTY
            self._add_year_reward("cycle_count_year_miss", CYCLE_COUNT_MONTH_MISS_PENALTY)

        self.year_reward += reward
        return reward

    def complete_cycle_count(self):
        """Called when the agent assigns a cycle count task."""
        self.cycle_counts_done_this_week += 1

    def _augment_state(self, daily_state: np.ndarray) -> np.ndarray:
        """Append year-level features to the daily state vector."""
        year_features = np.array([
            self.current_day_idx / max(1, self.total_work_days),  # day of year progress
            self.current_week / 52.0,  # week of year
            min(self.management_backlog / 20.0, 1.0),  # mgmt backlog (capped at 1.0)
            self.cycle_counts_done_this_week / max(1, CYCLE_COUNTS_PER_WEEK),  # this week's progress
            min(self.cycle_counts_overdue / 10.0, 1.0),  # overdue (capped)
            self.restock_level,  # carried-over restock level
            1.0 if self.current_day_idx < self.total_work_days and self.work_days[self.current_day_idx]["day_of_week"] == 0 else 0.0,  # is Monday
        ], dtype=np.float32)

        return np.concatenate([daily_state, year_features])

    def _get_year_state(self) -> np.ndarray:
        """Get state when year is done (final state)."""
        # Use the last daily state and augment it
        daily_state = self.day_env._get_state()
        return self._augment_state(daily_state)

    def _add_year_reward(self, key: str, value: float):
        self.year_reward_breakdown[key] = self.year_reward_breakdown.get(key, 0.0) + value

    def _get_year_summary(self) -> dict:
        """Comprehensive year-end summary."""
        grade_counts = {g: self.daily_grades.count(g) for g in ["A", "B", "C", "D", "F"]}
        total_days = len(self.daily_grades)

        # Year grade: based on % of A+B days
        win_pct = (grade_counts["A"] + grade_counts["B"]) / max(1, total_days)
        if win_pct >= 0.90 and grade_counts["F"] == 0:
            year_grade = "A"
        elif win_pct >= 0.80:
            year_grade = "B"
        elif win_pct >= 0.65:
            year_grade = "C"
        elif win_pct >= 0.50:
            year_grade = "D"
        else:
            year_grade = "F"

        return {
            "year": self.year,
            "total_work_days": total_days,
            "grade_distribution": grade_counts,
            "year_grade": year_grade,
            "total_reward": round(self.year_reward, 2),
            "year_reward_breakdown": {k: round(v, 2) for k, v in self.year_reward_breakdown.items()},
            "management_backlog_final": round(self.management_backlog, 2),
            "cycle_counts_overdue_final": self.cycle_counts_overdue,
            "daily_summaries": self.daily_summaries,
        }

    @property
    def state_size(self) -> int:
        """Total state size: daily state + year features."""
        from volt_sim.config import TOTAL_STATE_SIZE
        return TOTAL_STATE_SIZE + self.YEAR_STATE_SIZE
