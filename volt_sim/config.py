"""
Volt Warehouse RL Simulation — Central Configuration
All tunable parameters live here. Nothing hardcoded elsewhere.
"""

# ─── Worker Roster ───────────────────────────────────────────────────────────
WORKERS = [
    {"id": 0, "name": "Marcus", "oph": 17.0,  "shift_hours": 9.75, "role": "manager"},
    {"id": 1, "name": "Nolan",     "oph": 15.35, "shift_hours": 8.5,  "role": "assistant_manager"},
    {"id": 2, "name": "Felix",    "oph": 16.23, "shift_hours": 8.5,  "role": "warehouse"},
    {"id": 3, "name": "Blake",   "oph": 18.30, "shift_hours": 8.5,  "role": "warehouse"},
    {"id": 4, "name": "Reid",      "oph": 18.94, "shift_hours": 8.5,  "role": "warehouse"},
    {"id": 5, "name": "Trent",      "oph": 15.28, "shift_hours": 8.5,  "role": "warehouse"},
    {"id": 6, "name": "Omar",    "oph": 14.88, "shift_hours": 8.5,  "role": "warehouse"},
]

NUM_WORKERS = len(WORKERS)

# ─── Tasks ───────────────────────────────────────────────────────────────────
TASKS = ["pick", "pack", "restock", "side_project", "management", "idle", "cycle_count"]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
NUM_TASKS = len(TASKS)

# ─── Morning Pick Round ──────────────────────────────────────────────────────
# Everyone picks 1-2 carts at day start before assignments kick in.
# 12-14 carts total across the team, 1-6 orders per cart.
# Fixed mechanic, not an agent decision.
MORNING_PICK_CARTS_MIN = 1    # min carts per worker
MORNING_PICK_CARTS_MAX = 2    # max carts per worker
MORNING_PICK_PER_CART_MIN = 1  # min orders per cart
MORNING_PICK_PER_CART_MAX = 6  # max orders per cart

# ─── Hustle Mode ─────────────────────────────────────────────────────────────
# Agent-callable per worker. +12% OPH while active.
# Each worker has a daily cap (hours) and a weekly threshold (2× daily cap).
# Exceeding the weekly threshold triggers exhaustion for the rest of the week:
# -15% OPH on top of all other modifiers, hustle blocked until Monday.
HUSTLE_MODE_BONUS = 1.12           # +12% OPH while hustling
HUSTLE_EXHAUSTION_MULTIPLIER = 0.85  # -15% OPH when exhausted from overuse

# Per-worker daily hustle caps (hours)
HUSTLE_DAILY_CAPS = {
    0: 9.5,  # Marcus — all shift hours
    1: 7.0,  # Nolan     — all except first hour (easing in)
    2: 6.0,  # Felix    — moderate endurance
    3: 8.0,  # Blake   — all shift hours
    4: 8.0,  # Reid      — all shift hours
    5: 3.0,  # Trent      — low endurance
    6: 7.0,  # Omar    — most of shift
}
# Weekly threshold = 2× daily cap. Hitting it triggers exhaustion.
HUSTLE_WEEKLY_THRESHOLDS = {w_id: cap * 2 for w_id, cap in HUSTLE_DAILY_CAPS.items()}

# Tasks that cannot be hustled (hustle implies physical output — not admin or standing still)
HUSTLE_BLOCKED_TASKS = {"management", "idle", "cycle_count"}

# ─── Task OPH Multipliers ────────────────────────────────────────────────────
# Base OPH represents packing speed. Picking is inherently faster.
# Main picker (designated for the day) gets 2.5x, supplemental pickers get 2.25x.
PICK_MULTIPLIER_MAIN = 2.5       # today's designated picker
PICK_MULTIPLIER_SUPPLEMENT = 2.25  # everyone else picking
TASK_OPH_MULTIPLIER = {
    "pack": 1.0,          # base OPH = packing rate
    "restock": 1.0,       # restock measured in hours, not orders
    "side_project": 1.0,  # side projects measured in hours
    "management": 0.0,    # management is time-based, not output-based
    "idle": 0.0,
    "cycle_count": 0.0,   # audit work — time-based, not output-based
}
# Note: "pick" is NOT in TASK_OPH_MULTIPLIER — it's resolved dynamically
# based on whether the worker is the designated picker or supplementing.

# ─── Picker Schedule (day_of_week 0=Monday) ─────────────────────────────────
PICKER_SCHEDULE = {
    0: 4,  # Monday: Reid
    1: 3,  # Tuesday: Blake
    2: 2,  # Wednesday: Felix
    3: 6,  # Thursday: Omar
    4: 5,  # Friday: Trent
}

# ─── Seasonal Order Volume ──────────────────────────────────────────────────
ORDER_VOLUME_RANGES = {
    "January":   (60, 100),
    "February":  (60, 100),
    "March":     (150, 250),
    "April":     (300, 450),
    "May":       (350, 500),
    "June":      (350, 500),
    "July":      (280, 400),
    "August":    (280, 400),
    "September": (250, 350),
    "October":   (200, 300),
    "November":  (150, 220),
    "December":  (60, 100),
}

MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring", 5: "spring",
    6: "summer",  7: "summer", 8: "summer",
    9: "fall",   10: "fall",  11: "fall",
}

# ─── Weekly Volume Curve ─────────────────────────────────────────────────────
# Daily order volume as a percentage band of the seasonal range.
# Monday is heaviest, Friday can be dead slow.
# Each day rolls a random value within its band.
# day_of_week: (low_pct, high_pct) — applied to the seasonal spread above the minimum.
WEEKLY_VOLUME_CURVE = {
    0: (0.86, 1.00),  # Monday
    1: (0.71, 0.85),  # Tuesday
    2: (0.56, 0.70),  # Wednesday
    3: (0.41, 0.55),  # Thursday
    4: (0.00, 0.40),  # Friday — can be dead slow
}

# ─── Monthly Episode ─────────────────────────────────────────────────────────
WORK_DAYS_PER_MONTH = 20  # 4 weeks × 5 days

# Management backlog
MANAGEMENT_MIN_DAILY_HOURS = 1.5       # minimum mgmt hours per day
MANAGEMENT_BACKLOG_WEEK_THRESHOLD = 10.0  # >10hrs backlog entering new week = penalty
MANAGEMENT_BACKLOG_WEEKLY_PENALTY = -50.0  # per week over threshold

# Cycle counts
CYCLE_COUNTS_PER_WEEK = 2        # 2 required per week
CYCLE_COUNT_HOURS = 1.5          # 1.5 hours each (split between Marcus + Nolan)
CYCLE_COUNT_SLIP_PENALTY_1 = -30.0    # slipped 1 week
CYCLE_COUNT_SLIP_PENALTY_2 = -75.0    # slipped 2+ weeks
CYCLE_COUNT_MONTH_MISS_PENALTY = -200.0  # didn't complete all for the month
CYCLE_COUNT_WEEKLY_HOURS_REQUIRED = 3.0  # 2 × 1.5h = full week complete
CYCLE_COUNT_ELIGIBLE_WORKERS = {0, 1}    # Marcus and Nolan only

SEASONS = ["winter", "spring", "summer", "fall"]

# ─── Order Arrival Curve ────────────────────────────────────────────────────
# (bucket_label, fraction_low, fraction_high)
ORDER_ARRIVAL_BUCKETS = [
    ("day_start",    0.40, 0.50),  # 9:00 AM instant
    ("morning",      0.25, 0.30),  # 9:00 AM - 2:00 PM
    ("afternoon",    0.10, 0.15),  # 2:00 PM - 4:15 PM
    ("late_surge",   0.15, 0.20),  # 4:15 PM - 5:00 PM (order cutoff)
]

# High-volume days: heavier front-load, hard taper after 2 PM.
# Orders still arrive all day but the bulk hits by noon — EOD is cleanup, not new work.
ORDER_ARRIVAL_BUCKETS_HIGH_VOLUME = [
    ("day_start",    0.45, 0.55),  # 9:00 AM instant burst
    ("morning",      0.30, 0.38),  # 9:00 AM - 2:00 PM — heavy flow
    ("afternoon",    0.08, 0.12),  # 2:00 PM - 4:15 PM — slowing
    ("late_trickle", 0.03, 0.07),  # 4:15 PM - 5:00 PM — minimal, cleanup window
]

HIGH_VOLUME_PERCENTILE = 0.75  # top 25% of range = high volume day

# ─── Shift Timing ───────────────────────────────────────────────────────────
DAY_START_HOUR = 9.0          # 9:00 AM (Marcus arrives at 7:45 but pre-sim hours handled via MARCUS_PRE_SIM_MANAGEMENT)
LUNCH_HOUR = 13.0             # 1:00 PM
LUNCH_DURATION = 0.5          # 30 minutes
EOD_HOUR = 17.5               # 5:30 PM — everyone leaves, including Marcus
ORDER_CUTOFF_HOUR = 17.0      # 5:00 PM — orders after this don't need completing today
ORDER_ARRIVAL_END_HOUR = 16.83  # Last sim step before 5 PM — no orders scheduled after this
STEP_DURATION = 1/6           # 10-minute intervals (0.1667 hours)

# Morning arrival ends, afternoon starts, late surge starts
MORNING_END_HOUR = 14.0       # 2:00 PM
AFTERNOON_END_HOUR = 16.25    # 4:15 PM

# ─── Restock ─────────────────────────────────────────────────────────────────
RESTOCK_BASE_HOURS = 2.5
RESTOCK_VOLUME_COEFF = 0.008    # max 7.0h at 500 orders
RESTOCK_NOISE_RANGE = (-0.5, 0.5)

# Restock level system — restock is a resource that depletes as orders are picked.
# Starts at 100%. Each picked order drains it. When it hits 0%, picking speed
# drops drastically (shelves empty). Restocking refills the level.
RESTOCK_STARTING_LEVEL = 1.0            # 100%
RESTOCK_DRAIN_PER_ORDER = None          # calculated per-episode: 1.0 / total_orders * drain_factor
RESTOCK_DRAIN_FACTOR = 2.0              # level drains to 0 after ~50% of orders without restocking
RESTOCK_PICK_PENALTY_THRESHOLD = 0.2    # below 20%, picking OPH drops
RESTOCK_PICK_PENALTY_MULTIPLIER = 0.05  # at 0% restock, picking is 5% speed (nothing on shelves)
RESTOCK_REFILL_PER_HOUR = None          # calculated per-episode based on restock_hours

# ─── Side Projects ──────────────────────────────────────────────────────────
DELIBERATE_PROJECT_CHANCE = 0.25
DELIBERATE_PROJECT_SIZE_RANGE = (2.0, 6.0)  # worker-hours
FILLER_PROJECT_SIZE = float("inf")           # never depletes
FILLER_COMPLETION_THRESHOLD = 4.0            # hours of filler work to count as "complete"

# ─── Fatigue Thresholds ─────────────────────────────────────────────────────
PACK_FATIGUE_THRESHOLD = 110    # orders packed before fatigue kicks in
PICK_FATIGUE_THRESHOLD = 230    # orders picked before fatigue kicks in
FATIGUE_OPH_PENALTY = 0.85     # 15% drop

# Trent's soreness
TRENT_SORENESS_HOUR_THRESHOLD = 4.0   # hours of non-side-project work
TRENT_SORENESS_OPH_PENALTY = 0.50     # 50% OPH drop

# ─── Marcus Manager Constraints ───────────────────────────────────────────
MARCUS_MANAGEMENT_HOURS_REQUIRED = 4.0
MARCUS_PRE_SIM_MANAGEMENT = 1.25         # 7:45 to 9:00 = management before sim starts
MANAGEMENT_FALLBACK_WORKER_ID = 2          # Felix steps in when Marcus + Nolan are both absent

# Peak season early start — Marcus arrives 30 min earlier in spring/summer for cycle counts
MARCUS_PEAK_SEASONS = {"spring", "summer"}
MARCUS_PEAK_EARLY_CYCLE_COUNT = 0.5     # 0.5h pre-sim cycle count credit

# ─── Debuff: Sleep Category ─────────────────────────────────────────────────
SLEEP_DEBUFFS = [
    {"name": "well_rested", "multiplier": 1.05, "probability": 0.10},
    {"name": "normal",      "multiplier": 1.00, "probability": 0.75},
    {"name": "bad_sleep",   "multiplier": 0.95, "probability": 0.15},
]

# ─── Debuff: Health Category ────────────────────────────────────────────────
HEALTH_DEBUFFS = [
    {"name": "locked_in",     "multiplier": 1.20, "probability": 0.05},
    {"name": "normal",        "multiplier": 1.00, "probability": 0.69},
    {"name": "bad_headspace", "multiplier": None,  "probability": 0.08},  # per-worker
    {"name": "sick_mild",     "multiplier": 0.85, "probability": 0.05},
    {"name": "very_sick",     "multiplier": 0.60, "probability": 0.01},
    {"name": "injured",       "multiplier": 0.50, "probability": 0.02},
    {"name": "buffer",        "multiplier": 1.00, "probability": 0.10},
]

# ─── Bad Headspace Effects (per worker name) ────────────────────────────────
# Format: {task: multiplier} — tasks not listed get the default multiplier
BAD_HEADSPACE_EFFECTS = {
    "Marcus": {"default": 0.95},                                          # -5% all
    "Nolan":     {"side_project": 1.10, "default": 0.90},                    # +10% side, -10% rest
    "Felix":    {"pack": 1.10, "default": 0.90},                            # +10% pack, -10% rest
    "Reid":      {"default": 1.00},                                          # immune
    "Omar":    {"pack": 1.10, "default": 0.90},
    "Blake":   {"pack": 1.10, "default": 0.90},
    "Trent":      {"pack": 1.10, "default": 0.90},
}

# ─── Daily Call-Off System ──────────────────────────────────────────────────
# Any worker can call off on any day. Max 2 per day — 3rd is denied.
# Rolled at the start of each day before debuffs.
CALL_OFF_PROBABILITIES = {
    "Marcus": 0.01,    # 1% — management, very rare
    "Nolan":     0.02,    # 2% — assistant manager, rare
    "Felix":    0.035,   # 3.5%
    "Blake":   0.035,   # 3.5%
    "Reid":      0.035,   # 3.5%
    "Trent":      0.035,   # 3.5%
    "Omar":    0.035,   # 3.5%
}
MAX_CALL_OFFS_PER_DAY = 2
MAX_CALL_OFFS_HIGH_VOLUME = 1      # Only 1 call-off allowed on peak days
CALL_OFF_PROBABILITY_HIGH_VOLUME = 0.02  # 2% across the board on high-volume days

# ─── Individual Debuffs ─────────────────────────────────────────────────────
INDIVIDUAL_DEBUFFS = {
    "Marcus": {
        "name": "family_needs",
        "probability": 0.25,
        "cooldown_days": 5,
        "effect": "lose_hours",
        "hours_lost_range": (1.0, 2.0),
    },
    "Nolan": {
        "name": "family_needs",
        "probability": 0.25,
        "cooldown_days": 5,
        "effect": "lose_hours",
        "hours_lost_range": (1.0, 2.0),
    },
    "Felix": {
        "name": "stomach_issues",
        "probability": 0.30,
        "cooldown_days": 4,
        "effect": "oph_penalty",
        "oph_multiplier": 0.85,
    },
    "Blake": {
        "name": "eoe_muscle_flare",
        "probability": None,  # season-weighted, see below
        "cooldown_days": 0,   # no cooldown
        "effect": "pack_only",
    },
    "Reid": None,  # Reid's NCNS replaced by general call-off system
    "Trent": {
        "name": "soreness",
        "probability": 1.0 / 6.0,  # 16.7%
        "cooldown_days": 0,
        "effect": "soreness",
    },
    "Omar": {
        "name": "family_needs",
        "probability": 0.15,
        "cooldown_days": 7,
        "effect": "lose_hours",
        "hours_lost_range": (1.0, 2.0),
    },
}

# Blake flare probabilities by season
BLAKE_FLARE_PROBABILITIES = {
    "winter": 0.05,
    "spring": 0.20,
    "summer": 0.25,
    "fall":   0.12,
}

# ─── Reward Signals ─────────────────────────────────────────────────────────
REWARDS = {
    # Orders
    "per_order_shipped":              1.0,
    "all_orders_complete_bonus":     50.0,
    "per_order_incomplete":         -10.0,
    "per_ot_hour":                   -0.5,
    "ot_incomplete_flat":           -25.0,
    "ot_per_order_incomplete":      -10.0,
    "marcus_per_order":            -0.3,
    "nolan_per_order":                -0.15,

    # Restock — critical: empty shelves halt picking
    "per_restock_completed":          1.0,
    "all_restock_bonus":             25.0,
    "per_restock_bleed":             -2.0,
    "restock_pick_interruption":    -10.0,
    "warehouse_worker_restock":      -0.2,  # penalty for using non-manager on restock
    "restock_level_low":             -2.0,  # per step when restock level below 20%
    "restock_level_empty":           -5.0,  # per step when restock level at 0%

    # Side projects
    "per_filler_unit":                0.1,
    "filler_completion_bonus":        5.0,
    "per_deliberate_unit":            0.1,
    "deliberate_completion_bonus":    8.0,
    "side_project_during_crunch":    -2.0,

    # Worker management — idle penalty must be strong enough to discourage
    # leaving Marcus/Nolan idle after management quota is met
    "per_productive_hour":            0.3,
    "per_management_hour":            0.5,    # per-step signal so bot sees value in management
    "per_idle_hour":                 -0.5,
    "packers_starved":               -1.0,   # per packer with nothing to pack while queue has orders
    "picked_backlog":                -2.0,   # per 10 orders sitting picked but not packed
    "management_duty_met":           30.0,
    "management_duty_missed":       -50.0,
    "blake_prohibited_task":        -5.0,

    # Cycle counts
    "per_cycle_count_hour":          1.0,    # per hour spent on cycle count
    "cycle_count_week_complete":    40.0,    # bonus when weekly threshold met
    "cycle_count_week_missed":     -75.0,    # penalty if not completed by week end
}

# ─── Grading ───────────────────────────────────────────────────────────────
# Outcome-based demerit system (implemented in warehouse_env.py):
# F = any orders incomplete. Always.
# A = all orders + restock done + management duty met + no OT
# Each breach (restock, management, OT) drops one letter grade.

# ─── PPO Hyperparameters ────────────────────────────────────────────────────
PPO = {
    "lr": 3e-4,              # slightly higher LR works well with LSTM
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coeff": 0.02,
    "value_loss_coeff": 0.5,
    "max_grad_norm": 0.5,
    "epochs_per_update": 4,
    "hidden_size": 256,      # LSTM hidden state size — larger = better temporal memory
    "tbptt_chunk_size": 16,  # detach hidden every N steps — must be < day length (~50) to prevent NaN
}

# ─── Training ───────────────────────────────────────────────────────────────
TRAINING = {
    "total_episodes": 1000,      # each episode = 1 full year (~260 days, ~13000 steps at 10min intervals)
    "log_interval": 1,           # log every year (they're long)
    "save_interval": 10,         # checkpoint every 10 years
    "rolling_window": 522,       # 2 full years of daily logs (~261 work days/year)
}

# ─── State Vector Dimensions ────────────────────────────────────────────────
# Per worker: 7 one-hot task + 13 scalars = 20
# Scalars: oph, hours_worked, hours_remaining, generic_debuff, individual_debuff,
#          fatigue, is_picker, is_pack_only, soreness_progress, management_hours,
#          hustle_today_ratio, hustle_weekly_ratio, is_hustle_exhausted
WORKER_STATE_SIZE = NUM_TASKS + 13  # 20
ENV_STATE_SIZE = 15     # 1 hour + 1 orders_remaining + 1 orders_completed +
                        # 1 picked_not_audited + 1 restock_remaining +
                        # 1 side_project_progress + 4 season_onehot +
                        # 1 is_high_volume + 1 total_mgmt_hours + 1 picker_needs_replacement +
                        # 1 restock_level
TOTAL_STATE_SIZE = NUM_WORKERS * WORKER_STATE_SIZE + ENV_STATE_SIZE  # 7*20+15 = 155

# ─── OT ─────────────────────────────────────────────────────────────────────
# Everyone can stay until 6:30 PM (1 hour past 5:30 EOD).
# 7 workers × 1 hour = up to 7 worker-hours of OT available.
OT_WALL_CLOCK_MAX = 1.0  # 1 hour of wall clock OT (5:30 → 6:30)
OT_HARD_STOP = 18.5      # 6:30 PM — absolute latest anyone works
