"""
Action space encoding/decoding for the warehouse environment.

Each step the agent assigns a task to EVERY worker, plus a hustle flag.
7 workers × 12 actions per head = 7 independent heads, each outputting 0-11.

Encoding:
  0-5  → task without hustle  (pick, pack, restock, side_project, management, idle)
  6-11 → task with hustle     (same order, hustle blocked for management/idle)
"""
from volt_sim.config import (
    NUM_WORKERS, NUM_TASKS, TASKS, TASK_TO_IDX,
    MARCUS_MANAGEMENT_HOURS_REQUIRED as MGMT_REQUIRED,
    MANAGEMENT_MIN_DAILY_HOURS, MANAGEMENT_FALLBACK_WORKER_ID,
    HUSTLE_BLOCKED_TASKS,
)

# Each head outputs 0-11: first 6 = no hustle, next 6 = hustle
ACTION_HEAD_SIZE = NUM_TASKS * 2  # 12
NUM_ACTION_HEADS = NUM_WORKERS    # 7

IDLE_IDX = TASK_TO_IDX["idle"]
MGMT_IDX = TASK_TO_IDX["management"]

# Marcus (0) and Nolan (1) are primary management — Felix (2) is fallback only
MANAGEMENT_ELIGIBLE = {0, 1}


def decode_actions(action_list: list[int]) -> list[tuple[int, int, bool]]:
    """
    Decode list of per-worker action indices into (worker_id, task_id, hustle) tuples.
    action_list: [action_for_worker_0, ..., action_for_worker_6]
    action 0-5  → task without hustle
    action 6-11 → task with hustle
    """
    assignments = []
    for worker_id, action in enumerate(action_list):
        task_idx = action % NUM_TASKS
        hustle = action >= NUM_TASKS
        assignments.append((worker_id, task_idx, hustle))
    return assignments


def get_valid_action_mask(env) -> list[list[bool]]:
    """
    Returns a per-worker mask: list of NUM_WORKERS lists, each of length ACTION_HEAD_SIZE (12).
    mask[worker_id][action] = True if that worker can take that action.
    First 6 = tasks without hustle, next 6 = tasks with hustle.
    Works with both WarehouseEnv and YearEnv.
    """
    day_env = getattr(env, 'day_env', env)

    mask = []
    for w_id in range(NUM_WORKERS):
        worker = day_env.episode.workers[w_id]

        if worker.is_absent:
            # Absent: only idle (no hustle)
            worker_mask = [False] * ACTION_HEAD_SIZE
            worker_mask[IDLE_IDX] = True
        elif worker.hours_remaining <= 0 and not day_env.is_ot:
            # Shift over, no OT: only idle
            worker_mask = [False] * ACTION_HEAD_SIZE
            worker_mask[IDLE_IDX] = True
        elif worker.is_picker:
            worker_mask = [False] * ACTION_HEAD_SIZE
            worker_mask[TASK_TO_IDX["pick"]] = True  # always can pick
            # Hustle-pick: allowed if worker can hustle
            if worker.can_hustle:
                worker_mask[TASK_TO_IDX["pick"] + NUM_TASKS] = True
            # When queue is empty, other tasks open up
            if day_env.orders_in_queue == 0:
                worker_mask[TASK_TO_IDX["pack"]] = True
                if day_env.restock_level < 1.0:
                    worker_mask[TASK_TO_IDX["restock"]] = True
                worker_mask[TASK_TO_IDX["side_project"]] = True
                # Hustle variants for non-hustle-blocked tasks
                if worker.can_hustle:
                    worker_mask[TASK_TO_IDX["pack"] + NUM_TASKS] = True
                    if day_env.restock_level < 1.0:
                        worker_mask[TASK_TO_IDX["restock"] + NUM_TASKS] = True
                    worker_mask[TASK_TO_IDX["side_project"] + NUM_TASKS] = True
            # Felix fallback management (when both J+E absent, queue empty)
            both_primary_absent = all(
                day_env.episode.workers[i].is_absent for i in MANAGEMENT_ELIGIBLE
            )
            if (both_primary_absent and w_id == MANAGEMENT_FALLBACK_WORKER_ID and
                    day_env.orders_in_queue == 0):
                fallback_mgmt = day_env.episode.workers[MANAGEMENT_FALLBACK_WORKER_ID].management_hours
                if fallback_mgmt < MANAGEMENT_MIN_DAILY_HOURS:
                    worker_mask[TASK_TO_IDX["management"]] = True
                    # Management cannot be hustled
        else:
            worker_mask = [False] * ACTION_HEAD_SIZE
            both_primary_absent = all(
                day_env.episode.workers[i].is_absent for i in MANAGEMENT_ELIGIBLE
            )
            for t_id in range(NUM_TASKS):
                task = TASKS[t_id]
                # ── Determine if no-hustle version is valid ──
                if task == "idle":
                    # Idle only for absent/done workers
                    no_hustle_ok = False
                elif task == "management":
                    if both_primary_absent:
                        fallback_mgmt = day_env.episode.workers[MANAGEMENT_FALLBACK_WORKER_ID].management_hours
                        no_hustle_ok = (w_id == MANAGEMENT_FALLBACK_WORKER_ID and
                                        fallback_mgmt < MANAGEMENT_MIN_DAILY_HOURS)
                    else:
                        total_mgmt = sum(day_env.episode.workers[i].management_hours
                                         for i in MANAGEMENT_ELIGIBLE)
                        backlog = getattr(day_env, '_mgmt_backlog', 0.0)
                        daily_cap = MGMT_REQUIRED + backlog
                        no_hustle_ok = (w_id in MANAGEMENT_ELIGIBLE and
                                        total_mgmt < daily_cap)
                elif worker.is_pack_only and task != "pack":
                    no_hustle_ok = False
                elif task == "restock" and day_env.restock_level >= 1.0:
                    no_hustle_ok = False
                else:
                    no_hustle_ok = True

                worker_mask[t_id] = no_hustle_ok

                # ── Hustle version: same task, +6 offset ──
                hustle_ok = (no_hustle_ok and
                             task not in HUSTLE_BLOCKED_TASKS and
                             worker.can_hustle)
                worker_mask[t_id + NUM_TASKS] = hustle_ok

        mask.append(worker_mask)
    return mask
