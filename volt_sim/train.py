"""
Training loop for the Dolly warehouse RL simulation.

Usage:
  python volt_sim/train.py                    # fresh start
  python volt_sim/train.py --resume           # resume from latest checkpoint
  python volt_sim/train.py --resume ppo_ep900.pt  # resume from specific checkpoint
"""
import sys
import time
import os
import glob
import argparse
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volt_sim.config import TRAINING, PPO as PPO_CFG
from volt_sim.env.year_env import YearEnv
from volt_sim.agent.ppo import PPOAgent
from volt_sim.agent.actions import decode_actions, get_valid_action_mask, NUM_ACTION_HEADS, ACTION_HEAD_SIZE
from volt_sim.agent.state import RunningStats
from volt_sim.sim_logging.episode_logger import EpisodeLogger

CHECKPOINT_DIR = "volt_sim/data/checkpoints"


def find_latest_checkpoint() -> tuple[str, int] | None:
    """Find the most recently written checkpoint and extract its episode number.

    Sorts by file modification time, not episode number — this correctly handles
    cases where a fresh training run (ep 10) coexists with stale checkpoints from
    a previous run (ep 500+) that used a different architecture.
    """
    pattern = os.path.join(CHECKPOINT_DIR, "ppo_ep*.pt")
    files = glob.glob(pattern)
    if not files:
        return None

    def ep_num(path):
        name = os.path.basename(path)
        try:
            return int(name.replace("ppo_ep", "").replace(".pt", ""))
        except ValueError:
            return 0

    # Sort newest-written first (modification time descending)
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    return latest, ep_num(latest)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="Resume from checkpoint. 'latest' or a filename like ppo_ep900.pt")
    args = parser.parse_args()

    env = YearEnv()
    state_size = env.state_size
    agent = PPOAgent(state_size=state_size)
    logger = EpisodeLogger()
    state_stats = RunningStats(state_size)

    total_episodes = TRAINING["total_episodes"]
    log_interval = TRAINING["log_interval"]
    save_interval = TRAINING["save_interval"]
    start_episode = 1

    # Resume from checkpoint
    if args.resume:
        if args.resume == "latest":
            # Walk all checkpoints newest-first (by mtime) until one loads cleanly.
            # Stale/incompatible files are skipped, not deleted — they may belong to
            # a different run and are harmless once the architecture version tag rejects them.
            pattern = os.path.join(CHECKPOINT_DIR, "ppo_ep*.pt")
            candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if not candidates:
                print("No checkpoints found, starting fresh.")
            else:
                loaded_any = False
                for path in candidates:
                    def ep_num_from_path(p):
                        name = os.path.basename(p)
                        try:
                            return int(name.replace("ppo_ep", "").replace(".pt", ""))
                        except ValueError:
                            return 0
                    ep = ep_num_from_path(path)
                    loaded = agent.load(path, state_stats)
                    if loaded:
                        start_episode = ep + 1
                        print(f"Resumed from {os.path.basename(path)} (episode {ep})")
                        loaded_any = True
                        break
                    else:
                        print(f"Skipping {os.path.basename(path)} (incompatible architecture).")
                if not loaded_any:
                    print("No compatible checkpoints found, starting fresh.")
        else:
            path = os.path.join(CHECKPOINT_DIR, args.resume)
            if os.path.exists(path):
                loaded = agent.load(path, state_stats)
                if loaded:
                    try:
                        start_episode = int(args.resume.replace("ppo_ep", "").replace(".pt", "")) + 1
                    except ValueError:
                        start_episode = 1
                    print(f"Resumed from {args.resume} (starting at episode {start_episode})")
                else:
                    print("Checkpoint incompatible, starting fresh.")
                    start_episode = 1
            else:
                print(f"Checkpoint not found: {path}")
                return

    print(f"Training episodes {start_episode} to {total_episodes}...")
    print(f"State size: {state_size} (daily={state_size - YearEnv.YEAR_STATE_SIZE} + year={YearEnv.YEAR_STATE_SIZE})")
    print(f"Action heads: {NUM_ACTION_HEADS} workers × {ACTION_HEAD_SIZE} tasks")
    print(f"Each episode = 1 full year (~260 work days)")
    print()

    start_time = time.time()
    # Initialize day_count from start_episode so resume preserves true cumulative day numbers.
    # 261 work days/year — this keeps episode numbers in the log consistent across restarts.
    day_count = (start_episode - 1) * 261

    for episode_num in range(start_episode, total_episodes + 1):
        state = env.reset()
        state_stats.update(state)
        norm_state = state_stats.normalize(state)

        # Reset LSTM memory — each year is a fresh episode
        agent.reset_hidden()
        # Snapshot the entry hidden state before collecting the first day's rollout
        agent.buffer.set_entry_hidden(agent.hidden)

        episode_reward = 0.0
        done = False
        steps = 0
        ep_day_count = 0

        while not done:
            # Get valid action mask
            mask = get_valid_action_mask(env)

            # Select action — one task per worker (advances LSTM hidden state by 1 step)
            actions, log_prob, value = agent.select_action(norm_state, mask)

            # Decode into (worker_id, task_id, hustle) tuples
            reassignments = decode_actions(actions)

            # Step environment
            next_state, reward, done, info = env.step(reassignments)

            # Store transition
            agent.store_transition(
                norm_state, actions, log_prob, reward, value, done, mask
            )

            # Update state
            state_stats.update(next_state)
            norm_state = state_stats.normalize(next_state)
            episode_reward += reward
            steps += 1

            # End of day — learn immediately, don't wait for year-end
            if info.get("new_day") or done:
                ep_day_count += 1
                day_count += 1

                # PPO update — sequential replay preserves LSTM temporal order
                metrics = agent.update()

                # Snapshot hidden state for the NEXT day's rollout collection.
                # The hidden state carries across the day boundary — Dolly remembers
                # what happened earlier this week when making tomorrow's decisions.
                agent.buffer.set_entry_hidden(agent.hidden)

                # Log daily summary
                prev_summary = env.daily_summaries[-1] if env.daily_summaries else None
                if prev_summary:
                    logger.log_episode(prev_summary, day_count, write=False)

                    if day_count % 10 == 0:
                        f = prev_summary["footer"]
                        h = prev_summary["header"]
                        stats = logger.get_training_stats()
                        print(
                            f"  Day {ep_day_count:3d}/{env.total_work_days} | "
                            f"{h['day_of_week']:3s} {h['month']:9s} | "
                            f"Orders: {f['orders_shipped']}/{f['orders_total']} | "
                            f"Grade: {f['grade']} | "
                            f"Restock: {env.restock_level:.0%} | "
                            f"Backlog: {env.management_backlog:.1f}h | "
                            f"WinR: {stats['win_rate_last_100']:.0%}"
                        )

        # Year complete — log it
        year_summary = info.get("year_summary", {})
        grade_dist = year_summary.get("grade_distribution", {})
        year_grade = year_summary.get("year_grade", "?")

        elapsed = time.time() - start_time
        print(
            f"\nYear {episode_num} complete | "
            f"Grade: {year_grade} | "
            f"A:{grade_dist.get('A',0)} B:{grade_dist.get('B',0)} "
            f"C:{grade_dist.get('C',0)} D:{grade_dist.get('D',0)} F:{grade_dist.get('F',0)} | "
            f"Days: {ep_day_count} | Steps: {steps} | "
            f"Reward: {episode_reward:.0f} | "
            f"Backlog: {env.management_backlog:.1f}h | "
            f"Overdue CCs: {env.cycle_counts_overdue} | "
            f"{elapsed:.0f}s\n"
        )

        # Flush log at year end — write full rolling log and year snapshot
        logger._write_log()
        logger.write_year_snapshot(ep_day_count)

        # Save checkpoint
        if episode_num % save_interval == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            agent.save(f"{CHECKPOINT_DIR}/ppo_ep{episode_num}.pt", state_stats)

    # Final save
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    agent.save(f"{CHECKPOINT_DIR}/ppo_final.pt", state_stats)
    print(f"\nTraining complete. {total_episodes} year-episodes in {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    train()
