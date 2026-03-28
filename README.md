# Jack

PPO + LSTM agent for multi-week warehouse scheduling. Trained on a full simulated work year — 261 days, 7 workers, 15-minute decision intervals.

Built on top of [Dolly](https://github.com/jarmstrong158/Dolly). Dolly handles single-day optimization. Jack adds weekly scope: hustle pacing, worker exhaustion, seasonal demand, and consequence chains that carry across days.

---

## Simulation

7 workers. 6 tasks. Every 15 minutes, Jack assigns each worker a task and decides whether to push them into hustle mode.

**Tasks:** pick, pack, restock, side project, management, idle

Order volume swings from ~60/day in January to ~500/day at the May/June peak. Workers come in with varying debuffs — bad sleep, illness, injuries, call-offs. Some have hard constraints: Marcus carries management duty every day, Blake is pack-only during EOE flares, pickers rotate daily.

---

## Results

~2,450 training days (~9.4 simulated years).

![Training Header](docs/screenshots/01_training_header.png)

![Reward and Win Rate Trend](docs/screenshots/02_reward_trend.png)

---

## Year Overview

![Year Overview](docs/screenshots/03_year_overview.png)

| Stat | Value |
|------|-------|
| Days logged | 261 |
| Orders shipped | 68,189 |
| Completion rate | 98.2% |
| A-grade days | 151 (58%) |
| F-grade days | 25 |
| OT days | 99 (38%) |

---

## Season Breakdown

![Season Performance](docs/screenshots/04_season_performance.png)

| Season | Win Rate |
|--------|----------|
| Winter | 100% |
| Spring | 92% |
| Summer | 69% |
| Fall | 95% |

Primary F-grade driver: bad sleep stacking with call-offs on peak-volume days.

---

## Episode Detail

![Episode Detail](docs/screenshots/05_episode_detail.png)

Blake called off. 86/86 orders. 100% restock. No OT.

---

## Order Flow

![Order Flow](docs/screenshots/06_order_flow.png)


---

## Worker Utilization

![Worker Utilization](docs/screenshots/07_worker_utilization.png)

---

## Architecture

```
state (155-dim) → Linear → ReLU → LSTM(256) → 7 policy heads + 1 value head
```

- **State:** 19 features per worker × 7 workers + 7 year-level features
- **Action space:** 7 workers × 12 actions (6 tasks × hustle on/off)
- **Training:** PPO with TBPTT, chunk size 16. Updates at end of each day.
- **Hidden state:** persists across all ~13,000 steps in a year, reset at year start

Action masking enforces hard constraints at every step — absent workers, shift end, hustle exhaustion, pack-only restrictions.

---

## Hustle System

Per-worker daily hustle caps. Exceed 2× the cap in a week: -15% OPH, hustle locked for the remainder. Management and idle cannot be hustled.

| Worker | Daily Cap | Exhaustion Threshold |
|--------|-----------|----------------------|
| Marcus | 9.5h | 19h |
| Nolan | 7.0h | 14h |
| Felix | 6.0h | 12h |
| Blake | 8.0h | 16h |
| Reid | 8.0h | 16h |
| Trent | 3.0h | 6h |
| Omar | 7.0h | 14h |

---

## Worker Roster

| Name | Base OPH | Shift | Role |
|------|----------|-------|------|
| Marcus | 17.00 | 9.75h | Manager |
| Nolan | 15.35 | 8.5h | Assistant Manager |
| Felix | 16.23 | 8.5h | Warehouse |
| Blake | 18.30 | 8.5h | Warehouse |
| Reid | 18.94 | 8.5h | Warehouse |
| Trent | 15.28 | 8.5h | Warehouse |
| Omar | 14.88 | 8.5h | Warehouse |

Picker rotation: Mon=Reid, Tue=Blake, Wed=Felix, Thu=Omar, Fri=Trent

---

## Grading

| Grade | Criteria |
|-------|----------|
| A | All orders shipped, restock complete, management met, no OT |
| B | All orders shipped with OT, or minor restock/management miss |
| C | 95%+ completion |
| D | 85–94% completion |
| F | Under 85% or hard stop with open orders |

---

## Dashboard

```bash
python -m http.server 8080
```

`http://localhost:8080/volt_sim/dashboard/dashboard.html`

Load `volt_sim/data/episode_log.json`. Updates at year end.

---

## Usage

```bash
# Train from scratch
python volt_sim/train.py

# Resume
python volt_sim/train.py --resume
```

**Requirements:** Python 3.10+, PyTorch, NumPy

```bash
pip install torch numpy
```

---

## Structure

```
volt_sim/
  agent/
    ppo.py                  # PPO + LSTM actor-critic, TBPTT
    actions.py              # Action encoding, masking
    state.py                # Running stats normalization
  env/
    warehouse_env.py        # Core simulation
    year_env.py             # Year wrapper, weekly hustle tracking
    workers.py              # Worker state, OPH, debuffs, hustle
    episode_generator.py    # Daily scenario generation
  sim_logging/
    episode_logger.py
  dashboard/
    dashboard.html
  config.py
  train.py
```
