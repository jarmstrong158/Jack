# JACK

**Multi-week warehouse RL agent with temporal memory, hustle scheduling, and seasonal optimization.**

Jack builds on [Dolly](https://github.com/jarmstrong158/Dolly) — the single-day warehouse RL agent. Where Dolly learns to run one good day, Jack learns to run a full year. It uses an LSTM-based policy to carry memory across days and weeks, enabling it to learn seasonal patterns, manage worker fatigue across a week, and pace hustle strategically rather than burning workers out.

Start with Dolly to dial in your daily operation. Graduate to Jack once you're ready to think in seasons.

---

## What It Does

Jack simulates a 7-person warehouse across a full work year (~261 days) in 10-minute decision intervals. The LSTM agent makes per-step staffing decisions and carries memory across every shift, day, and week — learning that what happens Monday affects Thursday.

The simulation models real warehouse dynamics:
- **Seasonal demand cycles** — order volume ranges from ~60 orders (January) to ~500 (peak spring/summer)
- **Order arrival curves** — orders trickle in throughout the day across morning, afternoon, and late-day surge windows
- **Worker-specific debuffs** — bad sleep, illness, injuries, family needs, no-call-no-shows, stack per worker
- **Role constraints** — daily picker rotation, manager duty requirements, Blake's EOE pack-only restriction
- **Fatigue mechanics** — OPH degrades after sustained pick/pack thresholds
- **Hustle system** — agent chooses when to push workers harder; daily caps and weekly exhaustion prevent abuse
- **Weekly memory** — LSTM hidden state persists across all steps in a year, never reset mid-season

---

## Proof of Concept

The following data is from a live training run at ~9 years in.

### Training Progress

![Training Header](docs/screenshots/01_training_header.png)

| Metric | Value |
|--------|-------|
| Total days trained | 2,451 |
| Win rate (last 100 days) | **89.8%** |
| Avg reward | 268.48 |
| OT frequency | 44.6% |
| Best reward ever | 598.78 |

![Reward and Win Rate Trend](docs/screenshots/02_reward_trend.png)

The reward trend and win rate charts show the characteristic summer dip — high-volume months push the win rate down to ~62% before Jack recovers through fall and winter. This is expected behavior; summer is the hardest operational period.

### Full Year Overview

![Year Overview](docs/screenshots/03_year_overview.png)

One complete simulated year (261 work days):

| Stat | Value |
|------|-------|
| Orders shipped | 68,189 |
| Completion rate | **98.2%** |
| A-grade days | 151 (58%) |
| F-grade days | 25 |
| OT days | 99 (38%) |

The monthly bar chart shows green (shipped) matching or exceeding gray (total available) across all 12 months. May and June are the volume peak — the hardest stretch of the year.

### Season Performance

![Season Performance](docs/screenshots/04_season_performance.png)

| Season | Win Rate | Episodes |
|--------|----------|----------|
| Winter | **100%** | 66 / 66 |
| Spring | **92%** | 60 / 65 |
| Summer | **69%** | 45 / 65 |
| Fall | **95%** | 62 / 65 |

Summer is where the simulation earns its complexity. Call-offs stack with high volume and hustle exhaustion limits — the agent can't brute-force it by pushing everyone all week. It has to pace. At year 9, summer is still Jack's weakest season and the primary learning target going forward.

The debuff impact chart shows bad sleep as the leading F-grade contributor, followed by call-offs and bad headspace. These are the conditions Jack is still learning to absorb without losing the day.

### Episode Detail — A-Grade Winter Day

![Episode Detail](docs/screenshots/05_episode_detail.png)

Grade A day with Blake called off (6-person crew):

| Field | Value |
|-------|-------|
| Grade | **A** |
| Orders | 86 / 86 |
| OT | 0h |
| Restock | 100% |
| Reward | +202.7 |

Debuffs fired: Blake call-off, Reid well-rested. Jack redistributed Blake's workload across the remaining 6 without touching OT — the LSTM recognized it was a manageable-volume day and didn't over-hustle.

### Order Flow — Clean Execution

![Order Flow](docs/screenshots/06_order_flow.png)

The Order Queue Depth chart shows the hallmark of well-paced execution: the red line (orders in queue) stays near zero all day. Orders are being picked as fast as they arrive. The green completion line is a smooth ramp hitting the 91-order target at 17:20. No last-minute scramble.

The Running Reward shows a clean linear climb to ~97 — every 10-minute interval is net-positive. No stalls, no crashes.

### Worker Utilization

How each worker's hours split across tasks for the day. Blake is 100% idle — he called off. Everyone else is fully productive through close.

![Worker Utilization](docs/screenshots/07_worker_utilization.png)

---

## How It Learns

Jack uses **Proximal Policy Optimization (PPO)** with an **LSTM trunk** instead of a feedforward network. The LSTM hidden state persists across all ~1,300 decision steps in a year and is only reset at the start of each new training episode. This gives Jack genuine temporal memory — it can learn that Tuesday's hustle affects Wednesday's exhaustion threshold, or that a string of high-volume spring days means pacing the crew differently than an isolated spike.

**Architecture:**
```
state (155-dim) → Linear → ReLU → LSTM(256) → 7 policy heads + 1 value head
```

**Training method:** Truncated Backpropagation Through Time (TBPTT) with chunk size 16. Updates happen at the end of each work day, not at year-end — the agent learns from each day's experience immediately while still carrying LSTM memory into the next day.

**Action space:** 7 workers × 12 actions = 84-dimensional per step. Each worker head outputs 0–11: indices 0–5 assign a task without hustle, indices 6–11 assign the same task with hustle enabled. Action masking enforces hard constraints (absent workers, EOD, hustle exhaustion, Blake's pack-only restriction).

**State vector:** 155 dimensions — 7 workers × 19 features each (task one-hot, OPH, hours worked/remaining, debuff status, fatigue, hustle state) + 7 year-level environment features (season, volume signal, restock level, etc.).

The agent learns emergent behaviors across weeks:
- Front-loading hustle early in the week, easing off mid-week before exhaustion triggers
- Keeping the designated daily picker on pick, using Marcus and Nolan as flex floaters
- Shifting to restock when the queue clears instead of letting workers idle
- Recognizing high-volume days from the volume signal and pre-positioning effort

---

## Dashboard

Single self-contained HTML file. Serve from the repo root for live auto-refresh during training.

```bash
# From repo root
python -m http.server 8080
```

Then open: `http://localhost:8080/volt_sim/dashboard/dashboard.html`

The dashboard reads `volt_sim/data/year_snapshot.json`, which is written once per completed training year — it always shows a full year's worth of data, never a partial season view. It auto-refreshes every 10 seconds.

**Panels:**
- Training Progress — rolling win rate, avg reward, OT frequency, best/worst reward ever
- Reward Trend & Win Rate Trend — last 100 days plotted
- Year Overview — full 12-month bar chart, A/F grade counts, completion rate
- Season Performance — win rate by season, debuff impact on F-grade days
- Episode Detail — per-day grade, orders, OT, reward breakdown, debuff list
- Worker Timeline — task-by-task color chart across the shift
- Order Queue Depth & Running Reward — intra-day order flow and reward accumulation

---

## Project Structure

```
volt_sim/
  env/
    warehouse_env.py        # Core simulation — step, reward, state
    year_env.py             # Year-level wrapper — weekly hustle, day sequencing
    workers.py              # Worker state, OPH, debuffs, fatigue, hustle
    episode_generator.py    # Episode setup: date, volume, debuffs, arrivals
    order_arrival.py        # Order arrival curve generation
  agent/
    ppo.py                  # PPO + LSTM actor-critic, TBPTT updates
    state.py                # State normalization (running stats)
    actions.py              # Action space encoding, masking
  sim_logging/
    episode_logger.py       # Rolling log + year snapshot writer
    log_schema.py           # Log entry schemas
  dashboard/
    dashboard.html          # Live training dashboard
  data/
    year_snapshot.json      # Current year data (dashboard reads this)
    episode_log.json        # Full rolling log (last 2 years)
  checkpoints/              # Saved model weights (ppo_ep{N}.pt)
  train.py                  # Training entry point
  config.py                 # All parameters in one place
```

---

## Worker Roster

| Name     | Base OPH | Shift | Role              |
|----------|----------|-------|-------------------|
| Marcus | 17.00    | 9.75h | Manager           |
| Nolan     | 15.35    | 8.5h  | Assistant Manager |
| Felix    | 16.23    | 8.5h  | Warehouse         |
| Blake   | 18.30    | 8.5h  | Warehouse         |
| Reid      | 18.94    | 8.5h  | Warehouse         |
| Trent      | 15.28    | 8.5h  | Warehouse         |
| Omar    | 14.88    | 8.5h  | Warehouse         |

**Picker rotation:** Mon=Reid, Tue=Blake, Wed=Felix, Thu=Omar, Fri=Trent

---

## Hustle System

The agent can flag any worker for hustle on any task at any step. Hustle increases effective OPH but draws from a per-worker daily cap. Hit 2× the daily cap in a week and the worker enters exhaustion for the rest of the week: −15% OPH and hustle blocked.

| Worker   | Daily Cap | Weekly Exhaustion Threshold |
|----------|-----------|-----------------------------|
| Marcus | 9.5h      | 19h |
| Nolan     | 7.0h      | 14h |
| Felix    | 6.0h      | 12h |
| Blake   | 8.0h      | 16h |
| Reid      | 8.0h      | 16h |
| Trent      | 3.0h      | 6h  |
| Omar    | 7.0h      | 14h |

Management and idle cannot be hustled.

---

## Grading

| Grade | Criteria |
|-------|----------|
| A | All orders shipped, restock complete, management duty met, no OT |
| B | All orders shipped but used OT, or minor miss on restock/management |
| C | Minor order shortfall (95%+) |
| D | Moderate shortfall (85–94%) |
| F | Major shortfall (<85%) or orders incomplete at hard stop |

---

## Running

### Train from scratch
```bash
python volt_sim/train.py
```

### Resume from latest checkpoint
```bash
python volt_sim/train.py --resume
```

### Requirements
- Python 3.10+
- PyTorch
- NumPy

```bash
pip install torch numpy
```

---

*Built with PyTorch and Chart.js. Graduate from [Dolly](https://github.com/jarmstrong158/Dolly) once your daily ops are dialed in.*
