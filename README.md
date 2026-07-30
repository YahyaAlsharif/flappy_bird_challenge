# Flappy Bird — Deep Reinforcement Learning Agent

> **STATUS: SKELETON.** Headings and placeholders only. Full prose is written in a
> later pass, after the outline below is approved. Every `_TODO_` marks content that
> still has to be written or verified. Nothing in this file should be treated as
> final copy.

<!-- Placeholder: one-paragraph hook — what this repo is, the headline result
     (best recorded score 5,120), and the single most useful image or GIF. -->

---

## 1. Project Overview

_TODO_ — What the project is: a Dueling Double DQN agent that learns to play Flappy Bird
from a 12-feature state vector.

_TODO_ — What is in the repository: two training notebooks (Kaggle and local), a shared
model/checkpoint module, two inference viewers, and the artefacts of the runs.

---

## 2. Challenge Background

_TODO_ — The challenge was **suggested by Mr Abdullah**, one of the instructors during
the reinforcement-learning weeks of the **KAUST Academy AI Summer School**.

_TODO_ — There was **no official leaderboard** and no formal submission process.

_TODO_ — I treated it as an honest personal challenge: a way to apply and stress-test what
was taught during the RL weeks, and to see how far a from-scratch DQN variant could be
pushed on my own hardware.

---

## 3. Reinforcement Learning Algorithm and Environment

### 3.1 Environment

_TODO_ — `flappy-bird-gymnasium` `FlappyBird-v0`, `use_lidar=False`, `normalize_obs=True`,
`pipe_gap=100`, `audio_on=False`.

### 3.2 Observation and Action Space

_TODO_ — State: 12 normalised features. Actions: 2 (idle / flap).

### 3.3 Algorithm

_TODO_ — Dueling Double DQN with Prioritized Experience Replay. Network: shared trunk
`12 -> 192 -> 192`, then separate value (`192 -> 128 -> 1`) and advantage
(`192 -> 128 -> 2`) streams, combined as `V + A - mean(A)`.
Source of truth: [`flappy_model.py`](flappy_model.py).

### 3.4 Reward Design

_TODO_ — Table of the shaping terms and the rationale for each
(`survival +0.05`, `pipe +5.0`, `death -5.0`, `top -1.0`, `flap -0.01`).
_TODO_ — Note that shaping is **training-only**; evaluation always uses the unmodified
environment and the true `info["score"]`.
_TODO_ — Note what was deliberately *excluded* (gap-centering / altitude rewards) and why
(reward hacking: the agent hovers instead of reacting).
Source of truth: [`flappy_local_runtime.py`](flappy_local_runtime.py).

### 3.5 Replay, Target Network, and Exploration

_TODO_ — Prioritized replay (`alpha`, `beta` schedule), target-network update strategy,
epsilon-greedy schedule with the biased exploration flap probability.

---

## 4. Training Setup

_TODO_ — Intro sentence: training was done in two distinct phases, on two different
machines, with different hyperparameters. They must not be conflated.

### 4.1 Phase 1 — Kaggle GPU

_TODO_ — Notebook: [`flappy-bird-solved.ipynb`](flappy-bird-solved.ipynb).
_TODO_ — Table: 3,000,000 steps, 3 envs, hard target sync every 10,000 steps,
epsilon 1.0 -> 0.05 over 400,000 steps, ~5.05 h.
_TODO_ — Produced checkpoint `old_artifects/flappy_ddqn_score_1000.pt` (step 2,750,001).
Source of truth: `old_artifects/flappy_ddqn_score_1000_metadata.json`.

### 4.2 Phase 2 — Local GPU / CPU (v2)

_TODO_ — Notebook: [`flappy-bird-solved-local.ipynb`](flappy-bird-solved-local.ipynb).
_TODO_ — Table: 5,000,000 transitions, 8 parallel envs, soft (Polyak) target updates
`tau=5e-3`, budget-scaled epsilon decay, added `flap_cost`, ~2.56 h, ~545 transitions/s.
Source of truth: `artifacts/ddqn_local_v2/metadata.json` and `run_configuration.json`.

### 4.3 What Changed Between the Two Runs, and Why

_TODO_ — The two reliability defects the v2 notebook set out to fix:
1. a fixed 400,000-step epsilon decay that collapsed exploration far too early for a
   multi-million-step run;
2. hard target-network resyncs every 10,000 steps that destabilised the value function.
_TODO_ — Evidence: the Kaggle run's own evaluation history shows mean 5-seed score falling
from 861 (step 2,750,001) to 396 (step 3,000,000) — the exported "best" model was a
favourable snapshot rather than a converged policy.
_TODO_ — Honest note: v2 is the more *stable* training procedure, but it did **not**
reproduce the Kaggle checkpoint's peak score. Discuss this openly.

---

## 5. Repository Structure

_TODO_ — Annotated file tree covering:

```
flappy-bird-solved.ipynb                  Kaggle training notebook (phase 1)
flappy-bird-solved-local.ipynb            Local training notebook (phase 2, v2)
flappy_model.py                           Shared network + checkpoint schema/validation
flappy_local_runtime.py                   Windows-safe vector env + reward shaping
play_trained_model.py                     Viewer for the Kaggle checkpoint
play_improved_model.py                    Viewer for the local v2 checkpoint
requirements.txt                          Pinned dependency ranges
artifacts/ddqn_local_v2/                  Local v2 run: models, plots, logs, evaluations
artifacts/inference_screenshots/           Score evidence, including the 5,120 run
old_artifects/                            Kaggle checkpoint, its metadata, demo GIF, prior notebook
```

_TODO_ — Note that `old_artifects` is a legacy directory name (typo for "artifacts") kept
so existing checkpoint paths in the metadata stay valid.

---

## 6. Installation

_TODO_ — Python 3.11, virtual environment creation, `pip install -r requirements.txt`.
_TODO_ — Note the Windows-specific bits: `SubprocVecEnv` with `start_method="spawn"`,
`SDL_VIDEODRIVER=dummy` for headless training.
_TODO_ — Note that the viewers run on CPU by default and need a display; training does not.

---

## 7. Training

_TODO_ — How to run the local notebook end to end.
_TODO_ — Environment-variable switches:
`FLAPPY_RUN_MODE` (`full` / `smoke` / `benchmark`), `FLAPPY_TOTAL_TRANSITIONS`,
`FLAPPY_N_ENVS`, `FLAPPY_GRADIENT_STEPS`, `FLAPPY_EXPERIMENT_NAME`,
`FLAPPY_FORCE_FRESH_START`, `FLAPPY_BIRD_PROJECT_ROOT`.
_TODO_ — Resume behaviour: a `full` run writes to a stable directory and auto-resumes from
`checkpoints/latest.pt`.
_TODO_ — **Caveat to state explicitly:** `latest.pt` is not published in this repository
(it is an ignored rolling resume pointer), so a fresh clone starts training from scratch.

---

## 8. Evaluation

_TODO_ — Periodic five-slot evaluation during training: fixed seeds, deterministic greedy
policy, score limit, and the checkpoint ranking key
(`highest_completed_level`, `slots_reaching_limit`, `min`, `mean`, `max`).
_TODO_ — Final unseen-seed holdout: seeds 1,000,001–1,000,005, run once, never used for
checkpoint selection.
_TODO_ — Where the numbers live: `final_evaluation.json`, `evaluation_history.json`,
`result_summary.json`.

---

## 9. Running the Trained Model Viewer

_TODO_ — Two viewers and which checkpoint each defaults to:

| Script | Default checkpoint | Model |
|---|---|---|
| `play_trained_model.py` | `old_artifects/flappy_ddqn_score_1000.pt` | Kaggle (phase 1) |
| `play_improved_model.py` | `artifacts/ddqn_local_v2/best_model.pt` | Local v2 (phase 2) |

_TODO_ — CLI flags: `--checkpoint`, `--device`, `--fps`, `--seed`, `--screenshot-dir`.
_TODO_ — Controls table (verified against the code):

| Key | Action |
|---|---|
| `S` | Save a screenshot |
| `P` / `Space` | Pause / resume |
| `R` | Restart (after death) |
| `Esc` / `Q` | Exit |

_TODO_ — Note the deterministic seeding behaviour: restarts increment the seed.
_TODO_ — Expected checkpoint paths and what happens if one is missing
(`CheckpointError` with an actionable message).

---

## 10. Results

### 10.1 Headline — Best Recorded Score: 5,120

_TODO_ — State the number prominently, then immediately qualify it:
this is the **best single observed episode**, not an average and not a guaranteed result.

_TODO_ — Full provenance, all of which is legible in the screenshot itself:

| Field | Value |
|---|---|
| Pipe score | **5,120** |
| Model | `old_artifects/flappy_ddqn_score_1000.pt` (Kaggle, checkpoint step 2,750,001) |
| Script | `play_trained_model.py` |
| Seed | 777 |
| Device | CPU, 30 FPS target |
| Episode | 1 (uncapped) |
| Episode frames | 192,900 |
| Episode reward | 23,896.9 |
| Ended by | Bird crashed (genuine termination, not a score cap) |

_TODO_ — Also mention the independent Kaggle-notebook run of the same checkpoint that
reached 5,090 (`5090_Score_Kaggle.png`), as corroboration that 5,120 was not a one-off
artefact of the local viewer.

### 10.2 Formal Evaluation Results

_TODO_ — Table of the **local v2** final unseen-seed holdout (5 seeds, limit 10,000,
never reached): scores 1187 / 301 / 235 / 190 / 85 -> mean **399.6**, median **235**,
std 399.9. Best periodic checkpoint: step 4,250,000.

_TODO_ — Table of the **Kaggle** final deterministic evaluation (3 seeds):
1000 / 1000 / 584 -> mean 861.3.

### 10.3 Capped vs Uncapped Evaluations — Read This Before Comparing Numbers

_TODO_ — Explain the distinction carefully, because it is the easiest thing to
misrepresent:
- the Kaggle evaluation used a **score limit of 1,000**, so two of its three episodes were
  **truncated at the cap** and are lower bounds, not genuine crash scores;
- the local v2 holdout used a limit of 10,000 which was **never reached**, so all five
  episodes are genuine terminations;
- the 5,120 run was **uncapped** and ended in a real crash.

_TODO_ — Explicit statement: a single lucky episode is not the same as expected
performance. High-variance results are the norm for this agent; the mean is the honest
summary, and the mean is far below 5,120.

### 10.4 Training Curves

_TODO_ — Embed and briefly interpret:
`artifacts/ddqn_local_v2/training_scores.png`, `evaluation_scores.png`,
`episode_rewards.png`, `episode_lengths.png`, `training_loss.png`,
`epsilon_schedule.png`, `throughput.png`.

---

## 11. Comparison With the Reported Human Record

> _TODO — NOT YET RESEARCHED. Placeholder only; nothing here is verified._

_TODO_ — Verify the commonly reported human Flappy Bird record from credible current
sources before writing anything in this section. A figure of ~2,132 has been mentioned but
is **unconfirmed** and must not be stated as fact until checked.

_TODO_ — Requirements for this section when it is written:
- cite the source and record the **date checked**;
- do not rely on search snippets, social-media posts, or unsourced articles alone;
- if sources disagree, say plainly that the human record is **unofficial and
  inconsistently reported** rather than asserting a definitive number;
- if a figure is supported, frame the comparison precisely — a single recorded agent
  episode of 5,120 against a reported human figure, not "the model beats humans".

---

## 12. Screenshots and Visual Evidence

_TODO_ — Embed, with captions stating exactly what each one shows:

| Asset | Shows |
|---|---|
| `artifacts/inference_screenshots/flappy_20260726_195516_270794_score_5120_reward_p23896p900_gameover.png` | The 5,120 run, game-over frame with the full status panel |
| `artifacts/inference_screenshots/flappy_20260726_195604_782431_score_5120_reward_p23896p900_manual.png` | The same run, manual capture |
| `5090_Score_Kaggle.png` | Score 5,090 for the same checkpoint inside the Kaggle notebook |
| `artifacts/inference_screenshots/flappy_20260727_035724_207339_score_1096_reward_p5118p400_gameover.png` | A separate 1,096 run — evidence of run-to-run variance |
| `artifacts/inference_screenshots/improved_model/flappy_20260727_032752_695158_score_31_reward_p148p400_gameover.png` | A weak local-v2 episode — included deliberately, not hidden |
| `artifacts/ddqn_local_v2/demonstration.gif` | Local v2 agent playing |
| `old_artifects/flappy_ddqn_first_100_scores_downscaled.gif` | Kaggle agent's first 100 scores |

---

## 13. Limitations and Reproducibility Notes

_TODO_ — High variance across seeds is the dominant limitation; the mean is roughly two
orders of magnitude below the best recorded episode.
_TODO_ — The local v2 run resumed mid-training with an **empty replay buffer** (see the
resume line in `train_log.txt`), which is a real confound when reading its curves.
_TODO_ — Only 5 checkpoints of the original 21 are published; the rest are reproducible by
re-running the notebook.
_TODO_ — Seeds are fixed but exact reproduction is not guaranteed across PyTorch versions,
thread counts, or worker counts.
_TODO_ — No hyperparameter sweep was performed; the configuration was chosen from a single
throughput benchmark plus reasoning about the two failure modes described in §4.3.
_TODO_ — The agent was never evaluated against the original mobile game's physics, only
against `flappy-bird-gymnasium`.

---

## 14. Acknowledgements

_TODO_ — **Mr Abdullah**, for suggesting the challenge during the reinforcement-learning
weeks of the KAUST Academy AI Summer School.
_TODO_ — KAUST Academy AI Summer School.
_TODO_ — The `flappy-bird-gymnasium` environment authors.
_TODO_ — Kaggle, for the GPU hours used in phase 1.

---

## 15. License

_TODO_ — Decide and add a license (none is currently present in the repository).
