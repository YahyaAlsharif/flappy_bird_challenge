"""Windows-safe environment construction for the local Flappy Bird notebook."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_process_threads(thread_count: int) -> None:
    """Limit native thread pools before a worker creates its environment."""

    value = str(max(1, int(thread_count)))
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = value


def resolve_repository_root(start: Path | None = None) -> Path:
    """Resolve the repository and fail instead of writing under an arbitrary cwd."""

    override = os.environ.get("FLAPPY_BIRD_PROJECT_ROOT")
    candidates: list[Path] = []

    if override:
        candidates.append(Path(override).expanduser())

    start_path = Path.cwd() if start is None else Path(start)
    start_path = start_path.expanduser().resolve()
    candidates.extend((start_path, *start_path.parents))

    for candidate in candidates:
        candidate = candidate.resolve()
        if (
            (candidate / "flappy-bird-solved.ipynb").is_file()
            and (candidate / "flappy_local_runtime.py").is_file()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate the Flappy Bird repository. Open the notebook from "
        "the repository root or set FLAPPY_BIRD_PROJECT_ROOT to that path."
    )


class FlappyRewardWrapper(gym.Wrapper):
    """Apply reward shaping only to training environments."""

    def __init__(
        self,
        env: gym.Env,
        survival_reward: float = 0.05,
        pipe_reward: float = 5.0,
        death_penalty: float = -5.0,
        top_penalty: float = -1.0,
    ) -> None:
        super().__init__(env)
        self.survival_reward = float(survival_reward)
        self.pipe_reward = float(pipe_reward)
        self.death_penalty = float(death_penalty)
        self.top_penalty = float(top_penalty)
        self.previous_score = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_score = int(info.get("score", 0))
        return observation, info

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(
            action
        )
        info = dict(info)
        current_score = int(info.get("score", self.previous_score))
        pipes_passed = max(0, current_score - self.previous_score)

        shaped_reward = self.survival_reward + self.pipe_reward * pipes_passed
        if terminated:
            shaped_reward += self.death_penalty
        elif float(original_reward) < 0:
            shaped_reward += self.top_penalty

        self.previous_score = current_score
        info["original_reward"] = float(original_reward)
        info["shaped_reward"] = float(shaped_reward)
        return observation, float(shaped_reward), terminated, truncated, info


class WorkerIdentityWrapper(gym.Wrapper):
    """Expose worker identity in info so health checks can verify every process."""

    def __init__(self, env: gym.Env, worker_rank: int, worker_seed: int) -> None:
        super().__init__(env)
        self.worker_rank = int(worker_rank)
        self.worker_seed = int(worker_seed)

    def _with_identity(self, info):
        enriched = dict(info)
        enriched["worker_rank"] = self.worker_rank
        enriched["worker_seed"] = self.worker_seed
        return enriched

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_identity(info)

    def step(self, action):
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = self.env.step(action)
        enriched = self._with_identity(info)
        enriched["worker_env_step_seconds"] = float(
            time.perf_counter() - started
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            enriched,
        )


@dataclass(frozen=True)
class FlappyEnvFactory:
    """Picklable callable used by SubprocVecEnv with the Windows spawn method."""

    rank: int
    base_seed: int
    score_limit: int
    pipe_gap: int = 100
    torch_threads: int = 1

    def __call__(self):
        configure_process_threads(self.torch_threads)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

        import flappy_bird_gymnasium  # noqa: F401

        worker_seed = int(self.base_seed) + int(self.rank)
        env = gym.make(
            "FlappyBird-v0",
            render_mode=None,
            use_lidar=False,
            normalize_obs=True,
            pipe_gap=int(self.pipe_gap),
            score_limit=int(self.score_limit),
            audio_on=False,
        )
        env = FlappyRewardWrapper(env)
        env = WorkerIdentityWrapper(env, self.rank, worker_seed)
        env = Monitor(env, info_keywords=("score", "worker_rank", "worker_seed"))
        env.action_space.seed(worker_seed)
        env.reset(seed=worker_seed)
        return env


def make_training_env_factories(
    n_envs: int,
    base_seed: int,
    score_limit: int,
    pipe_gap: int = 100,
    torch_threads: int = 1,
) -> list[FlappyEnvFactory]:
    """Return one independently seeded, importable callable per worker."""

    return [
        FlappyEnvFactory(
            rank=rank,
            base_seed=base_seed,
            score_limit=score_limit,
            pipe_gap=pipe_gap,
            torch_threads=torch_threads,
        )
        for rank in range(int(n_envs))
    ]


def create_training_vec_env(
    n_envs: int,
    base_seed: int,
    score_limit: int,
    pipe_gap: int = 100,
    torch_threads: int = 1,
) -> SubprocVecEnv:
    """Create a process vector environment with explicit Windows-safe spawning."""

    factories = make_training_env_factories(
        n_envs=n_envs,
        base_seed=base_seed,
        score_limit=score_limit,
        pipe_gap=pipe_gap,
        torch_threads=torch_threads,
    )
    return SubprocVecEnv(factories, start_method="spawn")


def close_environments(environments: Iterable[object]) -> None:
    """Best-effort cleanup for partially completed benchmark or smoke runs."""

    for environment in environments:
        if environment is None:
            continue
        try:
            environment.close()
        except Exception:
            pass
