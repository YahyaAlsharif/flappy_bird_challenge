"""Run the legacy Flappy Bird DDQN checkpoint in a local Pygame viewer."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn

import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "old_artifects" / "flappy_ddqn_score_1000.pt"
)
DEFAULT_SCREENSHOT_DIR = PROJECT_ROOT / "artifacts" / "inference_screenshots"
EXPECTED_ALGORITHM = (
    "Dueling Double DQN with Prioritized Experience Replay"
)
DEFAULT_SEED = 42
DEFAULT_FPS = 30
PANEL_WIDTH = 520
WINDOW_TITLE = "Flappy Bird - Legacy Dueling DDQN Viewer"


class ViewerError(RuntimeError):
    """An actionable viewer startup or runtime error."""


class DuelingQNetwork(nn.Module):
    """Exact network architecture used by the legacy Kaggle checkpoint."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.features = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.features(states)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


@dataclass(frozen=True)
class ModelInfo:
    checkpoint_path: Path
    algorithm: str
    global_step: int
    state_dim: int
    action_dim: int
    hidden_dim: int
    pipe_gap: int


def positive_fps(value: str) -> int:
    try:
        fps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("FPS must be an integer.") from exc
    if not 1 <= fps <= 240:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 240.")
    return fps


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ViewerError(
                f"CUDA was explicitly requested ({requested!r}), but "
                "torch.cuda.is_available() is False."
            )
        try:
            device = torch.device(normalized)
        except (RuntimeError, ValueError) as exc:
            raise ViewerError(
                f"Unsupported CUDA device specification: {requested!r}."
            ) from exc
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ViewerError(
                f"CUDA device index {device.index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )
        return device
    raise ViewerError(
        f"Unsupported device {requested!r}. Use 'cpu', 'cuda', or 'cuda:N'."
    )


def _required_positive_int(
    config: Mapping,
    key: str,
    checkpoint_path: Path,
) -> int:
    if key not in config:
        raise ViewerError(
            f"Checkpoint configuration is missing required key {key!r}: "
            f"{checkpoint_path}"
        )
    try:
        value = int(config[key])
    except (TypeError, ValueError) as exc:
        raise ViewerError(
            f"Checkpoint configuration key {key!r} must be an integer: "
            f"{checkpoint_path}"
        ) from exc
    if value <= 0:
        raise ViewerError(
            f"Checkpoint configuration key {key!r} must be positive, "
            f"got {value}: {checkpoint_path}"
        )
    return value


def load_inference_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DuelingQNetwork, ModelInfo]:
    path = checkpoint_path.expanduser().resolve()
    if not path.is_file():
        raise ViewerError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ViewerError(
            f"Could not load checkpoint {path}. It may be corrupt or an "
            f"unsupported checkpoint type: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise ViewerError(
            f"Expected checkpoint {path} to contain a mapping, "
            f"got {type(checkpoint).__name__}."
        )

    algorithm = checkpoint.get("algorithm")
    if algorithm != EXPECTED_ALGORITHM:
        raise ViewerError(
            f"Checkpoint algorithm mismatch in {path}. Expected "
            f"{EXPECTED_ALGORITHM!r}, got {algorithm!r}."
        )

    config = checkpoint.get("model_config", checkpoint.get("config"))
    if not isinstance(config, Mapping):
        raise ViewerError(
            f"Checkpoint {path} does not contain a valid 'config' or "
            "'model_config' mapping."
        )

    state_dim = _required_positive_int(config, "state_dim", path)
    action_dim = _required_positive_int(config, "action_dim", path)
    hidden_dim = _required_positive_int(config, "hidden_dim", path)
    pipe_gap = _required_positive_int(config, "pipe_gap", path)

    state_dict = checkpoint.get("online_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ViewerError(
            f"Checkpoint {path} is missing a valid 'online_state_dict'."
        )

    model = DuelingQNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ViewerError(
            f"Checkpoint/model shape or key mismatch in {path}; strict "
            f"loading failed: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    try:
        global_step = int(checkpoint["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerError(
            f"Checkpoint {path} is missing a valid 'global_step'."
        ) from exc

    info = ModelInfo(
        checkpoint_path=path,
        algorithm=str(algorithm),
        global_step=global_step,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        pipe_gap=pipe_gap,
    )
    return model, info


def make_evaluation_environment(
    model_info: ModelInfo,
) -> gym.Env:
    try:
        environment = gym.make(
            "FlappyBird-v0",
            render_mode="rgb_array",
            use_lidar=False,
            normalize_obs=True,
            pipe_gap=model_info.pipe_gap,
            score_limit=None,
            audio_on=False,
        )
    except FileNotFoundError as exc:
        raise ViewerError(
            f"Flappy Bird rendering assets are missing: {exc}"
        ) from exc
    except Exception as exc:
        raise ViewerError(
            "Could not initialize FlappyBird-v0. Confirm that "
            "flappy-bird-gymnasium==0.4.0 and its rendering assets are "
            f"installed: {type(exc).__name__}: {exc}"
        ) from exc

    observation_shape = environment.observation_space.shape
    expected_shape = (model_info.state_dim,)
    if observation_shape != expected_shape:
        environment.close()
        raise ViewerError(
            "Environment/checkpoint observation mismatch: "
            f"environment={observation_shape}, checkpoint={expected_shape}."
        )

    action_count = getattr(environment.action_space, "n", None)
    if action_count != model_info.action_dim:
        environment.close()
        raise ViewerError(
            "Environment/checkpoint action mismatch: "
            f"environment={action_count}, checkpoint={model_info.action_dim}."
        )
    return environment


class FlappyInferenceViewer:
    """Responsive, inference-only Pygame application."""

    def __init__(
        self,
        model: DuelingQNetwork,
        model_info: ModelInfo,
        device: torch.device,
        target_fps: int,
        seed: int,
        screenshot_dir: Path,
    ) -> None:
        self.model = model
        self.model_info = model_info
        self.device = device
        self.target_fps = int(target_fps)
        self.base_seed = int(seed)
        self.screenshot_dir = screenshot_dir.expanduser().resolve()

        self.environment: gym.Env | None = None
        self.screen: pygame.Surface | None = None
        self.clock = pygame.time.Clock()
        self.font: pygame.font.Font | None = None
        self.small_font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None

        self.running = True
        self.state = "RUNNING"
        self.end_reason = ""
        self.episode_number = 1
        self.current_seed = self.base_seed
        self.episode_frames = 0
        self.total_rendered_frames = 0
        self.episode_reward = 0.0
        self.total_reward = 0.0
        self.pipe_score = 0
        self.best_pipe_score = 0
        self.current_action: int | None = None
        self.observation: np.ndarray | None = None
        self.game_surface: pygame.Surface | None = None
        self.last_screenshot_path: Path | None = None
        self.screenshot_error = ""

        self._initialize()

    def _initialize(self) -> None:
        try:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.environment = make_evaluation_environment(self.model_info)
            self.observation, info = self.environment.reset(
                seed=self.current_seed
            )
            self.observation = self._validated_observation(self.observation)
            self.pipe_score = int(info.get("score", 0))
            frame = self._validated_frame(self.environment.render())
            self._set_game_frame(frame)

            game_height, game_width = frame.shape[:2]
            self.screen = pygame.display.set_mode(
                (game_width + PANEL_WIDTH, game_height)
            )
            pygame.display.set_caption(WINDOW_TITLE)
            self.title_font = pygame.font.Font(None, 28)
            self.font = pygame.font.Font(None, 21)
            self.small_font = pygame.font.Font(None, 19)
        except ViewerError:
            self.close()
            raise
        except FileNotFoundError as exc:
            self.close()
            raise ViewerError(
                f"Required rendering asset was not found: {exc}"
            ) from exc
        except pygame.error as exc:
            self.close()
            raise ViewerError(
                f"Pygame window or renderer initialization failed: {exc}"
            ) from exc
        except Exception as exc:
            self.close()
            raise ViewerError(
                f"Viewer initialization failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _validated_observation(self, observation) -> np.ndarray:
        array = np.asarray(observation)
        expected_shape = (self.model_info.state_dim,)
        if array.shape != expected_shape:
            raise ViewerError(
                f"Unexpected observation shape {array.shape}; "
                f"expected {expected_shape}."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ViewerError(
                f"Unexpected observation dtype {array.dtype}; "
                "expected a numeric 12-feature vector."
            )
        if not np.isfinite(array).all():
            raise ViewerError("Observation contains NaN or infinite values.")

        if self.environment is not None:
            low = np.asarray(self.environment.observation_space.low)
            high = np.asarray(self.environment.observation_space.high)
            if np.any(array < low - 1e-6) or np.any(array > high + 1e-6):
                raise ViewerError(
                    "Observation falls outside the configured normalized "
                    f"bounds: min={array.min()}, max={array.max()}."
                )
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _validated_frame(frame) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise ViewerError(
                f"Renderer returned {type(frame).__name__}; expected an RGB "
                "NumPy array."
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ViewerError(
                f"Unexpected rendered frame shape {frame.shape}; expected "
                "(height, width, 3)."
            )
        if frame.dtype != np.uint8:
            raise ViewerError(
                f"Unexpected rendered frame dtype {frame.dtype}; "
                "expected uint8."
            )
        return frame

    def _set_game_frame(self, frame: np.ndarray) -> None:
        try:
            self.game_surface = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2))
            )
        except pygame.error as exc:
            raise ViewerError(
                f"Could not convert the game frame for display: {exc}"
            ) from exc

    def _select_action(self) -> int:
        if self.observation is None:
            raise ViewerError("No observation is available for inference.")
        state_tensor = torch.as_tensor(
            self.observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        expected_input_shape = (1, self.model_info.state_dim)
        if tuple(state_tensor.shape) != expected_input_shape:
            raise ViewerError(
                f"Unexpected network input shape {tuple(state_tensor.shape)}; "
                f"expected {expected_input_shape}."
            )

        with torch.inference_mode():
            q_values = self.model(state_tensor)

        expected_output_shape = (1, self.model_info.action_dim)
        if tuple(q_values.shape) != expected_output_shape:
            raise ViewerError(
                f"Unexpected network output shape {tuple(q_values.shape)}; "
                f"expected {expected_output_shape}."
            )
        if not torch.isfinite(q_values).all():
            raise ViewerError("Model produced NaN or infinite Q-values.")

        action = int(q_values.argmax(dim=1).item())
        if not 0 <= action < self.model_info.action_dim:
            raise ViewerError(
                f"Model produced invalid action {action}; expected an index "
                f"from 0 to {self.model_info.action_dim - 1}."
            )
        return action

    def _step_episode(self) -> None:
        if self.environment is None:
            raise ViewerError("The Flappy Bird environment is not available.")

        action = self._select_action()
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.environment.step(action)

        self.current_action = action
        self.episode_frames += 1
        self.episode_reward += float(reward)
        self.total_reward += float(reward)
        self.pipe_score = int(info.get("score", self.pipe_score))
        self.best_pipe_score = max(
            self.best_pipe_score,
            self.pipe_score,
        )
        self.observation = self._validated_observation(observation)
        frame = self._validated_frame(self.environment.render())
        self._set_game_frame(frame)

        if terminated or truncated:
            reason = (
                "bird crashed"
                if terminated
                else "environment truncated unexpectedly"
            )
            self._enter_game_over(reason)

    def _enter_game_over(self, reason: str) -> None:
        if self.state == "GAME OVER":
            return
        self.state = "GAME OVER"
        self.end_reason = reason
        self.best_pipe_score = max(
            self.best_pipe_score,
            self.pipe_score,
        )
        self.save_screenshot("gameover")

    def restart_episode(self) -> None:
        if self.state != "GAME OVER":
            return
        if self.environment is None:
            raise ViewerError("The Flappy Bird environment is not available.")

        self.episode_number += 1
        self.current_seed = self.base_seed + self.episode_number - 1
        self.episode_frames = 0
        self.episode_reward = 0.0
        self.pipe_score = 0
        self.current_action = None
        self.end_reason = ""
        self.screenshot_error = ""

        observation, info = self.environment.reset(seed=self.current_seed)
        self.observation = self._validated_observation(observation)
        self.pipe_score = int(info.get("score", 0))
        frame = self._validated_frame(self.environment.render())
        self._set_game_frame(frame)
        self.state = "RUNNING"

    def toggle_pause(self) -> None:
        if self.state == "RUNNING":
            self.state = "PAUSED"
        elif self.state == "PAUSED":
            self.state = "RUNNING"

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            elif event.key == pygame.K_s:
                self.save_screenshot("manual")
            elif event.key == pygame.K_r and self.state == "GAME OVER":
                self.restart_episode()
            elif event.key in (pygame.K_p, pygame.K_SPACE):
                self.toggle_pause()

    @staticmethod
    def _reward_filename_token(reward: float) -> str:
        sign = "m" if reward < 0 else "p"
        magnitude = f"{abs(reward):.3f}".replace(".", "p")
        return f"{sign}{magnitude}"

    def _unique_screenshot_path(self, reason: str) -> Path:
        timestamp = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        reward = self._reward_filename_token(self.episode_reward)
        stem = (
            f"flappy_{timestamp}_score_{self.pipe_score}_"
            f"reward_{reward}_{reason}"
        )
        candidate = self.screenshot_dir / f"{stem}.png"
        suffix = 1
        while candidate.exists():
            candidate = self.screenshot_dir / f"{stem}_{suffix:03d}.png"
            suffix += 1
        return candidate

    def save_screenshot(self, reason: str) -> Path | None:
        try:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self._unique_screenshot_path(reason)
            self.last_screenshot_path = path
            self.screenshot_error = ""
            self._present()
            if self.screen is None:
                raise ViewerError("The Pygame window surface is unavailable.")
            pygame.image.save(self.screen, str(path))
            return path
        except Exception as exc:
            self.screenshot_error = (
                f"Screenshot save failed: {type(exc).__name__}: {exc}"
            )
            print(self.screenshot_error, file=sys.stderr)
            try:
                self._present()
            except Exception:
                pass
            return None

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    def _draw_wrapped(
        self,
        text: str,
        x: int,
        y: int,
        max_width: int,
        color: tuple[int, int, int],
        line_height: int = 17,
        max_lines: int = 4,
    ) -> int:
        if self.screen is None or self.small_font is None:
            return y

        lines: list[str] = []
        remaining = text.strip()
        while remaining:
            if self.small_font.size(remaining)[0] <= max_width:
                lines.append(remaining)
                break

            low = 1
            high = len(remaining)
            while low <= high:
                midpoint = (low + high) // 2
                if self.small_font.size(remaining[:midpoint])[0] <= max_width:
                    low = midpoint + 1
                else:
                    high = midpoint - 1

            fitted_length = max(1, high)
            fitted_text = remaining[:fitted_length]
            word_break = fitted_text.rfind(" ")
            if word_break > 0:
                fitted_length = word_break
                fitted_text = remaining[:fitted_length]

            lines.append(fitted_text.rstrip())
            remaining = remaining[fitted_length:].lstrip()

        for line in lines[:max_lines]:
            rendered = self.small_font.render(line, True, color)
            self.screen.blit(rendered, (x, y))
            y += line_height
        return y

    def _draw_panel(self, game_width: int, game_height: int) -> None:
        if (
            self.screen is None
            or self.title_font is None
            or self.font is None
            or self.small_font is None
        ):
            raise ViewerError("Pygame fonts or window surface are unavailable.")

        panel_rect = pygame.Rect(
            game_width,
            0,
            PANEL_WIDTH,
            game_height,
        )
        pygame.draw.rect(self.screen, (20, 24, 31), panel_rect)
        pygame.draw.line(
            self.screen,
            (64, 74, 88),
            (game_width, 0),
            (game_width, game_height),
            2,
        )

        x = game_width + 18
        y = 12
        title = self.title_font.render(
            "Inference status",
            True,
            (245, 247, 250),
        )
        self.screen.blit(title, (x, y))
        y += 31

        action_name = {
            None: "-",
            0: "IDLE (0)",
            1: "FLAP (1)",
        }.get(self.current_action, str(self.current_action))
        state_color = {
            "RUNNING": (92, 214, 134),
            "PAUSED": (255, 202, 87),
            "GAME OVER": (255, 106, 106),
        }.get(self.state, (245, 247, 250))
        rows = [
            ("State", self.state, state_color),
            ("Current FPS", f"{self.clock.get_fps():.1f}", None),
            ("Target FPS", str(self.target_fps), None),
            ("Rendered frames", f"{self.total_rendered_frames:,}", None),
            ("Episode frames", f"{self.episode_frames:,}", None),
            ("Episode reward", f"{self.episode_reward:.3f}", None),
            ("Total reward", f"{self.total_reward:.3f}", None),
            ("Pipe score", f"{self.pipe_score:,}", None),
            ("Best pipe score", f"{self.best_pipe_score:,}", None),
            ("Action", action_name, None),
            ("Episode", f"{self.episode_number:,}", None),
            ("Seed", str(self.current_seed), None),
            ("Model", self.model_info.checkpoint_path.name, None),
            ("Device", str(self.device), None),
        ]
        label_color = (157, 167, 181)
        value_color = (235, 239, 245)
        for label, value, override_color in rows:
            label_surface = self.font.render(
                f"{label}:",
                True,
                label_color,
            )
            value_surface = self.font.render(
                value,
                True,
                override_color or value_color,
            )
            self.screen.blit(label_surface, (x, y))
            self.screen.blit(value_surface, (x + 150, y))
            y += 21

        config_text = (
            f"Architecture: {self.model_info.state_dim} -> "
            f"{self.model_info.hidden_dim} -> "
            f"{self.model_info.hidden_dim}; actions={self.model_info.action_dim}; "
            f"checkpoint step={self.model_info.global_step:,}"
        )
        y = self._draw_wrapped(
            config_text,
            x,
            y + 2,
            PANEL_WIDTH - 36,
            (183, 191, 203),
            max_lines=2,
        )

        if self.end_reason:
            y = self._draw_wrapped(
                f"End reason: {self.end_reason}",
                x,
                y + 2,
                PANEL_WIDTH - 36,
                (255, 150, 150),
                max_lines=1,
            )

        controls = (
            "Controls: S screenshot | P/Space pause | "
            "R restart after death | Esc/Q exit"
        )
        y = self._draw_wrapped(
            controls,
            x,
            y + 2,
            PANEL_WIDTH - 36,
            (157, 167, 181),
            max_lines=2,
        )

        if self.screenshot_error:
            self._draw_wrapped(
                self.screenshot_error,
                x,
                y + 2,
                PANEL_WIDTH - 36,
                (255, 106, 106),
                max_lines=4,
            )
        elif self.last_screenshot_path is not None:
            self._draw_wrapped(
                "Screenshot: "
                + self._display_path(self.last_screenshot_path),
                x,
                y + 2,
                PANEL_WIDTH - 36,
                (115, 195, 255),
                max_lines=4,
            )

    def _present(self) -> None:
        if self.screen is None or self.game_surface is None:
            raise ViewerError("The Pygame display surface is unavailable.")
        self.total_rendered_frames += 1
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.game_surface, (0, 0))
        game_width, game_height = self.game_surface.get_size()
        self._draw_panel(game_width, game_height)
        pygame.display.flip()

    def run(self) -> dict[str, object]:
        try:
            while self.running:
                self._handle_events()
                if not self.running:
                    break
                if self.state == "RUNNING":
                    self._step_episode()
                self._present()
                self.clock.tick(self.target_fps)
        finally:
            self.close()

        return {
            "episodes": self.episode_number,
            "best_pipe_score": self.best_pipe_score,
            "total_reward": self.total_reward,
            "rendered_frames": self.total_rendered_frames,
        }

    def close(self) -> None:
        environment = self.environment
        self.environment = None
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass
        try:
            pygame.quit()
        except Exception:
            pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy Flappy Bird Dueling DDQN checkpoint in an "
            "inference-only desktop viewer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu, cuda, or cuda:N (default: cpu).",
    )
    parser.add_argument(
        "--fps",
        type=positive_fps,
        default=DEFAULT_FPS,
        help=f"Target display FPS (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"First episode seed (default: {DEFAULT_SEED}); restarts "
            "increment it deterministically."
        ),
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=DEFAULT_SCREENSHOT_DIR,
        help=(
            "Directory for automatic and manual screenshots "
            f"(default: {DEFAULT_SCREENSHOT_DIR})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    viewer: FlappyInferenceViewer | None = None
    try:
        device = resolve_device(args.device)
        model, model_info = load_inference_model(
            args.checkpoint,
            device,
        )
        print(f"Checkpoint: {model_info.checkpoint_path}")
        print(
            "Model: Dueling DDQN | "
            f"step={model_info.global_step:,} | "
            f"input=(1, {model_info.state_dim}) | "
            f"actions={model_info.action_dim}"
        )
        print(
            f"Device: {device} | Target FPS: {args.fps} | "
            f"Seed: {args.seed}"
        )
        print(f"Screenshots: {args.screenshot_dir.expanduser().resolve()}")
        print("Controls: S screenshot, P/Space pause, R restart, Esc/Q exit")

        viewer = FlappyInferenceViewer(
            model=model,
            model_info=model_info,
            device=device,
            target_fps=args.fps,
            seed=args.seed,
            screenshot_dir=args.screenshot_dir,
        )
        summary = viewer.run()
        print(
            "Viewer closed cleanly | "
            f"episodes={summary['episodes']} | "
            f"best_score={summary['best_pipe_score']} | "
            f"rendered_frames={summary['rendered_frames']}"
        )
        return 0
    except KeyboardInterrupt:
        if viewer is not None:
            viewer.close()
        print("Viewer interrupted; resources released.")
        return 130
    except ViewerError as exc:
        if viewer is not None:
            viewer.close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if viewer is not None:
            viewer.close()
        print(
            f"Unexpected fatal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
