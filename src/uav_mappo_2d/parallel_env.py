from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from .config import EnvironmentConfig, RenderConfig, RewardConfig
from .env import UAVCombatEnv
from .opponents import RuleBasedOpponent


@dataclass
class VectorStep:
    observations: np.ndarray
    states: np.ndarray
    active_masks: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: list[dict[str, Any]]


def _worker(
    connection: Connection,
    environment_config: EnvironmentConfig,
    reward_config: RewardConfig,
    render_config: RenderConfig,
    project_root: str,
) -> None:
    env = UAVCombatEnv(
        environment_config,
        reward_config,
        render_config,
        project_root=project_root,
    )
    opponent = RuleBasedOpponent()
    base_seed = 0
    episode_index = 0
    try:
        connection.send(("ready", os.getpid()))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                base_seed = int(payload)
                episode_index = 0
                opponent.reset()
                observations, info = env.reset(seed=base_seed)
                connection.send(
                    (
                        "result",
                        (
                            observations[0],
                            info["global_state"],
                            info["active_masks"][0],
                            info,
                        ),
                    )
                )
            elif command == "step":
                red_actions = np.asarray(payload, dtype=np.float32)
                blue_actions = opponent.act(env, 1)
                observations, rewards, terminated, _, info = env.step(
                    np.stack([red_actions, blue_actions])
                )
                done = bool(terminated)
                transition_info = dict(info)
                if done:
                    episode_index += 1
                    opponent.reset()
                    observations, reset_info = env.reset(
                        seed=base_seed + 1_000_003 * episode_index
                    )
                    current_state = reset_info["global_state"]
                    current_mask = reset_info["active_masks"][0]
                else:
                    current_state = info["global_state"]
                    current_mask = info["active_masks"][0]
                connection.send(
                    (
                        "result",
                        (
                            observations[0],
                            current_state,
                            current_mask,
                            float(rewards[0, 0]),
                            done,
                            transition_info,
                        ),
                    )
                )
            elif command == "close":
                break
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        env.close()
        connection.close()


class ParallelUAVEnvs:
    """Persistent one-process-per-environment vector sampler."""

    def __init__(
        self,
        num_envs: int,
        environment_config: EnvironmentConfig,
        reward_config: RewardConfig,
        render_config: RenderConfig,
        project_root: str | Path,
        *,
        base_seed: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        for _ in range(self.num_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    child,
                    environment_config,
                    reward_config,
                    render_config,
                    str(Path(project_root).resolve()),
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        self.worker_pids = []
        for connection in self._connections:
            tag, payload = connection.recv()
            if tag != "ready":
                self._raise_worker(tag, payload)
            self.worker_pids.append(int(payload))

    @staticmethod
    def _raise_worker(tag: str, payload: Any) -> None:
        if tag == "error":
            raise RuntimeError(f"Parallel environment worker failed:\n{payload}")
        raise RuntimeError(f"Unexpected worker response: {tag!r}")

    def _receive(self, index: int) -> Any:
        tag, payload = self._connections[index].recv()
        if tag != "result":
            self._raise_worker(tag, payload)
        return payload

    def reset(
        self, seeds: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if seeds is None:
            seeds = [self.base_seed + index for index in range(self.num_envs)]
        if len(seeds) != self.num_envs:
            raise ValueError("reset requires one seed per environment")
        for connection, seed in zip(self._connections, seeds):
            connection.send(("reset", int(seed)))
        results = [self._receive(index) for index in range(self.num_envs)]
        return (
            np.stack([result[0] for result in results]),
            np.stack([result[1] for result in results]),
            np.stack([result[2] for result in results]),
            [result[3] for result in results],
        )

    def step(self, red_actions: np.ndarray) -> VectorStep:
        red_actions = np.asarray(red_actions, dtype=np.float32)
        expected = (self.num_envs, UAVCombatEnv.N_AGENTS, UAVCombatEnv.ACTION_DIM)
        if red_actions.shape != expected:
            raise ValueError(f"Expected red actions {expected}, got {red_actions.shape}")
        for index, connection in enumerate(self._connections):
            connection.send(("step", red_actions[index]))
        results = [self._receive(index) for index in range(self.num_envs)]
        return VectorStep(
            observations=np.stack([result[0] for result in results]),
            states=np.stack([result[1] for result in results]),
            active_masks=np.stack([result[2] for result in results]),
            rewards=np.asarray([result[3] for result in results], dtype=np.float32),
            dones=np.asarray([result[4] for result in results], dtype=np.float32),
            infos=[result[5] for result in results],
        )

    def close(self) -> None:
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> "ParallelUAVEnvs":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
