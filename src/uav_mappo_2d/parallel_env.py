from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from .config import EnvironmentConfig, RenderConfig, RewardConfig
from .env import UAVCombatEnv
from .opponents import RuleBasedOpponent


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
    rule = RuleBasedOpponent()
    training_team = 0
    try:
        connection.send(("ready", os.getpid()))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                seed, training_team = payload
                observation, info = env.reset(seed=int(seed))
                info["worker_pid"] = os.getpid()
                connection.send(("result", (observation, info)))
            elif command == "step":
                learner_action, opponent_action = payload
                opponent_team = 1 - training_team
                joint_action = np.zeros(env.action_space.shape, dtype=np.float32)
                joint_action[training_team] = learner_action
                joint_action[opponent_team] = (
                    rule.act(env, opponent_team)
                    if opponent_action is None
                    else opponent_action
                )
                step_started_ns = time.perf_counter_ns()
                result = env.step(joint_action)
                step_finished_ns = time.perf_counter_ns()
                result[-1]["worker_pid"] = os.getpid()
                result[-1]["worker_step_started_ns"] = step_started_ns
                result[-1]["worker_step_finished_ns"] = step_finished_ns
                connection.send(("result", result))
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
    """Independent UAV environments stepped concurrently in separate processes."""

    def __init__(
        self,
        num_envs: int,
        environment_config: EnvironmentConfig,
        reward_config: RewardConfig,
        render_config: RenderConfig,
        project_root: str | Path,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        for _ in range(num_envs):
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

    def reset(
        self, indices: list[int], seeds: list[int], training_teams: list[int]
    ) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
        if not (len(indices) == len(seeds) == len(training_teams)):
            raise ValueError("indices, seeds, and training_teams must have equal length")
        for index, seed, team in zip(indices, seeds, training_teams):
            self._connections[index].send(("reset", (seed, team)))
        return {index: self._receive(index) for index in indices}

    def step(
        self,
        indices: list[int],
        learner_actions: list[np.ndarray],
        opponent_actions: list[np.ndarray | None],
    ) -> dict[int, tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]]:
        if not (len(indices) == len(learner_actions) == len(opponent_actions)):
            raise ValueError("parallel step inputs must have equal length")
        for index, learner_action, opponent_action in zip(
            indices, learner_actions, opponent_actions
        ):
            self._connections[index].send(
                ("step", (learner_action, opponent_action))
            )
        return {index: self._receive(index) for index in indices}

    def _receive(self, index: int) -> Any:
        tag, payload = self._connections[index].recv()
        if tag != "result":
            self._raise_worker(tag, payload)
        return payload

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
