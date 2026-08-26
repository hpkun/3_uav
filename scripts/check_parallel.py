"""Short diagnostic proving that environment workers execute env.step concurrently."""

from pathlib import Path

import numpy as np

from uav_mappo_2d.config import load_config
from uav_mappo_2d.parallel_env import ParallelUAVEnvs


def maximum_overlap(intervals: list[tuple[int, int]]) -> int:
    events = [(start, 1) for start, _ in intervals]
    events += [(finish, -1) for _, finish in intervals]
    active = maximum = 0
    for _, change in sorted(events, key=lambda event: (event[0], -event[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def main() -> None:
    config = load_config("configs/base.yaml")
    # Make each real environment step long enough for reliable interval measurement.
    config.environment.physics_dt = 0.01
    count = config.training.parallel_envs
    indices = list(range(count))
    actions = [
        np.zeros((config.environment.agents_per_team, 2), dtype=np.float32)
        for _ in indices
    ]
    with ParallelUAVEnvs(
        count,
        config.environment,
        config.reward,
        config.render,
        Path.cwd(),
    ) as environments:
        environments.reset(indices, list(range(count)), [0] * count)
        results = environments.step(indices, actions, [None] * count)
        intervals = [
            (
                results[index][-1]["worker_step_started_ns"],
                results[index][-1]["worker_step_finished_ns"],
            )
            for index in indices
        ]
        wall_ms = (max(end for _, end in intervals) - min(start for start, _ in intervals)) / 1e6
        serial_ms = sum(end - start for start, end in intervals) / 1e6
        print(f"workers={count} unique_pids={len(set(environments.worker_pids))}")
        print(f"max_simultaneous_env_steps={maximum_overlap(intervals)}")
        print(f"batch_wall_ms={wall_ms:.2f} summed_worker_ms={serial_ms:.2f}")


if __name__ == "__main__":
    main()
