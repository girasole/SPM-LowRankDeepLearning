import os
from collections.abc import Iterator

import jax.random as jr
from jaxtyping import PRNGKeyArray


def create_rng_stream(seed: int) -> Iterator[PRNGKeyArray]:
    key = jr.key(seed)
    while True:
        key, subkey = jr.split(key)
        yield subkey


def get_step_vals(experiment_name: str) -> list[int]:
    params_dir = os.path.join("checkpoints", experiment_name, "params")
    return sorted([int(f.split(".")[0]) for f in os.listdir(params_dir)])
