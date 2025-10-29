import os
import pickle as pl

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from cycler import cycler
from jaxtyping import Array, PyTree
from matplotlib.axes import Axes
from tqdm import tqdm

from common.analyze import compute_cosine_angle, compute_matrix_rank, svdvals
from common.data import create_bert_data_iterator
from common.models import (
    BertWithHead,
    MatrixFactorization,
    get_lora_bert_model,
    get_lora_filter_spec,
)
from common.train import TrainState, train
from common.utils import create_rng_stream, get_step_vals

RNG_STREAM = create_rng_stream(0)


def get_params_static(
    depth: int, rank: int | None, vanilla_lora: bool
) -> tuple[PyTree, PyTree]:
    model = get_lora_bert_model(
        key=next(RNG_STREAM),
        lora_init_scale=1e-3,
        lora_depth=depth,
        lora_rank=rank,
        vanilla_lora=vanilla_lora,
    )
    filter_spec = get_lora_filter_spec(model)
    return eqx.partition(model, filter_spec)


def get_results():

    # figure 6ab
    iterator = create_bert_data_iterator(key=next(RNG_STREAM), batch_size=16)
    params, static = get_params_static(3, None, False)
    lr_schedule = optax.linear_schedule(
        init_value=1e-4, end_value=1e-5, transition_steps=2000
    )
    init_state = TrainState(
        params=params, static=static, tx=optax.adam(lr_schedule, eps=1e-6)
    )
    train(
        init_state,
        iterator,
        save_steps=jnp.array(
            [
                0,
                1,
                10,
                20,
                30,
                100,
                500,
                1000,
                1500,
                2000,
            ]
        ),
        experiment_name="figure_6ab",
    )

    # figure 6c
    params, static = get_params_static(2, 8, True)
    init_state = TrainState(params=params, static=static, tx=optax.adam(1e-4, eps=1e-6))
    train(
        init_state,
        iterator,
        save_steps=jnp.array([500]),
        experiment_name="figure_6c_vanilla",
    )

    params, static = get_params_static(3, None, False)
    init_state = TrainState(params=params, static=static, tx=optax.adam(1e-4, eps=1e-6))
    train(
        init_state,
        iterator,
        save_steps=jnp.array([500]),
        experiment_name="figure_6c_deep",
    )


def get_list_of_end_to_end(
    step: int, experiment_name: str, params: PyTree
) -> list[Array]:
    params_: BertWithHead = eqx.tree_deserialise_leaves(
        f"checkpoints/{experiment_name}/params/{step}.eqx", params
    )
    return [
        x.end_to_end()
        for x in jtu.tree_leaves(
            params_,
            is_leaf=lambda x: isinstance(x, MatrixFactorization),
        )
        if isinstance(x, MatrixFactorization)
    ]


def plot_results():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axes: list[Axes]

    # Figure 6a
    save_steps = get_step_vals("figure_6ab")
    params, _ = get_params_static(3, None, False)
    end_to_end_list = get_list_of_end_to_end(save_steps[-1], "figure_6ab", params)
    sval_list = [svdvals(np.array(x)) for x in end_to_end_list]

    axes[0].set_prop_cycle(
        cycler("color", plt.get_cmap("viridis")(np.linspace(0, 1, len(sval_list))))
    )
    axes[0].set_xlabel("SV Index", fontsize=16)
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    for sval in sval_list:
        axes[0].semilogy(sval, linewidth=2, alpha=0.8)

    # Figure 6b
    if not os.path.exists("deep_lora_subspaces.pkl"):
        left_series = jtu.tree_map(
            np.array,
            jtu.tree_map(
                lambda *ys: jnp.stack(ys),
                *(
                    [
                        jnp.linalg.svd(x)[0][:, :8]
                        for x in get_list_of_end_to_end(step, "figure_6ab", params)
                    ]
                    for step in tqdm(save_steps)
                ),
            ),
        )
        with open("deep_lora_subspaces.pkl", "wb") as f:
            pl.dump(left_series, f)

    with open("deep_lora_subspaces.pkl", "rb") as f:
        left_series = pl.load(f)

    angles_series = []

    for U_vals in left_series:
        U_final = U_vals[-1]
        angles = [compute_cosine_angle(U, U_final) for U in U_vals]
        if angles[2] < 0.6:
            continue
        angles_series.append(angles)

    axes[1].set_prop_cycle(
        cycler("color", plt.get_cmap("viridis")(np.linspace(0, 1, len(angles_series))))
    )
    for angles in angles_series:
        axes[1].plot(save_steps, angles, linewidth=2, alpha=0.8)
    axes[1].set_xlabel("Iteration", fontsize=16)
    axes[1].set_ylabel("Cosine Angle", fontsize=16)
    axes[1].tick_params(axis="both", which="major", labelsize=14)

    # Figure 6c
    params2, _ = get_params_static(2, 8, True)
    end_to_end_list2 = get_list_of_end_to_end(500, "figure_6c_vanilla", params2)
    rank2 = [compute_matrix_rank(np.array(x), 8) for x in end_to_end_list2]

    params3, _ = get_params_static(3, None, False)
    end_to_end_list3 = get_list_of_end_to_end(500, "figure_6c_deep", params3)
    rank3 = [compute_matrix_rank(np.array(x), 8) for x in end_to_end_list3]

    sns.histplot(rank2, ax=axes[2], color="orange", discrete=True)
    sns.histplot(rank3, ax=axes[2], color="blue", discrete=True)
    axes[2].set_xlabel("Rank", fontsize=16)
    axes[2].set_ylabel("Count", fontsize=16)
    axes[2].legend(["Vanilla LoRA", "Deep LoRA"], fontsize=12)

    fig.savefig("figure_6.pdf", bbox_inches="tight")


if __name__ == "__main__":
    get_results()
    plot_results()
