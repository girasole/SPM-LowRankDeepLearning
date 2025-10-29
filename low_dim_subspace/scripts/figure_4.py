import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from cycler import cycler
from jaxtyping import PyTree
from matplotlib.axes import Axes

from common.analyze import compute_svd_series
from common.data import create_bert_data_iterator
from common.models import (
    BertWithHead,
    LoraLinear,
    MatrixFactorization,
    get_lora_bert_model,
    get_lora_filter_spec,
)
from common.train import TrainState, train
from common.utils import create_rng_stream, get_step_vals

EXPERIMENT_NAME = "figure_4"
RNG_STREAM = create_rng_stream(0)


def get_params_static() -> tuple[PyTree, PyTree]:
    model = get_lora_bert_model(key=next(RNG_STREAM), lora_init_scale=1.0, lora_depth=3)
    filter_spec = get_lora_filter_spec(model)
    return eqx.partition(model, filter_spec)


def get_results():
    iterator = create_bert_data_iterator(key=next(RNG_STREAM), batch_size=16)
    params, static = get_params_static()
    init_state = TrainState(params=params, static=static, tx=optax.adam(1e-4, eps=1e-6))
    train(
        init_state,
        iterator,
        save_steps=jnp.arange(40 + 1, step=5),
        experiment_name=EXPERIMENT_NAME,
    )


def get_mf(params_: BertWithHead) -> MatrixFactorization:
    lora_linear: LoraLinear = params_.bert.encoder.layer[
        11
    ].attention.self.value  # pyright: ignore[reportAssignmentType]
    return lora_linear.mf


def plot_results():
    params, _ = get_params_static()
    save_steps = get_step_vals(EXPERIMENT_NAME)

    params_history: MatrixFactorization = jax.tree.map(
        lambda *ys: jnp.stack(ys),
        *(
            get_mf(
                eqx.tree_deserialise_leaves(
                    f"checkpoints/{EXPERIMENT_NAME}/params/{save_step}.eqx", params
                )
            )
            for save_step in save_steps
        ),
    )
    sval_series, right_series, left_series = compute_svd_series(
        jax.vmap(params_history.end_to_end.__func__)(params_history)
    )

    k = 32
    indices = np.r_[0:k, 768 - k : 768]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axes: list[Axes]
    series = [
        sval_series[:, indices],
        right_series[:, indices],
        left_series[:, indices],
    ]
    titles = ["Singular Values", "Right Singular Vectors", "Left Singular Vectors"]
    ylabels = [
        r"$\sigma_i(t)$",
        r"$\angle(v_i(t), v_i(0))$",
        r"$\angle(u_i(t), u_i(0))$",
    ]

    for i, ax in enumerate(axes):
        ax: Axes
        ax.set_prop_cycle(
            cycler("color", plt.get_cmap("viridis")(np.linspace(0, 1, len(save_steps))))
        )
        for j, t in enumerate(save_steps):
            ax.plot(series[i][j], "-", alpha=1.0, label=f"Iteration {t:2d}")
        ax.set_xlabel(r"SV Index ($i$)", fontsize=16)
        ax.set_ylabel(ylabels[i], fontsize=16)
        ax.set_title(titles[i], fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xticks(range(len(indices))[::6])
        ax.set_xticklabels(indices[::6].tolist())

    axes[0].text(31, 0.635, "//", fontsize=14)
    axes[1].text(31, -0.01, "//", fontsize=14)
    axes[2].text(31, -0.01, "//", fontsize=14)

    axes[1].legend(fontsize=12)
    fig.savefig("figure_4.pdf", bbox_inches="tight")


if __name__ == "__main__":
    get_results()
    plot_results()
