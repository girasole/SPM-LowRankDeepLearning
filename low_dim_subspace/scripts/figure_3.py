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
from common.data import create_mf_data_iterator
from common.models import MatrixFactorization, SymmetricMFConfig
from common.train import TrainState, train
from common.utils import create_rng_stream, get_step_vals

jax.config.update("jax_enable_x64", True)

EXPERIMENT_NAME = "figure_3"
RNG_STREAM = create_rng_stream(0)


def get_params_static() -> tuple[PyTree, PyTree]:
    mf_config = SymmetricMFConfig(shape=(30, 30), depth=3, init_scale=1.0, rank=None)
    model = MatrixFactorization(mf_config, key=next(RNG_STREAM))
    return eqx.partition(model, eqx.is_inexact_array)


def get_results():
    iterator = create_mf_data_iterator(key=next(RNG_STREAM), shape=(30, 30), rank=3)
    params, static = get_params_static()
    init_state = TrainState(params=params, static=static, tx=optax.sgd(1.0))
    train(
        init_state,
        iterator,
        save_steps=jnp.arange(30 + 1, step=3),
        experiment_name=EXPERIMENT_NAME,
    )


def plot_results():
    params, _ = get_params_static()
    save_steps = get_step_vals(EXPERIMENT_NAME)
    params_history = jax.tree.map(
        lambda *ys: jnp.stack(ys),
        *(
            eqx.tree_deserialise_leaves(
                f"checkpoints/{EXPERIMENT_NAME}/params/{save_step}.eqx", params
            )
            for save_step in save_steps
        ),
    )
    sval_series, right_series, left_series = compute_svd_series(
        params_history.layers[0]
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axes: list[Axes]
    series = [sval_series, right_series, left_series]
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

    axes[1].legend(fontsize=12)
    fig.savefig("figure_3.pdf", bbox_inches="tight")


if __name__ == "__main__":
    get_results()
    plot_results()
