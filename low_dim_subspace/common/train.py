import os
from collections.abc import Callable, Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Int, PRNGKeyArray, PyTree, Scalar
from tensorboardX import SummaryWriter
from tqdm import tqdm

from common.data import BertData, Data, MFData
from common.models import BertWithHead, MatrixFactorization
from common.utils import create_rng_stream


class TrainState(eqx.Module):
    params: PyTree
    static: PyTree
    tx: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState

    def __init__(
        self,
        params: PyTree,
        static: PyTree,
        tx: optax.GradientTransformation,
    ):
        self.params = params
        self.static = static
        self.tx = tx
        self.opt_state = tx.init(params)

    def update(self, grads: PyTree) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = eqx.apply_updates(self.params, updates)
        return eqx.tree_at(
            lambda s: (s.params, s.opt_state), self, (new_params, new_opt_state)
        )


def loss_fn(
    params: PyTree, static: PyTree, data: Data, dropout_key: PRNGKeyArray
) -> Scalar:
    model = eqx.combine(params, static)

    if isinstance(data, MFData):
        assert isinstance(model, MatrixFactorization)
        return optax.l2_loss(model.end_to_end(), data.target).mean()

    if isinstance(data, BertData):
        assert isinstance(model, BertWithHead)
        preds = jax.vmap(model)(
            data.input_ids,
            data.token_type_ids,
            data.attention_mask,
            jr.split(dropout_key, data.input_ids.shape[0]),
        )
        return optax.l2_loss(preds, data.labels).mean()

    raise ValueError(f"Unknown data type: {type(data)}")


def create_train_step() -> (
    Callable[[TrainState, Data, PRNGKeyArray], tuple[TrainState, Scalar]]
):
    def train_step(
        state: TrainState, data: Data, dropout_key: PRNGKeyArray
    ) -> tuple[TrainState, Scalar]:
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, state.static, data, dropout_key
        )
        return state.update(grads), loss

    return jax.jit(train_step)


def train(
    init_state: TrainState,
    iterator: Iterator[Data],
    save_steps: Int[Array, "num_save_steps"],
    experiment_name: str,
):
    num_steps = jnp.max(save_steps).item() + 1
    rng_stream = create_rng_stream(0)
    state = init_state
    train_step = create_train_step()
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    params_dir = os.path.join(checkpoint_dir, "params")

    if os.path.exists(checkpoint_dir):
        return

    writer = SummaryWriter(checkpoint_dir)
    os.makedirs(params_dir)

    for step_num, data in tqdm(zip(range(num_steps), iterator), total=num_steps):
        dropout_key = next(rng_stream)
        state, loss = train_step(state, data, dropout_key)
        writer.add_scalar("loss", loss.item(), step_num)

        if step_num in save_steps:
            eqx.tree_serialise_leaves(
                os.path.join(params_dir, f"{step_num}.eqx"), state.params
            )
