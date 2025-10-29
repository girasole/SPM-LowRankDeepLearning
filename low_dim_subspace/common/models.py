import os
import re
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from safetensors.numpy import load_file

from common.bert import BertConfig, BertModel


@dataclass
class SymmetricMFConfig:
    shape: tuple[int, ...]
    depth: int
    init_scale: float
    rank: int | None


@dataclass
class VanillaLoRAConfig:
    shape: tuple[int, ...]
    rank: int | None


class MatrixFactorization(eqx.Module):
    layers: list[Array]

    def __init__(
        self, mf_config: SymmetricMFConfig | VanillaLoRAConfig, key: PRNGKeyArray
    ):
        assert len(mf_config.shape) == 2
        width = mf_config.rank if mf_config.rank is not None else min(mf_config.shape)
        if isinstance(mf_config, SymmetricMFConfig):
            keys = jr.split(key, mf_config.depth)
            layers = [
                mf_config.init_scale
                * jr.orthogonal(keys[0], n=width, m=mf_config.shape[1])
            ]
            for i in range(1, mf_config.depth - 1):
                layers.append(mf_config.init_scale * jr.orthogonal(keys[i], n=width))
            layers.append(
                mf_config.init_scale
                * jr.orthogonal(keys[-1], n=mf_config.shape[0], m=width)
            )
            self.layers = layers
        elif isinstance(mf_config, VanillaLoRAConfig):
            self.layers = [
                jr.normal(key, (width, mf_config.shape[1])),
                jnp.zeros((mf_config.shape[0], width)),
            ]
        else:
            raise ValueError("Invalid mf_config type")

    def end_to_end(self) -> Float[Array, "out_size in_size"]:
        return jax.vmap(self)(jnp.eye(self.layers[0].shape[1]))

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        for layer in self.layers:
            x = layer @ x
        return x


class LoraLinear(eqx.Module):
    linear: eqx.nn.Linear
    mf: MatrixFactorization

    def __init__(
        self,
        mf_config: SymmetricMFConfig | VanillaLoRAConfig,
        linear: eqx.nn.Linear,
        key: PRNGKeyArray,
    ):
        self.linear = linear
        self.mf = MatrixFactorization(mf_config, key)

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        return self.linear(x) + self.mf(x)


class BertWithHead(eqx.Module):
    bert: BertModel
    head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        bert_init_key, head_init_key = jr.split(init_key)
        self.bert = BertModel(bert_config, bert_init_key)
        self.dropout = eqx.nn.Dropout(p=bert_config.hidden_dropout_prob)
        self.head = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features="scalar",
            key=head_init_key,
        )

    def __call__(
        self,
        input_ids: Int[Array, "seq_len"],
        token_type_ids: Int[Array, "seq_len"],
        attention_mask: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, ""]:
        bert_key, drop_key = (
            (None, None) if dropout_key is None else jr.split(dropout_key)
        )
        pooled_output = self.bert(input_ids, token_type_ids, attention_mask, bert_key)
        pooled_output = self.dropout(
            pooled_output, inference=dropout_key is None, key=drop_key
        )
        pred = self.head(pooled_output)
        return pred


def get_processed_model_dict() -> dict[str, np.ndarray]:

    def process_key(key: str) -> str:
        # remove bert prefix
        key = key.replace("bert", "")

        # replace layernorm fields
        key = key.replace("LayerNorm.gamma", "layernorm.weight")
        key = key.replace("LayerNorm.beta", "layernorm.bias")

        # replace .x with [x]
        key = re.sub(r"\.(\d+)\.", r"[\1].", key)

        return key

    if not os.path.exists("model.safetensors"):
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="google-bert/bert-base-cased",
            filename="model.safetensors",
            local_dir=".",
        )

    model_dict = load_file("model.safetensors")
    processed_model_dict = {
        process_key(k): v for k, v in model_dict.items() if "bert" in k
    }
    return processed_model_dict


def get_bert_model(pretrained: bool = False) -> BertWithHead:
    bert_config = BertConfig()
    model = BertWithHead(bert_config=bert_config, init_key=jr.key(0))
    if pretrained:
        model_dict = get_processed_model_dict()
        weights, rest = eqx.partition(
            model.bert, jtu.tree_map(eqx.is_array, model.bert)
        )
        weights = eqx.tree_at(
            lambda m: [leaf for leaf in jtu.tree_leaves(m) if leaf is not None],
            weights,
            [
                jnp.array(model_dict[jtu.keystr(path)])
                for path, leaf in jtu.tree_leaves_with_path(weights)
                if leaf is not None
            ],
        )
        model = eqx.tree_at(lambda m: m.bert, model, eqx.combine(weights, rest))

    return model


def get_lora_bert_model(
    key: PRNGKeyArray,
    lora_init_scale: float | None,
    lora_depth: int | None = 3,
    lora_rank: int | None = None,
    vanilla_lora: bool = False,
) -> BertWithHead:
    model = get_bert_model(pretrained=True)

    is_linear = lambda leaf: isinstance(leaf, eqx.nn.Linear)
    is_embedding = lambda leaf: isinstance(leaf, eqx.nn.Embedding)
    get_linear_layers = lambda m: [
        leaf
        for path, leaf in jtu.tree_leaves_with_path(m, is_leaf=is_linear)
        if (
            is_linear(leaf)
            and ("layer" in jtu.keystr(path) or "pooler" in jtu.keystr(path))
        )
        or is_embedding(leaf)
    ]
    linear_layers: list[eqx.nn.Linear] = get_linear_layers(model)

    def get_config(shape):
        if vanilla_lora:
            return VanillaLoRAConfig(shape=shape, rank=lora_rank)
        else:
            assert lora_init_scale is not None
            assert lora_depth is not None
            return SymmetricMFConfig(
                shape=shape,
                depth=lora_depth,
                init_scale=lora_init_scale,
                rank=lora_rank,
            )

    lora_linear_layers = [
        LoraLinear(
            get_config(leaf.weight.shape),
            leaf,
            key,
        )
        for leaf, key in zip(linear_layers, jr.split(key, len(linear_layers)))
    ]
    model = eqx.tree_at(
        get_linear_layers,
        model,
        lora_linear_layers,
    )
    return model


def get_lora_filter_spec(lora_model: BertWithHead) -> PyTree:

    def is_trainable(path: tuple) -> bool:
        path_str = jtu.keystr(path)
        return "mf" in path_str or "head" in path_str

    return jtu.tree_map_with_path(lambda path, _: is_trainable(path), lora_model)
