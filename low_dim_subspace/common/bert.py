from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray


@dataclass
class BertConfig:
    vocab_size: int = 28996
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


class BertEmbeddings(eqx.Module):
    word_embeddings: eqx.nn.Embedding
    position_embeddings: eqx.nn.Embedding
    token_type_embeddings: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        word_init_key, pos_init_key, type_init_key = jr.split(init_key, 3)
        self.word_embeddings = eqx.nn.Embedding(
            num_embeddings=bert_config.vocab_size,
            embedding_size=bert_config.hidden_size,
            key=word_init_key,
        )
        self.position_embeddings = eqx.nn.Embedding(
            num_embeddings=bert_config.max_position_embeddings,
            embedding_size=bert_config.hidden_size,
            key=pos_init_key,
        )
        self.token_type_embeddings = eqx.nn.Embedding(
            num_embeddings=bert_config.type_vocab_size,
            embedding_size=bert_config.hidden_size,
            key=type_init_key,
        )
        self.layernorm = eqx.nn.LayerNorm(
            shape=bert_config.hidden_size, eps=bert_config.layer_norm_eps
        )
        self.dropout = eqx.nn.Dropout(p=bert_config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids: Int[Array, "seq_len"],
        token_type_ids: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "seq_len hidden_size"]:
        position_ids = jnp.arange(len(input_ids))

        inputs_embeds = jax.vmap(self.word_embeddings)(input_ids)
        position_embeds = jax.vmap(self.position_embeddings)(position_ids)
        token_type_embeds = jax.vmap(self.token_type_embeddings)(token_type_ids)

        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        hidden_states = jax.vmap(self.layernorm)(hidden_states)
        hidden_states = self.dropout(
            hidden_states, inference=dropout_key is None, key=dropout_key
        )
        return hidden_states


class BertSelfAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        query_init_key, key_init_key, value_init_key = jr.split(init_key, 3)
        self.query = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features=bert_config.hidden_size,
            key=query_init_key,
        )
        self.key = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features=bert_config.hidden_size,
            key=key_init_key,
        )
        self.value = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features=bert_config.hidden_size,
            key=value_init_key,
        )
        self.dropout = eqx.nn.Dropout(p=bert_config.attention_probs_dropout_prob)
        self.num_heads = bert_config.num_attention_heads

    def transpose_for_scores(
        self, inputs: Float[Array, "seq_len hidden_size"]
    ) -> Float[Array, "num_heads seq_len head_size"]:
        return rearrange(
            inputs,
            "seq_len (num_heads head_size) -> num_heads seq_len head_size",
            num_heads=self.num_heads,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "seq_len hidden_size"]:
        query_ = self.transpose_for_scores(jax.vmap(self.query)(hidden_states))
        key_ = self.transpose_for_scores(jax.vmap(self.key)(hidden_states))
        value_ = self.transpose_for_scores(jax.vmap(self.value)(hidden_states))

        attention_scores = query_ @ jnp.matrix_transpose(key_)
        attention_scores = attention_scores / jnp.sqrt(key_.shape[-1])
        attention_scores = attention_scores + jnp.finfo(attention_scores.dtype).min * (
            1 - attention_mask
        )

        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(
            attention_probs, inference=dropout_key is None, key=dropout_key
        )

        context = attention_probs @ value_
        context = rearrange(
            context,
            "num_heads seq_len head_size -> seq_len (num_heads head_size)",
        )
        return context


class BertSelfOutput(eqx.Module):
    dense: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        self.dense = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features=bert_config.hidden_size,
            key=init_key,
        )
        self.layernorm = eqx.nn.LayerNorm(
            shape=bert_config.hidden_size, eps=bert_config.layer_norm_eps
        )
        self.dropout = eqx.nn.Dropout(p=bert_config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "hidden_size"],
        input_tensor: Float[Array, "hidden_size"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "hidden_size"]:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(
            hidden_states, inference=dropout_key is None, key=dropout_key
        )
        hidden_states = hidden_states + input_tensor
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class BertAttention(eqx.Module):
    self: BertSelfAttention
    output: BertSelfOutput

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        self_init_key, output_init_key = jr.split(init_key)
        self.self = BertSelfAttention(bert_config, self_init_key)
        self.output = BertSelfOutput(bert_config, output_init_key)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "seq_len hidden_size"]:
        self_key, output_key = (
            (None, None) if dropout_key is None else jr.split(dropout_key)
        )
        self_output = self.self(hidden_states, attention_mask, self_key)
        attention_output = eqx.filter_vmap(self.output)(
            self_output,
            hidden_states,
            (
                jr.split(output_key, hidden_states.shape[0])
                if output_key is not None
                else None
            ),
        )
        return attention_output


class BertIntermediate(eqx.Module):
    dense: eqx.nn.Linear

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        self.dense = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features=bert_config.intermediate_size,
            key=init_key,
        )

    def __call__(
        self, hidden_states: Float[Array, "hidden_size"]
    ) -> Float[Array, "intermediate_size"]:
        hidden_states = self.dense(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        return hidden_states


class BertOutput(eqx.Module):
    dense: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        self.dense = eqx.nn.Linear(
            in_features=bert_config.intermediate_size,
            out_features=bert_config.hidden_size,
            key=init_key,
        )
        self.layernorm = eqx.nn.LayerNorm(
            shape=bert_config.hidden_size, eps=bert_config.layer_norm_eps
        )
        self.dropout = eqx.nn.Dropout(p=bert_config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "intermediate_size"],
        input_tensor: Float[Array, "hidden_size"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "hidden_size"]:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(
            hidden_states, inference=dropout_key is None, key=dropout_key
        )
        hidden_states = hidden_states + input_tensor
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class BertLayer(eqx.Module):
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        attn_init_key, inter_init_key, output_init_key = jr.split(init_key, 3)
        self.attention = BertAttention(bert_config, attn_init_key)
        self.intermediate = BertIntermediate(bert_config, inter_init_key)
        self.output = BertOutput(bert_config, output_init_key)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "seq_len hidden_size"]:
        attn_key, out_key = (
            (None, None) if dropout_key is None else jr.split(dropout_key, 2)
        )
        attention_output = self.attention(hidden_states, attention_mask, attn_key)
        intermediate_output = jax.vmap(self.intermediate)(attention_output)
        layer_output = eqx.filter_vmap(self.output)(
            intermediate_output,
            attention_output,
            (
                jr.split(out_key, intermediate_output.shape[0])
                if out_key is not None
                else None
            ),
        )
        return layer_output


class BertEncoder(eqx.Module):
    layer: list[BertLayer]

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        layer_init_keys = jr.split(init_key, bert_config.num_hidden_layers)
        self.layer = [
            BertLayer(bert_config, layer_init_key) for layer_init_key in layer_init_keys
        ]

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "seq_len hidden_size"]:
        l_key = dropout_key
        for layer_module in self.layer:
            cl_key, l_key = (None, None) if l_key is None else jr.split(l_key)
            hidden_states = layer_module(hidden_states, attention_mask, cl_key)
        return hidden_states


class BertPooler(eqx.Module):
    dense: eqx.nn.Linear

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        self.dense = eqx.nn.Linear(
            in_features=bert_config.hidden_size,
            out_features=bert_config.hidden_size,
            key=init_key,
        )

    def __call__(
        self, hidden_states: Float[Array, "seq_len hidden_size"]
    ) -> Float[Array, "hidden_size"]:
        first_token_tensor = hidden_states[0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = jnp.tanh(pooled_output)
        return pooled_output


class BertModel(eqx.Module):
    embeddings: BertEmbeddings
    encoder: BertEncoder
    pooler: BertPooler

    def __init__(self, bert_config: BertConfig, init_key: PRNGKeyArray):
        emb_init_key, enc_init_key, pool_init_key = jr.split(init_key, 3)
        self.embeddings = BertEmbeddings(bert_config, emb_init_key)
        self.encoder = BertEncoder(bert_config, enc_init_key)
        self.pooler = BertPooler(bert_config, pool_init_key)

    def __call__(
        self,
        input_ids: Int[Array, "seq_len"],
        token_type_ids: Int[Array, "seq_len"],
        attention_mask: Int[Array, "seq_len"],
        dropout_key: PRNGKeyArray | None,
    ) -> Float[Array, "hidden_size"]:
        emb_key, enc_key = (
            (None, None) if dropout_key is None else jr.split(dropout_key, 2)
        )
        embedding_output = self.embeddings(input_ids, token_type_ids, emb_key)
        sequence_output = self.encoder(embedding_output, attention_mask, enc_key)
        pooled_output = self.pooler(sequence_output)
        return pooled_output
