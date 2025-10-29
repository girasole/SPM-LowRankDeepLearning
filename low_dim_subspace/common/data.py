from collections.abc import Iterator

import datasets
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import transformers
from jaxtyping import Array, Float, Int, PRNGKeyArray

Data = eqx.Module

MAX_SEQ_LENGTH = 128


class MFData(Data):
    target: Float[Array, "num_rows num_cols"]


class BertData(eqx.Module):
    input_ids: Int[Array, "batch seq_len"]
    attention_mask: Int[Array, "batch seq_len"]
    token_type_ids: Int[Array, "batch seq_len"]
    labels: Float[Array, "batch"]


def create_mf_data_iterator(
    key: PRNGKeyArray, shape: tuple[int, int], rank: int
) -> Iterator[MFData]:
    target = jr.normal(key, shape)
    U, S, Vt = jnp.linalg.svd(target, full_matrices=False)
    target = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
    data = MFData(target=target)
    while True:
        yield data


def load_stsb_dataset() -> datasets.Dataset:
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    raw_datasets = datasets.load_dataset("nyu-mll/glue", "stsb")
    assert isinstance(raw_datasets, datasets.DatasetDict)
    raw_train_dataset: datasets.Dataset = raw_datasets["train"]

    def length_of(example) -> int:
        texts = (example["sentence1"], example["sentence2"])
        return len(tokenizer(*texts)["input_ids"])

    filtered_raw_train_dataset = raw_train_dataset.filter(
        lambda example: length_of(example) <= MAX_SEQ_LENGTH
    )

    def preprocess_fn(example):
        texts = (example["sentence1"], example["sentence2"])
        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="np",
        )
        result["labels"] = example["label"]
        return result

    train_dataset = filtered_raw_train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=filtered_raw_train_dataset.column_names,
    )

    return train_dataset


def create_bert_data_iterator(key: PRNGKeyArray, batch_size: int) -> Iterator[BertData]:
    dataset = load_stsb_dataset()
    num_examples = len(dataset)
    steps_per_epoch = num_examples // batch_size

    while True:
        key, subkey = jr.split(key)
        perms = jr.permutation(subkey, num_examples)[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        for perm in perms:
            batch = dataset[perm]
            data = BertData(
                input_ids=jnp.array(batch["input_ids"]),
                attention_mask=jnp.array(batch["attention_mask"]),
                token_type_ids=jnp.array(batch["token_type_ids"]),
                labels=jnp.array(batch["labels"], dtype=jnp.float32),
            )
            yield data
