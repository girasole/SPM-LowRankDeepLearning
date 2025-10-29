import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu

from common.models import BertConfig, BertWithHead

model = BertWithHead(BertConfig(), init_key=jr.key(0))
jtu.tree_map_with_path(
    lambda path, x: print(jtu.keystr(path), x.shape if eqx.is_array(x) else None),
    model,
)
