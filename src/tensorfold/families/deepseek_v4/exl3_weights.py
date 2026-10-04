"""The backbone from an EXL3 checkpoint (ExLlamaV3's DeepSeek-V4 layout): DeepSeek's own tensor names, trellis linears.

turboderp's DeepSeek-V4-Flash EXL3 conversions keep DeepSeek's names (``layers.N.attn.wq_a``, ``ffn.experts.E.w1``,
``hc_attn_fn``, ``embed``, ``head``), store every linear as an EXL3 group (one codebook, widths per tensor: routed
experts at 2-3 bits, the rest wider) and leave the router, the indexer's head weights, the hyper-connections, norms
and the embedding unquantized. The engine's modules take ``Exl3Linear``/``Exl3Experts`` where the MLX layout has
``Q``/``FP4``; everything downstream of loading is the same code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4.attention import Attention
from tensorfold.families.deepseek_v4.compressor import Compressor, Indexer
from tensorfold.families.deepseek_v4.config import Config
from tensorfold.families.deepseek_v4.model import Block, DeepSeekV4, HeadHC
from tensorfold.families.deepseek_v4.moe import MoE, Shared
from tensorfold.families.deepseek_v4.weights import Weights, block_arrays
from tensorfold.families.glm5_next.linear import Dense
from tensorfold.families.glm5_next.model import HC
from tensorfold.kernels.exl3.v1.linear import Exl3Experts, Exl3Grouped, Exl3Linear


class Plain(Dense):
    """An unquantized fp16/fp32 linear [out, in] (the indexer's head weights): fp32 math, the input's dtype out."""

    def __init__(self, weight: mx.array) -> None:
        super().__init__(weight.astype(mx.float32))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.matmul(x.astype(mx.float32), self.weight.T).astype(x.dtype)


class Table:
    """The token embedding as stored (bf16 rows); ``DeepSeekV4.embed_tokens`` takes rows of ``weight``."""

    bits = None
    group = None

    def __init__(self, weight: mx.array) -> None:
        self.weight = weight

    def arrays(self) -> list[mx.array]:
        return [self.weight]


def is_exl3(config: dict[str, Any]) -> bool:
    from tensorfold.families import quant_method

    return quant_method(config) == "exl3"


class Exl3Weights(Weights):
    def lin(self, prefix: str) -> Exl3Linear:
        return Exl3Linear.load(self.get, self.has, prefix)


def load_attention(w: Exl3Weights, p: str, cfg: Config, layer: int) -> Attention:
    a = f"{p}.attn"
    parts: dict[str, Any] = {n: w.lin(f"{a}.{n}") for n in ("wq_a", "wq_b", "wkv", "wo_b")}
    parts["wo_a"] = Exl3Grouped([w.lin(f"{a}.wo_a.slice.{g}") for g in range(cfg.o_groups)])
    parts.update(q_norm=w.get(f"{a}.q_norm.weight"), kv_norm=w.get(f"{a}.kv_norm.weight"),
                 attn_sink=w.get(f"{a}.attn_sink"))
    ratio = cfg.ratio(layer)
    freqs = cfg.inv_freq(layer)
    if ratio:
        parts["compressor"] = compressor(w, f"{a}.compressor", ratio, cfg, freqs)
    if ratio == 4:
        parts["indexer"] = Indexer(w.lin(f"{a}.indexer.wq_b"), Plain(w.get(f"{a}.indexer.weights_proj.weight")),
                                   compressor(w, f"{a}.indexer.compressor", ratio, cfg, freqs), cfg.index_n_heads,
                                   cfg.index_head_dim, cfg.index_topk, freqs)
    return Attention(parts, cfg, layer)


def compressor(w: Exl3Weights, p: str, ratio: int, cfg: Config, freqs: mx.array) -> Compressor:
    return Compressor(w.lin(f"{p}.wkv"), w.lin(f"{p}.wgate"), w.get(f"{p}.ape"), w.get(f"{p}.norm.weight"), ratio,
                      cfg.rms_norm_eps, freqs)


def load_moe(w: Exl3Weights, p: str, cfg: Config, layer: int) -> MoE:
    f = f"{p}.ffn"
    hashed = layer < cfg.num_hash_layers
    shared = Shared(w.lin(f"{f}.shared_experts.w1"), w.lin(f"{f}.shared_experts.w3"),
                    w.lin(f"{f}.shared_experts.w2"), cfg.swiglu_limit)
    experts = [Exl3Experts([w.lin(f"{f}.experts.{e}.{m}") for e in range(cfg.n_routed_experts)])
               for m in ("w1", "w3", "w2")]
    table = w.get(f"{f}.gate.tid2eid") if hashed else None
    if table is not None:
        table = table.astype(mx.int32)
        top = mx.sort(table, axis=-1)
        if bool(mx.any(top[:, 1:] == top[:, :-1]).item()):
            raise ValueError(f"{f}.gate.tid2eid: a token picks one expert twice; the EXL3 experts group a window's "
                             f"rows by expert, at most once a row")
    bias = None if hashed else w.get(f"{f}.gate.bias")
    return MoE(w.get(f"{f}.gate.weight"), bias, table, *experts, shared, cfg)


def load_block(w: Exl3Weights, layer: int, cfg: Config) -> Block:
    p = f"layers.{layer}"
    block = Block(load_attention(w, p, cfg, layer), load_moe(w, p, cfg, layer), w.get(f"{p}.attn_norm.weight"),
                  w.get(f"{p}.ffn_norm.weight"),
                  HC(w.get(f"{p}.hc_attn_fn"), w.get(f"{p}.hc_attn_base"), w.get(f"{p}.hc_attn_scale"), cfg),
                  HC(w.get(f"{p}.hc_ffn_fn"), w.get(f"{p}.hc_ffn_base"), w.get(f"{p}.hc_ffn_scale"), cfg),
                  cfg.rms_norm_eps)
    mx.eval(*block_arrays(block))
    return block


def load_backbone(model_dir: Path, layers: int | None = None) -> DeepSeekV4:
    """The backbone from an EXL3 folder (``layers``: the first few only, for probes and tests)."""

    raw = json.loads((model_dir / "config.json").read_text())
    cfg = Config.from_dict(raw)
    w = Exl3Weights(model_dir)
    count = cfg.num_hidden_layers if layers is None else int(layers)
    blocks = [load_block(w, i, cfg) for i in range(count)]
    head = HeadHC(w.get("hc_head_fn"), w.get("hc_head_base"), w.get("hc_head_scale"), cfg.rms_norm_eps, cfg.hc_eps)
    model = DeepSeekV4(cfg, Table(w.get("embed.weight")), blocks, head, w.get("norm.weight"), w.lin("head"))
    mx.eval(*model.embed.arrays(), *model.lm_head.arrays(), model.norm, head.fn, head.base, head.scale)
    return model


__all__ = ["Exl3Weights", "Plain", "Table", "is_exl3", "load_backbone"]
