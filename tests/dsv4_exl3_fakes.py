"""A tiny random DeepSeek-V4-Flash checkpoint in ExLlamaV3's EXL3 layout (DeepSeek's tensor names, trellis linears).

Every quantized dimension is a multiple of 128 (the Hadamard blocks), routed experts mix 2-, 2.5- and 3-bit tiles
and the rest is 4-bit, all mul1, as turboderp's DeepSeek-V4-Flash packs are.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

D, VOCAB = 128, 256
TEXT = {
    "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"], "hidden_size": D,
    "num_hidden_layers": 5, "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "num_attention_heads": 4,
    "num_key_value_heads": 1, "head_dim": 128, "qk_rope_head_dim": 32, "q_lora_rank": 128, "o_lora_rank": 128,
    "o_groups": 2, "sliding_window": 8, "compress_ratios": [0, 4, 128, 4, 128, 0], "rope_theta": 10000,
    "compress_rope_theta": 160000,
    "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32,
                     "beta_slow": 1},
    "index_n_heads": 2, "index_head_dim": 128, "index_topk": 4, "n_routed_experts": 8, "n_shared_experts": 1,
    "num_experts_per_tok": 2, "moe_intermediate_size": 128, "num_hash_layers": 1, "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
    "num_nextn_predict_layers": 1, "eos_token_id": 1, "bos_token_id": 0,
}
QUANT = {"quant_method": "exl3", "version": "1.3.0", "bits": 2.6, "head_bits": 4, "codebook": "mul1",
         "out_scales": "always"}
MUL1 = np.array(0x83DCD12D, dtype=np.uint32).view(np.int32)
EXPERT_BITS = (2, 3, 2.5)


def _exl3(t: dict, rng: np.random.Generator, name: str, outs: int, ins: int, bits: float = 4, scale: float = 0.08) -> None:
    """An EXL3 group for a [outs, ins] linear: trellis [ins/16, outs/16, 16 * bits], sign suh, constant svh."""

    words = int(16 * bits)
    t[f"{name}.trellis"] = mx.array(rng.integers(-32768, 32768, size=(ins // 16, outs // 16, words)).astype(np.int16))
    t[f"{name}.suh"] = mx.array(np.where(rng.random(ins) < 0.5, -1.0, 1.0).astype(np.float16))
    t[f"{name}.svh"] = mx.array((scale * (1.0 + 0.1 * rng.standard_normal(outs))).astype(np.float16))
    t[f"{name}.mul1"] = mx.array(MUL1)


def _norm(t: dict, rng: np.random.Generator, name: str, n: int) -> None:
    t[name] = mx.array((1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)).astype(mx.bfloat16)


def _f32(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> mx.array:
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32))


def _hc(t: dict, rng: np.random.Generator, name: str, mixes: int) -> None:
    t[f"{name}_fn"] = _f32(rng, (mixes, 4 * D), 0.02)
    t[f"{name}_base"] = _f32(rng, (mixes,), 0.1)
    t[f"{name}_scale"] = 1.0 + _f32(rng, (3 if mixes > 4 else 1,), 0.1)


def _compressor(t: dict, rng: np.random.Generator, p: str, ratio: int, dim: int) -> None:
    width = dim * (2 if ratio == 4 else 1)
    _exl3(t, rng, f"{p}.wkv", width, D, scale=0.3)
    _exl3(t, rng, f"{p}.wgate", width, D, scale=0.3)
    t[f"{p}.ape"] = _f32(rng, (ratio, width), 0.1)
    _norm(t, rng, f"{p}.norm.weight", dim)


def _block(t: dict, rng: np.random.Generator, i: int) -> None:
    c, p = TEXT, f"layers.{i}"
    ratio, hashed = c["compress_ratios"][i], i < c["num_hash_layers"]
    h, hd, g = c["num_attention_heads"], c["head_dim"], c["o_groups"]
    _hc(t, rng, f"{p}.hc_attn", 24)
    _hc(t, rng, f"{p}.hc_ffn", 24)
    _norm(t, rng, f"{p}.attn_norm.weight", D)
    _norm(t, rng, f"{p}.ffn_norm.weight", D)
    a = f"{p}.attn"
    _exl3(t, rng, f"{a}.wq_a", c["q_lora_rank"], D)
    _norm(t, rng, f"{a}.q_norm.weight", c["q_lora_rank"])
    _exl3(t, rng, f"{a}.wq_b", h * hd, c["q_lora_rank"])
    _exl3(t, rng, f"{a}.wkv", hd, D)
    _norm(t, rng, f"{a}.kv_norm.weight", hd)
    for s in range(g):
        _exl3(t, rng, f"{a}.wo_a.slice.{s}", c["o_lora_rank"], h * hd // g)
    _exl3(t, rng, f"{a}.wo_b", D, g * c["o_lora_rank"])
    t[f"{a}.attn_sink"] = _f32(rng, (h,), 0.5)
    if ratio:
        _compressor(t, rng, f"{a}.compressor", ratio, hd)
    if ratio == 4:
        _exl3(t, rng, f"{a}.indexer.wq_b", c["index_n_heads"] * c["index_head_dim"], c["q_lora_rank"], scale=0.3)
        t[f"{a}.indexer.weights_proj.weight"] = mx.array((0.3 * rng.standard_normal((c["index_n_heads"], D)))
                                                         .astype(np.float16))
        _compressor(t, rng, f"{a}.indexer.compressor", ratio, c["index_head_dim"])
    f = f"{p}.ffn"
    e, inter = c["n_routed_experts"], c["moe_intermediate_size"]
    t[f"{f}.gate.weight"] = mx.array((0.3 * rng.standard_normal((e, D))).astype(np.float16))
    if hashed:
        t[f"{f}.gate.tid2eid"] = mx.array([[(v + k) % e for k in range(c["num_experts_per_tok"])]
                                           for v in range(VOCAB)], dtype=mx.int64)
    else:
        t[f"{f}.gate.bias"] = mx.array((0.1 * rng.standard_normal(e)).astype(np.float16))
    _exl3(t, rng, f"{f}.shared_experts.w1", inter, D)
    _exl3(t, rng, f"{f}.shared_experts.w3", inter, D)
    _exl3(t, rng, f"{f}.shared_experts.w2", D, inter)
    for x in range(e):
        bits = EXPERT_BITS[(x + i) % len(EXPERT_BITS)]
        _exl3(t, rng, f"{f}.experts.{x}.w1", inter, D, bits)
        _exl3(t, rng, f"{f}.experts.{x}.w3", inter, D, bits)
        _exl3(t, rng, f"{f}.experts.{x}.w2", D, inter, bits)


def write_checkpoint(folder: Path, seed: int = 0, quant: dict | None = None) -> Path:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    t: dict = {}
    t["embed.weight"] = mx.array(rng.standard_normal((VOCAB, D)).astype(np.float32)).astype(mx.bfloat16)
    _exl3(t, rng, "head", VOCAB, D)
    _norm(t, rng, "norm.weight", D)
    _hc(t, rng, "hc_head", 4)
    for i in range(TEXT["num_hidden_layers"]):
        _block(t, rng, i)
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(folder / shard), {k: t[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({**TEXT, "quantization_config": quant or QUANT}))
    return folder
