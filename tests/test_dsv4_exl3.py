"""DeepSeek-V4-Flash from a tiny EXL3 checkpoint: refusals, loading, both paths and exact windows (CPU reference path)."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dsv4_exl3_fakes import QUANT, TEXT, write_checkpoint  # noqa: E402
from tensorfold import families  # noqa: E402
from tensorfold.cuda.exl3 import format as fmt  # noqa: E402
from tensorfold.families.deepseek_v4 import weights  # noqa: E402
from tensorfold.kernels.exl3.v1.linear import Exl3Experts, Exl3Grouped, Exl3Linear  # noqa: E402

CPU_ROWS = 7          # MLX's CPU fp32 rms_norm is row-invariant below 8 rows


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("dsv4-exl3"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def model(checkpoint):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return weights.load_backbone(checkpoint)
    finally:
        mx.set_default_device(previous)


def tokens(n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(2, TEXT["vocab_size"], size=n)]


def logits_of(model, ids, cache):
    return model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))[0]


def test_family_reads_exl3_on_the_mac_backend(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    family = families.detect(checkpoint)
    families.require_readable(family, config, "mlx")
    family.package.check(checkpoint)
    scan = fmt.scan(checkpoint)
    assert not scan.bad and {g.codebook for g in scan.groups.values()} == {"mul1"}


def test_a_codebook_the_format_does_not_know_is_refused_before_download(checkpoint, tmp_path):
    config = json.loads((checkpoint / "config.json").read_text())
    config["quantization_config"] = {**QUANT, "codebook": "lut9"}
    with pytest.raises(ValueError, match="EXL3 module"):
        families.require_readable(families.detect(checkpoint), config, "mlx")


def test_only_deepseek_reads_exl3_on_the_mac():
    readers = {kind for kind, family in families.families().items()
               if families.EXL3_QUANT in families.readable_quants(family, "mlx")}
    assert readers == {"deepseek_v4"}


def test_layers_load_as_exl3(model):
    layer = model.layers[1]
    assert isinstance(model.lm_head, Exl3Linear) and model.embed.bits is None
    assert isinstance(layer.attn.wo_a_grouped, Exl3Grouped) and layer.attn.wo_a_all is None
    assert all(isinstance(m, Exl3Experts) for m in (layer.moe.gate, layer.moe.up, layer.moe.down))
    assert set(layer.moe.gate.k2_of) == {4, 5, 6}            # 2, 2.5 and 3 bits in one stack
    assert [layer.attn.ratio for layer in model.layers] == TEXT["compress_ratios"][:TEXT["num_hidden_layers"]]
    assert [layer.attn.indexer is not None for layer in model.layers] == [r == 4 for r in TEXT["compress_ratios"][:5]]
    assert model.layers[0].moe.table is not None and model.layers[1].moe.table is None


def test_linear_matches_the_float64_reference(model, checkpoint):
    w = weights.Weights(checkpoint)
    t, suh, svh = (np.array(w.get(f"layers.2.attn.wq_b.{n}")) for n in ("trellis", "suh", "svh"))
    x = np.random.default_rng(2).standard_normal((3, TEXT["q_lora_rank"])).astype(np.float32)
    got = np.array(model.layers[2].attn.wq_b(mx.array(x)))
    want = fmt.forward(x, t, suh, svh, 4, "mul1")
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-3


@pytest.mark.parametrize("length", [6, 40, 300])
def test_prefill_path_agrees_with_decode_path(model, length):
    ids = tokens(length)
    whole = model.make_cache()
    a = logits_of(model, ids, whole)[-1]
    step = model.make_cache()
    for t in ids:
        b = logits_of(model, [t], step)[-1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 0.05 * np.max(np.abs(b)) + 0.05


@pytest.mark.parametrize("prompt", [5, 38, 124])
def test_decode_windows_give_one_row_bits(model, prompt):
    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(prompt, seed=5)], dtype=mx.uint32), base))
    window = tokens(CPU_ROWS, seed=6)
    cache = LaneEngine.copy_single_cache(base)
    serial = [logits_of(model, [t], cache)[-1] for t in window]
    joint = logits_of(model, window, LaneEngine.copy_single_cache(base))
    for i in range(CPU_ROWS):
        assert mx.array_equal(joint[i], serial[i]).item(), f"row {i}"


def test_lane_engine_checks_exact_windows_and_generates(model):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v4 import engine_settings
    from tensorfold.families.deepseek_v4.runtime import DeepSeekFlash

    runtime = DeepSeekFlash(model, None, drafts=0, check=False)
    width, _ = runtime.check_windows(widest=CPU_ROWS)
    assert width == CPU_ROWS
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    engine = LaneEngine(runtime, **engine_settings(runtime))
    stream = LaneStream(stream_id="s", prompt_ids=tokens(33, seed=4), max_new_tokens=12, drafts=False)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert len(stream.emitted) == 12
