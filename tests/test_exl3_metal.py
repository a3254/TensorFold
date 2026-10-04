"""The Metal EXL3 kernels on an Apple GPU: decoded weights bit for bit, every row its one-row bits, the float64 layer."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs an Apple GPU (Metal)")

from tensorfold.cuda.exl3 import format as fmt  # noqa: E402
from tensorfold.kernels.exl3.v1 import trellis as T  # noqa: E402
from tensorfold.kernels.exl3.v1.linear import Exl3Experts, Exl3Grouped, Exl3Linear  # noqa: E402

WIDTHS = [(k2, cb) for k2 in T.K2S for cb in fmt.CODEBOOKS if k2 / 2 in fmt.BITS and (k2 % 2 == 0 or cb == "mul1")]


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _bits(k2: int) -> float:
    return k2 // 2 if k2 % 2 == 0 else k2 / 2


def _group(rng: np.random.Generator, k: int, n: int, k2: int, scale: float = 0.05):
    t = rng.integers(-32768, 32768, size=(k // 16, n // 16, 8 * k2)).astype(np.int16)
    suh = np.where(rng.random(k) < 0.5, -1.0, 1.0).astype(np.float16)
    svh = (scale * (1 + 0.1 * rng.standard_normal(n))).astype(np.float16)
    return t, suh, svh


def _linear(t, suh, svh, cb, bias=None) -> Exl3Linear:
    return Exl3Linear(mx.array(t), mx.array(suh), mx.array(svh), None if bias is None else mx.array(bias), codebook=cb)


def _same(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a.view(mx.uint16) if a.dtype != mx.float32 else a.view(mx.uint32),
                               b.view(mx.uint16) if b.dtype != mx.float32 else b.view(mx.uint32)).item())


@pytest.mark.parametrize("k2,codebook", WIDTHS)
def test_kernel_decodes_every_weight_bit_for_bit(k2, codebook):
    """One-hot rotated rows through mm give W_q's rows exactly (one nonzero product each)."""

    k, n = 256, 384
    t, suh, svh = _group(np.random.default_rng(k2), k, n, k2)
    lin = _linear(t, suh, svh, codebook)
    xh = mx.eye(k, dtype=mx.float16)
    z = T.mm(xh, lin.words, lin.tb, lin.tk, (k2,), T.dense_group(k), k=k, n=n, sk=lin.sk,
             codebook=T.CODEBOOK_IDS[codebook], units=1, members=k)
    got = np.array(z.sum(axis=0)).astype(np.float16)
    assert np.array_equal(got.view(np.uint16), fmt.unpack(t, _bits(k2), codebook).view(np.uint16))


@pytest.mark.parametrize("k2,codebook", [(4, "mul1"), (5, "mul1"), (6, "mul1"), (8, "mcg"), (12, "mul1"),
                                         (16, "3inst")])
@pytest.mark.parametrize("shape", [(256, 384), (4096, 1024), (1024, 4096), (8192, 512)])
def test_layer_matches_the_float64_reference(k2, codebook, shape):
    k, n = shape
    rng = np.random.default_rng(k + n + k2)
    t, suh, svh = _group(rng, k, n, k2)
    bias = (0.1 * rng.standard_normal(n)).astype(np.float16)
    lin = _linear(t, suh, svh, codebook, bias)
    x = rng.standard_normal((5, k)).astype(np.float32)
    want = fmt.forward(x, t, suh, svh, _bits(k2), codebook, bias)
    for dtype in (mx.float32, mx.bfloat16, mx.float16):
        got = np.array(lin(mx.array(x).astype(dtype)).astype(mx.float32))
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < (2e-3 if dtype == mx.float32 else 8e-3), dtype


@pytest.mark.parametrize("shape", [(256, 128), (4096, 1024), (1024, 32768), (8192, 4096)])
def test_rows_never_depend_on_the_rows_beside_them(shape):
    k, n = shape
    rng = np.random.default_rng(7)
    lin = _linear(*_group(rng, k, n, 5), "mul1")
    x = mx.array(rng.standard_normal((40, k)).astype(np.float32)).astype(mx.bfloat16)
    one = [lin(x[r:r + 1]) for r in range(40)]
    for count in (2, 3, 7, 16, 17, 40):
        many = lin(x[:count])
        assert all(_same(many[r:r + 1], one[r]) for r in range(count)), count
    shifted = lin(x[5:21])
    assert all(_same(shifted[r:r + 1], one[5 + r]) for r in range(16))


def _experts(rng, count, k, n, widths=(4, 5, 6)):
    groups = [_group(rng, k, n, widths[e % len(widths)]) for e in range(count)]
    return Exl3Experts([_linear(*g, "mul1") for g in groups]), groups


@pytest.mark.parametrize("rows", [1, 2, 5, 16, 40])
def test_experts_give_each_pick_its_alone_bits_and_the_reference(rows):
    rng = np.random.default_rng(rows)
    k, n, count, top = 512, 256, 9, 3
    ex, groups = _experts(rng, count, k, n)
    idx = np.stack([rng.choice(count, size=top, replace=False) for _ in range(rows)]).astype(np.int32)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    pick = mx.array(idx.reshape(-1))
    got = ex(mx.array(x).astype(mx.bfloat16), pick, div=top, members=rows, out_dtype=mx.float32)
    for r in range(rows):
        alone = ex(mx.array(x[r:r + 1]).astype(mx.bfloat16), mx.array(idx[r]), div=top, members=1,
                   out_dtype=mx.float32)
        assert _same(got[r * top:(r + 1) * top], alone), r
        for s in range(top):
            e = int(idx[r, s])
            want = fmt.forward(np.array(mx.array(x[r]).astype(mx.bfloat16).astype(mx.float32)), *groups[e],
                               _bits((4, 5, 6)[e % 3]), "mul1")
            g = np.array(got[r * top + s])
            assert np.linalg.norm(g - want) / np.linalg.norm(want) < 2e-3


def test_experts_trace_into_a_compiled_step():
    rng = np.random.default_rng(3)
    ex, _ = _experts(rng, 6, 256, 128)
    idx = mx.array(np.stack([rng.choice(6, size=2, replace=False) for _ in range(4)]).astype(np.int32))
    x = mx.array(rng.standard_normal((4, 256)).astype(np.float32)).astype(mx.bfloat16)

    def step(xs, ids):
        pick = ids.reshape(-1)
        return ex(xs, pick, T.groups(pick, 6), div=2, members=4)

    assert _same(mx.compile(step)(x, idx), step(x, idx))


def test_grouped_linear_runs_each_block_through_its_own_linear():
    rng = np.random.default_rng(4)
    parts = [_group(rng, 512, 128, k2) for k2 in (4, 6)]
    g = Exl3Grouped([_linear(*p, "mul1") for p in parts])
    x = rng.standard_normal((3, 1024)).astype(np.float32)
    got = np.array(g(mx.array(x)))
    for b, (part, k2) in enumerate(zip(parts, (4, 6))):
        want = fmt.forward(x[:, 512 * b:512 * (b + 1)], *part, _bits(k2), "mul1")
        block = got[:, 128 * b:128 * (b + 1)]
        assert np.linalg.norm(block - want) / np.linalg.norm(want) < 2e-3
    one = np.array(g(mx.array(x[1:2])))
    assert np.array_equal(one.view(np.uint32), got[1:2].view(np.uint32))
