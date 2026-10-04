"""EXL3 trellis tiles on Metal: rotate rows in, a grouped decode-and-multiply where each row keeps its own bits, rotate out.

The format is ``tensorfold.cuda.exl3.format``'s (docs/recipes/exl3.md): W = diag(suh) H W_q H diag(svh), W_q stored
as 16x16 tiles of 16-bit trellis states, and a layer is ``y = rotate(rotate(x * suh) @ W_q) * svh + bias``. Three
kernels run it here:

- ``rot_in``: x * suh, a 128-point Walsh-Hadamard transform, / sqrt(128), rounded to fp16 (as ExLlamaV3 does).
- ``mm``: a simdgroup a 16-column tile strip and a fixed range of k tiles. Each lane decodes its 8 values of a tile
  once and folds them into every row's own fp32 fma chain, so a row's bits never depend on the rows beside it, how
  many share the call or which expert group it arrives in. Partials go out per k split.
- ``finish``: the splits summed in order, the output transform, * svh + bias.

The split count depends on the layer's shape only. Experts are the general case (a linear is one expert whose
members are all rows): a pick finds its expert's tiles through a word offset and a width (K2 = 2 * bits) per expert,
so experts of one layer can have different widths.
"""

from __future__ import annotations

import hashlib
from typing import Any

CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}
K2S = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)   # 2 * format.BITS: 1..8 bits, x.5 below 4 (mul1 only)
SG = 8                         # simdgroups a threadgroup: 8 n tiles, one 128-column block (256 threads)
ROWS = 16                      # rows a threadgroup takes at most (more: further threadgroups)
HAD = 128

HEADER = r"""
#define TFX3_UNROLL _Pragma("clang loop unroll(full)")
// E(j): where value j of a lane's 8 ends in the tile's bitstream, relative to the lane (exclusive)
template <int K2>
inline int tfx3_end(int j) { return (j >> 1) * K2 + ((j & 1) ? K2 : (K2 >> 1)); }

// The 16-bit states of values 8 * lane .. 8 * lane + 7 of a tile: windows of a circular MSB-first bitstream of
// 4 * K2 uint32 words, value p's window ending just before bit E(p)
template <int K2>
inline void tfx3_states(const device uint* tile, int lane, thread uint* s) {
  constexpr int NW = 4 * K2;
  const int f0 = 4 * lane * K2 + tfx3_end<K2>(0) - 16 + 128 * K2;
  const int w0 = f0 >> 5, o0 = f0 & 31;
  uint v[4];
  TFX3_UNROLL
  for (int q = 0; q < 4; q++) v[q] = tile[(w0 + q) % NW];
  TFX3_UNROLL
  for (int j = 0; j < 8; j++) {
    const int b = o0 + tfx3_end<K2>(j) - tfx3_end<K2>(0);
    const int wi = b >> 5, sh = b & 31;
    const uint a = wi == 0 ? v[0] : wi == 1 ? v[1] : v[2];
    const uint c = wi == 0 ? v[1] : wi == 1 ? v[2] : v[3];
    const uint x = sh <= 16 ? (a >> (16 - sh)) : ((a << (sh - 16)) | (c >> (48 - sh)));
    s[j] = x & 0xFFFFu;
  }
}

// A state's codebook value: 3inst (0) and mcg (1) one fp16 add of two bit-cast halves; mul1 (2) one fp16 fma of
// 1024 + the byte sum of s * 0x83DCD12D (each a single rounding, as ExLlamaV3's kernels)
template <int CB>
inline half tfx3_value(uint s) {
  if (CB == 2) {
    const uint x = s * 0x83DCD12Du;
    const uint t = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);
    const uint sum = (t & 0xFFFFu) + (t >> 16);
    return fma(as_type<half>(ushort(0x6400u + sum)), as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u)));
  } else {
    uint x = CB == 1 ? s * 0xCBAC1FEDu : s * 89226354u + 64248484u;
    x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return as_type<half>(ushort(x & 0xFFFFu)) + as_type<half>(ushort(x >> 16));
  }
}

// value j of a lane is at row 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), column lane / 4 + 8 * (j >> 2) of its tile
"""

FWHT = r"""
constant constexpr float TFX3_HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Unscaled Walsh-Hadamard transform of 128 values, 4 a lane (value 4 * lane + j), in a fixed butterfly order
inline void tfx3_fwht128(thread float* v, uint lane) {
  const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c;
  v[1] = b + d;
  v[2] = a - c;
  v[3] = b - d;
  TFX3_UNROLL
  for (ushort m = 1; m < 32; m <<= 1) {
    TFX3_UNROLL
    for (int j = 0; j < 4; j++) {
      const float o = simd_shuffle_xor(v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
  }
}
"""

_ROT_IN = r"""
  // a simdgroup one 128-block of one row: x * suh (its expert's), transformed, / sqrt(128), as fp16
  const uint lane = thread_index_in_simdgroup;
  const int blk = int(thread_position_in_grid.x) / 32;
  const int p = int(thread_position_in_grid.y);
  const int e = EXPERTS ? int(PICK[p]) : 0;
  const size_t xr = size_t(p / DIV) * K, sr = size_t(e) * K;
  float v[4];
  TFX3_UNROLL
  for (int j = 0; j < 4; j++) {
    const int i = blk * 128 + int(lane) * 4 + j;
    v[j] = float(X[xr + i]) * float(SUH[sr + i]);
  }
  tfx3_fwht128(v, lane);
  TFX3_UNROLL
  for (int j = 0; j < 4; j++) XH[size_t(p) * K + blk * 128 + int(lane) * 4 + j] = half(v[j] * TFX3_HAD_SCALE);
"""

_MM_RUN = r"""
// k tiles kt0 .. kt1 - 1 of one 16-column tile strip, every member row its own fma chain (j order fixed)
template <int K2, int CB, int RT, int K, int N>
inline void tfx3_run(const device uint* base, const device half* XH, thread const int* pick, int rows, int nt,
                     int kt0, int kt1, int lane, thread float (*acc)[2]) {
  constexpr int NT = N / 16;
  const int kq = 2 * (lane & 3);
  for (int kt = kt0; kt < kt1; kt++) {
    uint s[8];
    tfx3_states<K2>(base + (size_t(kt) * NT + nt) * (4 * K2), lane, s);
    float w[8];
    TFX3_UNROLL
    for (int j = 0; j < 8; j++) w[j] = float(tfx3_value<CB>(s[j]));
    const int kb = kt * 16 + kq;
    TFX3_UNROLL
    for (int r = 0; r < RT; r++) {
      if (r < rows) {
        const device half* xr = XH + size_t(pick[r]) * K + kb;
        const float x0 = float(xr[0]), x1 = float(xr[1]), x8 = float(xr[8]), x9 = float(xr[9]);
        acc[r][0] = fma(x0, w[0], acc[r][0]);
        acc[r][0] = fma(x1, w[1], acc[r][0]);
        acc[r][0] = fma(x8, w[2], acc[r][0]);
        acc[r][0] = fma(x9, w[3], acc[r][0]);
        acc[r][1] = fma(x0, w[4], acc[r][1]);
        acc[r][1] = fma(x1, w[5], acc[r][1]);
        acc[r][1] = fma(x8, w[6], acc[r][1]);
        acc[r][1] = fma(x9, w[7], acc[r][1]);
      }
    }
  }
}
"""

_MM = r"""
  // threadgroup: 8 tile strips (128 columns) x one k split x RT member rows of one active expert
  const int lane = int(thread_index_in_simdgroup);
  const int nt = int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup);
  const int ks = int(threadgroup_position_in_grid.y);
  const int u = int(threadgroup_position_in_grid.z) / MT, mt = int(threadgroup_position_in_grid.z) % MT;
  if (u >= int(UC[0])) return;
  const int e = int(ACT[u]);
  const int m0 = int(OFF[e]) + mt * RT, m1 = min(int(OFF[e + 1]), m0 + RT);
  if (m0 >= m1) return;
  const int rows = m1 - m0;
  constexpr int KT = K / 16;
  const int kt0 = ks * KT / SK, kt1 = (ks + 1) * KT / SK;
  const int P = int(XH_shape[0]);
  int pick[RT];
  float acc[RT][2];
  TFX3_UNROLL
  for (int r = 0; r < RT; r++) {
    pick[r] = int(MEM[min(m0 + r, m1 - 1)]);
    acc[r][0] = 0.0f;
    acc[r][1] = 0.0f;
  }
  const device uint* base = W + size_t(uint(TB[e]));
  switch (int(TK[e])) {
CASES
    default: return;
  }
  TFX3_UNROLL
  for (int r = 0; r < RT; r++) {
    if (r < rows) {
      TFX3_UNROLL
      for (int c = 0; c < 2; c++) {
        float a = acc[r][c];
        a += simd_shuffle_xor(a, ushort(1));
        a += simd_shuffle_xor(a, ushort(2));
        if ((lane & 3) == 0) Z[(size_t(ks) * P + pick[r]) * N + nt * 16 + (lane >> 2) + 8 * c] = a;
      }
    }
  }
"""

_FINISH = r"""
  // a simdgroup one 128-block of one row: the k splits summed in order, transformed, * svh (its expert's) + bias
  const uint lane = thread_index_in_simdgroup;
  const int blk = int(thread_position_in_grid.x) / 32;
  const int p = int(thread_position_in_grid.y);
  const int P = int(Z_shape[1]);
  const int e = EXPERTS ? int(PICK[p]) : 0;
  float v[4];
  TFX3_UNROLL
  for (int j = 0; j < 4; j++) {
    const size_t i = size_t(blk) * 128 + lane * 4 + j;
    float a = Z[size_t(p) * N + i];
    for (int s = 1; s < SK; s++) a += Z[(size_t(s) * P + p) * N + i];
    v[j] = a;
  }
  tfx3_fwht128(v, lane);
  TFX3_UNROLL
  for (int j = 0; j < 4; j++) {
    const size_t i = size_t(blk) * 128 + lane * 4 + j;
    float y = v[j] * TFX3_HAD_SCALE * float(SVH[size_t(e) * N + i]);
    if (HAS_BIAS) y += float(BIAS[i]);
    OUT[size_t(p) * N + i] = OutT(y);
  }
"""


def splits(k: int, n: int, target: int = 256) -> int:
    """k splits for a [K, N] layer: enough threadgroups at one active expert, from the shape alone (a row's bits)."""

    kt, blocks = k // 16, n // HAD
    sk = 1
    while blocks * sk * 2 <= target and sk * 2 <= 32 and kt // (sk * 2) >= 8:
        sk *= 2
    return sk


def row_tile(rows: int) -> int:
    """Rows a threadgroup takes (1, 2, 4, 8 or 16): any choice leaves every row's bits alone."""

    rt = 1
    while rt < min(rows, ROWS):
        rt *= 2
    return rt


_kernels: dict[tuple, Any] = {}


def _kernel(kind: str, consts: tuple[tuple[str, Any], ...], inputs: list[str], outputs: list[str], body: str) -> Any:
    import mlx.core as mx

    key = (kind, consts)
    kernel = _kernels.get(key)
    if kernel is None:
        source = "".join(f"  constexpr int {name} = {int(value)};\n" for name, value in consts if name != "OutT")
        source += "".join(f"  using OutT = {value};\n" for name, value in consts if name == "OutT") + body
        full = HEADER + FWHT + _MM_RUN
        digest = hashlib.sha256((full + source).encode()).hexdigest()[:16]
        kernel = _kernels[key] = mx.fast.metal_kernel(name=f"tensorfold_exl3_{kind}_{digest}", input_names=inputs,
                                                      output_names=outputs, source=source, header=full)
    return kernel


def rot_in(x: Any, suh: Any, pick: Any | None, *, div: int = 1, rows: int | None = None) -> Any:
    """fp16 rotated inputs [P, K]: row p of x [R, K] (row p // div) times its expert's suh [E, K], transformed."""

    import mlx.core as mx

    from tensorfold.kernels.inputs import padded

    k = int(x.shape[-1])
    p = int(rows if rows is not None else x.shape[0] * div)
    consts = (("K", k), ("DIV", div), ("EXPERTS", int(pick is not None)))
    kernel = _kernel("rot_in", consts, ["X", "SUH", "PICK"], ["XH"], _ROT_IN)
    pick = padded(pick if pick is not None else mx.zeros((1,), dtype=mx.int32))
    blocks = k // HAD
    tg = 32 * max(b for b in (1, 2, 4, 8) if blocks % b == 0)
    return kernel(inputs=[mx.contiguous(x), suh, pick], grid=(32 * blocks, p, 1), threadgroup=(tg, 1, 1),
                  output_shapes=[(p, k)], output_dtypes=[mx.float16])[0]


def mm(xh: Any, words: Any, tb: Any, tk: Any, k2s: tuple[int, ...], group: tuple[Any, Any, Any, Any], *, k: int,
       n: int, sk: int, codebook: int, units: int, members: int) -> Any:
    """Rotated-domain partials [SK, P, N] fp32 of xh [P, K] through each pick's expert (``group`` from ``groups``)."""

    import mlx.core as mx

    act, off, mem, uc = group
    p = int(xh.shape[0])
    rt = row_tile(members)
    mt = -(-members // rt)
    cases = "\n".join(f"    case {k2}: tfx3_run<{k2}, CB, RT, K, N>(base, XH, pick, rows, nt, kt0, kt1, lane, acc); "
                      "break;" for k2 in sorted(set(k2s)))
    consts = (("K", k), ("N", n), ("SK", sk), ("RT", rt), ("MT", mt), ("CB", codebook), ("SG", SG),
              ("WIDTHS", sum(1 << v for v in set(k2s))))
    kernel = _kernel("mm", consts, ["XH", "W", "TB", "TK", "ACT", "OFF", "MEM", "UC"], ["Z"],
                     _MM.replace("CASES", cases))
    return kernel(inputs=[xh, words, tb, tk, act, off, mem, uc], grid=(32 * SG * (n // HAD), sk, units * mt),
                  threadgroup=(32 * SG, 1, 1), output_shapes=[(sk, p, n)], output_dtypes=[mx.float32])[0]


def finish(z: Any, svh: Any, bias: Any | None, pick: Any | None, out_dtype: Any) -> Any:
    """[P, N] outputs from mm's partials: splits summed in order, transformed, * svh (each pick's expert) + bias."""

    import mlx.core as mx

    from tensorfold.kernels.inputs import padded

    sk, p, n = (int(v) for v in z.shape)
    out_t = {mx.float32: "float", mx.float16: "half", mx.bfloat16: "bfloat16_t"}[out_dtype]
    consts = (("N", n), ("SK", sk), ("EXPERTS", int(pick is not None)), ("HAS_BIAS", int(bias is not None)),
              ("OutT", out_t))
    kernel = _kernel("finish", consts, ["Z", "SVH", "BIAS", "PICK"], ["OUT"], _FINISH)
    bias = padded(bias if bias is not None else mx.zeros((1,), dtype=mx.float16))
    pick = padded(pick if pick is not None else mx.zeros((1,), dtype=mx.int32))
    blocks = n // HAD
    tg = 32 * max(b for b in (1, 2, 4, 8) if blocks % b == 0)
    return kernel(inputs=[z, svh, bias, pick], grid=(32 * blocks, p, 1), threadgroup=(tg, 1, 1),
                  output_shapes=[(p, n)], output_dtypes=[out_dtype])[0]


def groups(pick: Any, experts: int) -> tuple[Any, Any, Any, Any]:
    """Picks [P] (expert ids) grouped by expert: active experts in id order, member offsets [E + 1], picks sorted by
    (expert, pick) and the active count; MLX ops, so it traces into compiled decode steps."""

    import mlx.core as mx

    from tensorfold.kernels.inputs import padded

    p = int(pick.shape[0])
    order = mx.arange(p, dtype=mx.int32)
    mem = mx.argsort(pick.astype(mx.int32) * p + order).astype(mx.int32)
    counts = mx.zeros((experts,), dtype=mx.int32).at[pick].add(mx.ones((p,), dtype=mx.int32))
    off = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)]).astype(mx.int32)
    ids = mx.arange(experts, dtype=mx.int32)
    present = counts > 0
    act = mx.argsort(mx.where(present, ids, ids + experts)).astype(mx.int32)
    uc = present.astype(mx.int32).sum(keepdims=True)
    return padded(act), padded(off), padded(mem), padded(uc)


def dense_group(rows: int) -> tuple[Any, Any, Any, Any]:
    """One expert (0) whose members are rows 0 .. rows - 1."""

    import mlx.core as mx

    from tensorfold.kernels.inputs import ints, padded

    return ints([0]), ints([0, rows]), padded(mx.arange(rows, dtype=mx.int32)), ints([1])


__all__ = ["CODEBOOK_IDS", "FWHT", "HEADER", "K2S", "ROWS", "SG", "dense_group", "finish", "groups", "mm",
           "row_tile", "rot_in", "splits"]
