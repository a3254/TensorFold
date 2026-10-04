"""EXL3 linears and expert stacks for the MLX engines: Metal kernels on Apple GPUs, a dequantized reference elsewhere.

``Exl3Linear`` and ``Exl3Experts`` stand where an engine keeps its ``Q`` linears (``outs``, ``ins``, ``arrays()``,
``__call__``). Their rows never share bits (``exact_rows``), so a decode window runs in one call and a prompt chunk
gets the same bits as decoding it would. Without Metal (CPU tests) they compute the same layer from W_q in fp32.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.kernels.exl3.v1 import trellis as T
from tensorfold.kernels.inputs import ints


def metal() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _scales(get: Callable[[str], mx.array], has: Callable[[str], bool], prefix: str, kind: str) -> mx.array:
    """``suh``/``svh`` as fp16, or the packed sign words ``su``/``sv`` of older checkpoints expanded."""

    if has(f"{prefix}.{kind}h"):
        return get(f"{prefix}.{kind}h").astype(mx.float16)
    import numpy as np

    return mx.array(fmt.unpack_signs(np.array(get(f"{prefix}.{kind}"))))


def codebook_of(has: Callable[[str], bool], prefix: str) -> str:
    found = [m for m in ("mcg", "mul1") if has(f"{prefix}.{m}")]
    if len(found) > 1:
        raise ValueError(f"{prefix}: both mcg and mul1 markers")
    return found[0] if found else "3inst"


def _check_trellis(prefix: str, trellis: mx.array, codebook: str) -> int:
    """K2 (2 * bits) of an int16 [K/16, N/16, 16 * bits] trellis; ValueError for what the kernels don't read."""

    if trellis.ndim != 3 or trellis.dtype != mx.int16:
        raise ValueError(f"{prefix}: trellis must be int16 [K/16, N/16, 16 * bits], got {trellis.dtype} "
                         f"{tuple(trellis.shape)}")
    k2 = int(trellis.shape[-1]) // 8
    if k2 * 8 != int(trellis.shape[-1]) or k2 not in T.K2S:
        raise ValueError(f"{prefix}: {int(trellis.shape[-1]) / 16:g}-bit tiles are not an EXL3 width")
    if k2 % 2 and codebook != "mul1":
        raise ValueError(f"{prefix}: {k2 / 2:g}-bit tiles need the mul1 codebook, found {codebook}")
    k, n = 16 * int(trellis.shape[0]), 16 * int(trellis.shape[1])
    if k % T.HAD or n % T.HAD:
        raise ValueError(f"{prefix}: K={k} and N={n} must be multiples of {T.HAD} (the Hadamard blocks)")
    return k2


def _hadamard() -> mx.array:
    return mx.array(fmt.hadamard() / (fmt.HAD ** 0.5), dtype=mx.float32)


def _rotate(x: mx.array) -> mx.array:
    """H / sqrt(128) on every 128-block of the last axis, fp32 (the reference path; a block a product, so a row's bits
    never depend on the rows beside it)."""

    shape = x.shape
    return (x.astype(mx.float32).reshape(-1, 1, T.HAD) @ _hadamard()).reshape(shape)


class Exl3Linear:
    """One EXL3 linear: y = rotate(fp16(rotate(x * suh)) @ W_q) * svh + bias, K and N multiples of 128."""

    bits = None
    group = None
    exact_rows = True

    def __init__(self, trellis: mx.array, suh: mx.array, svh: mx.array, bias: mx.array | None = None, *,
                 codebook: str = "3inst", prefix: str = "") -> None:
        if codebook not in T.CODEBOOK_IDS:
            raise ValueError(f"{prefix}: unknown EXL3 codebook {codebook!r}")
        k2 = _check_trellis(prefix, trellis, codebook)
        self.k, self.n = 16 * int(trellis.shape[0]), 16 * int(trellis.shape[1])
        if tuple(suh.shape) != (self.k,) or tuple(svh.shape) != (self.n,):
            raise ValueError(f"{prefix}: scales {tuple(suh.shape)} / {tuple(svh.shape)} do not fit K={self.k} "
                             f"N={self.n}")
        self.prefix, self.codebook, self.k2 = prefix, codebook, k2
        self.trellis = None if metal() else trellis                 # the reference unpacks it (no Metal)
        self.words = trellis.reshape(-1).view(mx.uint32)
        self.suh = suh.astype(mx.float16).reshape(1, self.k)
        self.svh = svh.astype(mx.float16).reshape(1, self.n)
        self.bias = None if bias is None else bias.astype(mx.float16).reshape(self.n)
        self.tb, self.tk = ints([0]), ints([k2])
        self.sk = T.splits(self.k, self.n)
        self._wq: mx.array | None = None
        if self.trellis is not None:
            self.wq()                                   # now, not inside a compiled step (it reads to numpy)

    @classmethod
    def load(cls, get: Callable[[str], mx.array], has: Callable[[str], bool], prefix: str) -> "Exl3Linear":
        """The group under ``prefix`` (``.trellis``, ``.suh``/``.su``, ``.svh``/``.sv``, optional ``.bias``)."""

        return cls(get(f"{prefix}.trellis"), _scales(get, has, prefix, "su"), _scales(get, has, prefix, "sv"),
                   get(f"{prefix}.bias") if has(f"{prefix}.bias") else None, codebook=codebook_of(has, prefix),
                   prefix=prefix)

    @property
    def ins(self) -> int:
        return self.k

    @property
    def outs(self) -> int:
        return self.n

    @property
    def bits_per_weight(self) -> float:
        return self.k2 / 2

    def arrays(self) -> list[mx.array]:
        return [self.words, self.suh, self.svh, *([self.bias] if self.bias is not None else [])]

    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.arrays())

    def __call__(self, x: mx.array, out_dtype: Any = None) -> mx.array:
        """x [..., K] -> [..., N] in x's dtype (or ``out_dtype``)."""

        out_dtype = out_dtype or x.dtype
        shape = x.shape
        x2 = x.reshape(-1, self.k)
        rows = int(x2.shape[0])
        if not metal():
            return self.reference(x2).astype(out_dtype).reshape(*shape[:-1], self.n)
        xh = T.rot_in(x2, self.suh, None)
        z = T.mm(xh, self.words, self.tb, self.tk, (self.k2,), T.dense_group(rows), k=self.k, n=self.n, sk=self.sk,
                 codebook=T.CODEBOOK_IDS[self.codebook], units=1, members=rows)
        return T.finish(z, self.svh, self.bias, None, out_dtype).reshape(*shape[:-1], self.n)

    def wq(self) -> mx.array:
        """W_q [K, N] fp16 (the rotated-domain weight), unpacked once on the CPU by the format reference."""

        if self._wq is None:
            import numpy as np

            self._wq = mx.array(fmt.unpack(np.array(self.trellis), self.k2 / 2 if self.k2 % 2 else self.k2 // 2,
                                           self.codebook))
        return self._wq

    def reference(self, x2: mx.array) -> mx.array:
        """The layer without Metal, fp32 after the fp16 rotated input (the kernels' arithmetic, not their bits)."""

        xh = _rotate(x2.astype(mx.float32) * self.suh.astype(mx.float32)).astype(mx.float16)
        y = _rotate((xh.astype(mx.float32)[:, None, :] @ self.wq().astype(mx.float32)).squeeze(1))
        y = y * self.svh.astype(mx.float32)
        return y if self.bias is None else y + self.bias.astype(mx.float32)


class Exl3Experts:
    """E EXL3 matrices of one shape read by picks: pick p's row through expert pick[p], each expert at its own width.

    The trellises sit in one buffer (one copy at load), each expert at a word offset with its K2, so the kernel finds
    a pick's tiles without repacking and experts of a layer can mix widths.
    """

    exact_rows = True

    def __init__(self, parts: list[Exl3Linear]) -> None:
        if not parts:
            raise ValueError("no experts")
        k, n = parts[0].k, parts[0].n
        codebooks = {p.codebook for p in parts}
        if len(codebooks) > 1 or any((p.k, p.n) != (k, n) for p in parts) or any(p.bias is not None for p in parts):
            raise ValueError(f"{parts[0].prefix}: experts of one stack share their shape and codebook and have no bias "
                             f"(found {sorted(codebooks)}, {sorted({(p.k, p.n) for p in parts})})")
        self.k, self.n, self.count = k, n, len(parts)
        self.codebook = parts[0].codebook
        offsets, at = [], 0
        for p in parts:
            offsets.append(at)
            at += int(p.words.shape[0])
        if at >= 1 << 31:
            raise ValueError(f"{parts[0].prefix}: {at} trellis words in one expert stack (at most 2^31)")
        self.words = mx.concatenate([p.words for p in parts])
        self.tb, self.tk = ints(offsets), ints([p.k2 for p in parts])
        self.k2s = tuple(sorted({p.k2 for p in parts}))
        self.k2_of = [p.k2 for p in parts]
        self.suh = mx.concatenate([p.suh for p in parts])           # [E, K]
        self.svh = mx.concatenate([p.svh for p in parts])           # [E, N]
        self.sk = T.splits(k, n, target=32)
        self._wq = None if metal() else mx.stack([p.wq().astype(mx.float32) for p in parts])   # [E, K, N]

    def arrays(self) -> list[mx.array]:
        return [self.words, self.suh, self.svh]

    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.arrays())

    @property
    def ins(self) -> int:
        return self.k

    @property
    def outs(self) -> int:
        return self.n

    def __call__(self, x: mx.array, pick: mx.array, group: tuple[Any, ...] | None = None, *, div: int = 1,
                 members: int | None = None, out_dtype: Any = None) -> mx.array:
        """Picks [P] (expert ids) through their experts: pick p reads row p // div of x [P / div, K]; -> [P, N].

        ``members`` bounds the picks any one expert takes (rows of x when each row picks an expert once).
        """

        out_dtype = out_dtype or x.dtype
        pick = pick.reshape(-1).astype(mx.int32)
        p = int(pick.shape[0])
        if not metal():
            return self.reference(x, pick, div).astype(out_dtype)
        group = group if group is not None else T.groups(pick, self.count)
        xh = T.rot_in(x, self.suh, pick, div=div, rows=p)
        z = T.mm(xh, self.words, self.tb, self.tk, self.k2s, group, k=self.k, n=self.n, sk=self.sk,
                 codebook=T.CODEBOOK_IDS[self.codebook], units=min(p, self.count), members=members or p)
        return T.finish(z, self.svh, None, pick, out_dtype)

    def reference(self, x: mx.array, pick: mx.array, div: int) -> mx.array:
        """The picks without Metal (no host reads, so it traces into compiled steps): fp32 after the fp16 inputs."""

        if self._wq is None:
            raise RuntimeError("the EXL3 expert reference keeps W_q only where Metal is unavailable")
        rows = x.reshape(-1, self.k)[mx.arange(int(pick.shape[0])) // div].astype(mx.float32)
        xh = _rotate(rows * self.suh[pick].astype(mx.float32)).astype(mx.float16).astype(mx.float32)
        z = (xh[:, None, :] @ self._wq[pick]).squeeze(1)
        return _rotate(z) * self.svh[pick].astype(mx.float32)


class Exl3Grouped:
    """G EXL3 linears where input block g feeds output block g (DeepSeek-V4's grouped low-rank output): one call."""

    exact_rows = True
    bits = None
    group = None

    def __init__(self, parts: list[Exl3Linear]) -> None:
        self.groups = len(parts)
        self.experts = Exl3Experts(parts)
        self._pick: dict[int, mx.array] = {}

    @property
    def ins(self) -> int:
        return self.groups * self.experts.k

    @property
    def outs(self) -> int:
        return self.groups * self.experts.n

    def arrays(self) -> list[mx.array]:
        return self.experts.arrays()

    def __call__(self, x: mx.array) -> mx.array:
        """x [R, G * K] (block g read by linear g) -> [R, G * N]."""

        rows = int(x.shape[0])
        pick = self._pick.get(rows)
        if pick is None:
            pick = self._pick[rows] = mx.tile(mx.arange(self.groups, dtype=mx.int32), rows)
        y = self.experts(x.reshape(rows * self.groups, self.experts.k), pick, members=rows)
        return y.reshape(rows, self.outs)


__all__ = ["Exl3Experts", "Exl3Grouped", "Exl3Linear", "codebook_of", "metal"]
