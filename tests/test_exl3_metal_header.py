"""The Metal EXL3 header's tile decode, compiled for the CPU, against format.py's reference for every width and codebook."""

from __future__ import annotations

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.kernels.exl3.v1 import trellis

WIDTHS = [(k2, cb) for k2 in trellis.K2S for cb in fmt.CODEBOOKS if k2 / 2 in fmt.BITS and (k2 % 2 == 0 or cb == "mul1")]


@pytest.fixture(scope="module")
def decode(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ compiles the Metal header (it has _Float16) to check its tile decode on the CPU")
    folder = tmp_path_factory.mktemp("exl3-header-cpu")
    source, library = folder / "tiles.cpp", folder / "tiles.so"
    shim = r"""
#include <cstdint>
#include <cstring>
using uint = std::uint32_t;
using ushort = std::uint16_t;
using half = _Float16;
#define device
#define thread
template <typename T, typename U> T as_type(U u) { T t; static_assert(sizeof t == sizeof u); std::memcpy(&t, &u, sizeof t); return t; }
// the fp16 fma rounds once: a * b + c is exact in double for these operands
inline half fma(half a, half b, half c) { return (half)((double)a * (double)b + (double)c); }
"""
    cases = "".join(f"case {k2 * 4 + cb}: for (int l = 0; l < 32; l++) {{ uint s[8]; tfx3_states<{k2}>(words, l, s); "
                    f"for (int j = 0; j < 8; j++) out[8 * l + j] = as_type<ushort>(tfx3_value<{cb}>(s[j])); }} return 1;"
                    for k2 in trellis.K2S for cb in range(3))
    source.write_text(shim + trellis.HEADER + '\nextern "C" int tile(const uint* words, int k2, int cb, ushort* out) '
                      "{ switch (k2 * 4 + cb) {" + cases + " default: return 0; } }\n"
                      'extern "C" void values(const uint* s, int n, int cb, ushort* out) { for (int i = 0; i < n; i++) '
                      "out[i] = as_type<ushort>(cb == 0 ? tfx3_value<0>(s[i]) : cb == 1 ? tfx3_value<1>(s[i]) "
                      ": tfx3_value<2>(s[i])); }\n")
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O1", str(source), "-o", str(library)],
                   check=True, capture_output=True, timeout=120)
    lib = ctypes.CDLL(str(library))
    lib.tile.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint16))
    lib.tile.restype = ctypes.c_int

    def run(words: np.ndarray, k2: int, cb: int) -> np.ndarray:
        out = np.zeros(256, dtype=np.uint16)
        w = np.ascontiguousarray(words, dtype=np.uint32)
        assert lib.tile(w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), k2, cb,
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))) == 1
        return out

    def values(states: np.ndarray, cb: int) -> np.ndarray:
        s = np.ascontiguousarray(states, dtype=np.uint32)
        out = np.zeros(s.shape[0], dtype=np.uint16)
        lib.values(s.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), int(s.shape[0]), cb,
                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
        return out

    run.values = values
    return run


@pytest.mark.parametrize("k2,codebook", WIDTHS)
def test_header_decodes_tiles_as_the_reference(decode, k2, codebook):
    bits = k2 // 2 if k2 % 2 == 0 else k2 / 2
    rng = np.random.default_rng(k2 * 7 + len(codebook))
    tiles = rng.integers(-32768, 32768, size=(3, 1, 8 * k2), dtype=np.int64).astype(np.int16)
    want = fmt.unpack(tiles, bits, codebook)                          # [48, 16] fp16
    rows, cols = fmt.tile_positions()
    for t in range(tiles.shape[0]):
        stream = decode(tiles[t, 0].view(np.uint32), k2, trellis.CODEBOOK_IDS[codebook]).view(np.float16)
        got = np.zeros((16, 16), dtype=np.float16)
        got[rows, cols] = stream
        assert np.array_equal(got.view(np.uint16), want[16 * t:16 * (t + 1)].view(np.uint16))


def test_every_state_takes_its_codebook_value(decode):
    states = np.arange(65536, dtype=np.uint32)
    for codebook, cb in trellis.CODEBOOK_IDS.items():
        assert np.array_equal(decode.values(states, cb), fmt.codebook(codebook).view(np.uint16)), codebook


def test_split_k_ranges_cover_every_tile_once():
    for k in (128, 1024, 4096, 8192):
        for n in (128, 1024, 4096, 32768, 129280):
            sk = trellis.splits(k, n)
            kt = k // 16
            ranges = [(s * kt // sk, (s + 1) * kt // sk) for s in range(sk)]
            assert ranges[0][0] == 0 and ranges[-1][1] == kt
            assert all(a[1] == b[0] and a[0] < a[1] for a, b in zip(ranges, ranges[1:]))
            assert sk == trellis.splits(k, n)


def test_row_tiles_are_powers_of_two_up_to_sixteen():
    assert [trellis.row_tile(r) for r in (1, 2, 3, 5, 9, 16, 17, 2048)] == [1, 2, 4, 8, 16, 16, 16, 16]
