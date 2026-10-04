"""Time the Metal EXL3 kernels at DeepSeek-V4-Flash's shapes against MLX's own quantized matmul (Apple GPU only).

For each projection: microseconds a call at 1, 4 and 16 rows, the trellis bytes it reads and the bandwidth that
gives, beside ``mx.quantized_matmul`` at 4 and 2 bits on the same shape. The routed experts run as a decode step
does (6 picks a row, gate + up + down through the grouped kernel). The last lines estimate one token's projection
time over 43 layers from the one-row timings.

  python tools/bench_exl3_metal.py [--reps 50] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import numpy as np

from tensorfold.kernels.exl3.v1.linear import Exl3Experts, Exl3Grouped, Exl3Linear

# (name, K, N, bits, count a layer) at turboderp's 2.52bpw pack's widths
DENSE = [("wq_a", 4096, 1024, 4, 1), ("wkv", 4096, 512, 4, 1), ("wq_b", 1024, 32768, 4, 1),
         ("wo_b", 8192, 4096, 4, 1), ("compressor wkv+wgate", 4096, 1024, 4, 2), ("shared w1/w3", 4096, 2048, 4, 2),
         ("shared w2", 2048, 4096, 4, 1)]
HEAD = ("head", 4096, 129280, 6)
EXPERTS, TOP, INTER, D = 256, 6, 2048, 4096


def _group(rng, k, n, bits):
    t = mx.array(rng.integers(-32768, 32768, size=(k // 16, n // 16, int(16 * bits))).astype(np.int16))
    suh = mx.array(np.where(rng.random(k) < 0.5, -1.0, 1.0).astype(np.float16))
    svh = mx.array((0.02 * np.ones(n)).astype(np.float16))
    return t, suh, svh


def _time(fn, reps):
    for _ in range(3):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def bench_dense(rng, name, k, n, bits, reps):
    lin = Exl3Linear(*_group(rng, k, n, bits), codebook="mul1")
    mx.eval(*lin.arrays())
    w = mx.array(rng.standard_normal((n, k)).astype(np.float32)).astype(mx.bfloat16)
    q4, q2 = mx.quantize(w, group_size=64, bits=4), mx.quantize(w, group_size=64, bits=2)
    mx.eval(*q4, *q2)
    out = {"name": name, "K": k, "N": n, "bits": bits, "trellis_MB": k * n * bits / 8 / 1e6}
    for rows in (1, 4, 16):
        x = mx.array(rng.standard_normal((rows, k)).astype(np.float32)).astype(mx.bfloat16)
        us = _time(lambda: lin(x), reps)
        out[f"exl3_us_{rows}"] = us
        out[f"exl3_GBs_{rows}"] = out["trellis_MB"] / 1e3 / (us / 1e6)
        out[f"mlx4_us_{rows}"] = _time(lambda: mx.quantized_matmul(x, *q4, transpose=True, group_size=64, bits=4), reps)
        out[f"mlx2_us_{rows}"] = _time(lambda: mx.quantized_matmul(x, *q2, transpose=True, group_size=64, bits=2), reps)
    return out


def bench_experts(rng, reps):
    widths = (2, 2.5, 3)
    mats = []
    for k, n in ((D, INTER), (D, INTER), (INTER, D)):
        mats.append(Exl3Experts([Exl3Linear(*_group(rng, k, n, widths[e % 3]), codebook="mul1")
                                 for e in range(EXPERTS)]))
        mx.eval(*mats[-1].arrays())
    gate, up, down = mats
    out = {"name": "routed experts (6 a row, gate+up+down)"}
    for rows in (1, 4, 16):
        idx = mx.array(np.stack([rng.choice(EXPERTS, TOP, replace=False) for _ in range(rows)]).astype(np.int32))
        x = mx.array(rng.standard_normal((rows, D)).astype(np.float32)).astype(mx.bfloat16)

        def step():
            from tensorfold.kernels.exl3.v1 import trellis as T

            pick = idx.reshape(-1)
            grp = T.groups(pick, EXPERTS)
            g = gate(x, pick, grp, div=TOP, members=rows)
            u = up(x, pick, grp, div=TOP, members=rows)
            return down(g * u, pick, grp, members=rows)

        out[f"exl3_us_{rows}"] = _time(step, reps)
    return out


def bench_grouped(rng, reps):
    g = Exl3Grouped([Exl3Linear(*_group(rng, 4096, 1024, 4), codebook="mul1") for _ in range(8)])
    mx.eval(*g.arrays())
    out = {"name": "wo_a (8 groups)", "K": 8 * 4096, "N": 8 * 1024, "bits": 4}
    for rows in (1, 4, 16):
        x = mx.array(rng.standard_normal((rows, 8 * 4096)).astype(np.float32)).astype(mx.bfloat16)
        out[f"exl3_us_{rows}"] = _time(lambda: g(x), reps)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--json")
    args = p.parse_args()
    if not mx.metal.is_available():
        raise SystemExit("needs an Apple GPU (Metal)")
    mx.set_default_device(mx.gpu)
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    print(f"# {info.get('device_name') or info.get('architecture')}, MLX {mx.__version__}")
    rng = np.random.default_rng(0)
    rows = [bench_dense(rng, *d[:4], args.reps) for d in DENSE] + [bench_dense(rng, *HEAD, args.reps)]
    rows += [bench_grouped(rng, args.reps), bench_experts(rng, args.reps)]
    for r in rows:
        cells = "  ".join(f"{key}={value:.1f}" for key, value in r.items() if isinstance(value, float))
        print(f"{r['name']:<40} {cells}")
    layer_us = sum(r["exl3_us_1"] * d[4] for r, d in zip(rows, DENSE)) + rows[-2]["exl3_us_1"] + rows[-1]["exl3_us_1"]
    token_ms = (43 * layer_us + rows[len(DENSE)]["exl3_us_1"]) / 1e3
    print(f"# one row: {layer_us:.0f} us of projections a layer, ~{token_ms:.1f} ms a token over 43 layers + head "
          f"(projections only, ~{1e3 / token_ms:.1f} tok/s ceiling before attention, norms and routing)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
