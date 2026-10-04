# EXL3 weights on Metal (experimental)

The MLX lane reads EXL3 checkpoints through its own Metal kernels, `src/tensorfold/kernels/exl3/v1/`. The format
is the one [the CUDA module reads](exl3.md): every codebook (3inst, mcg, mul1) and width ExLlamaV3 writes (1 to 8
bits, the half widths with mul1), K and N multiples of 128. DeepSeek-V4-Flash is the first family that uses them
(`turboderp/DeepSeek-V4-Flash-0731-exl3`, see [its recipe](deepseek-v4-flash.md#exl3-on-a-128-gb-mac-experimental)).

- `trellis.py`: the kernels and the Metal header: tile decode, codebooks, the 128-point transform.
- `linear.py`: `Exl3Linear` (one layer), `Exl3Experts` (a stack read by picks, a width each), `Exl3Grouped` (G
  linears, block g of the input to block g of the output: DeepSeek's `wo_a`). Without Metal they run a reference
  built from `format.unpack`, so CPU tests drive whole engines.

## The layer

`y = rotate(fp16(rotate(x * suh)) @ W_q) * svh + bias`, three dispatches:

1. `rot_in`: a simdgroup a 128-block of a row. x times its expert's `suh` in fp32, the Walsh-Hadamard butterflies
   (two in registers, five through `simd_shuffle_xor`) in a fixed order, / sqrt(128), rounded to fp16 as ExLlamaV3
   rounds it.
2. `mm`: a threadgroup is 8 simdgroups, one 128-column block of output, one k split and up to 16 member rows of one
   active expert. A simdgroup walks its 16-column tile strip over the split's k tiles; each lane loads the 4 words
   holding its 8 windows, extracts the 16-bit states, decodes them through the codebook once and folds them into
   every row's own fp32 `fma` chain. Two `simd_shuffle_xor` adds give a column's sum. Partials go to `[SK, P, N]`.
3. `finish`: a simdgroup a 128-block of a row. The splits are summed in split order, transformed, `* svh + bias`.

The split count comes from the layer's shape only (`trellis.splits`), so **a row's bits depend on nothing but the
row**: not the rows beside it, not how many share the call, not which expert group it lands in. A decode window
needs no load-time twin check, and a prompt row gets the bits decoding it would give. The cost is that prompts
decode each weight once per 16 rows; a dedicated MMA prompt kernel is the next step if prefill becomes the limit.

## Experts

An `Exl3Experts` stack keeps its experts' trellises in one buffer with a word offset and K2 (2 x bits) per expert;
the kernel switches on the expert's K2 at runtime, with only the widths the stack holds compiled in. Picks are
grouped with MLX ops (`trellis.groups`: an argsort by (expert, pick), offsets from a scatter-add), so the grouping
traces into compiled decode steps. Each matrix of an MoE layer is one call: gate and up read the row (`div` = top
k), down reads each pick's activation.

## Arithmetic

- Codebooks: 3inst and mcg are one fp16 add of two bit-cast halves; mul1 is one fp16 `fma` of `1024 + byte sum`
  (Metal has no `dp4a`: two masked adds give the byte sum). Both round once, as the reference does.
- `tests/test_exl3_metal_header.py` compiles the header with clang (`_Float16`) and checks every tile decode and all
  65,536 states of each codebook against `format.py`; `tests/test_exl3_metal.py` checks on the GPU that the kernel
  returns W_q bit for bit, that rows are independent of their neighbours, and the layer against `format.forward`
  in float64 (relative error under 2e-3).

## Traps

- The CUDA kernels' bits and these are not the same; exactness is within one engine.
- `Exl3Experts` and `Exl3Linear` choose splits with different targets, so the same matrix in a stack and alone
  gives different (equally exact) bits.
- `tid2eid` rows that pick an expert twice would overflow an expert's member tiles; the loader refuses them.
