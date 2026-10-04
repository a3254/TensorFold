#!/usr/bin/env bash
# Check the Metal EXL3 port on an Apple Silicon Mac and write one report to paste back.
#
#   tools/mac_check_exl3.sh                 # kernel tests + kernel bench (minutes)
#   MODEL=1 tools/mac_check_exl3.sh         # also serve + bench turboderp's DeepSeek-V4-Flash EXL3 pack (~91 GB, a
#                                           # 128 GB Mac) from $MODEL_DIR, downloading it there only if it is absent
#   MODEL=1 MODEL_DIR=/path/to/it tools/mac_check_exl3.sh   # a copy you downloaded yourself
#
# Run from the repo root inside the venv TensorFold is installed in (python -m pip install -e ".[test]").
set -uo pipefail

REPORT=${REPORT:-exl3-mac-report.txt}
BRANCH=${BRANCH:-2.52bpw}
MODEL_DIR=${MODEL_DIR:-$HOME/models/DeepSeek-V4-Flash-0731-exl3-$BRANCH}
DRAFTER=${DRAFTER:-TensorFold/DeepSeek-V4-Flash-MTP-MLX}
PORT=${PORT:-8080}

exec > >(tee "$REPORT") 2>&1

section() { printf '\n===== %s =====\n' "$*"; }

section "host"
date
sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null
sysctl iogpu.wired_limit_mb 2>/dev/null
sw_vers 2>/dev/null | tr '\n' ' '; echo
git log --oneline -1
python -c "import mlx.core as mx, platform; print('python', platform.python_version(), 'mlx', mx.__version__); \
print({k: v for k, v in mx.device_info().items() if k in ('device_name', 'architecture', 'memory_size', \
'max_recommended_working_set_size')})"

section "tests: Metal header on the CPU, Metal kernels, EXL3 engine (CPU reference), DeepSeek family"
python -m pytest -q -p no:cacheprovider tests/test_exl3_metal_header.py tests/test_exl3_metal.py \
  tests/test_dsv4_exl3.py tests/test_deepseek_v4_family.py tests/test_unsupported_checkpoints.py \
  tests/test_exl3_format.py 2>&1 | tail -40

section "kernel bench"
python tools/bench_exl3_metal.py --reps 30

if [ "${MODEL:-0}" != "1" ]; then
  section "done (set MODEL=1 to also serve DeepSeek-V4-Flash EXL3)"
  exit 0
fi

if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  section "using the checkpoint already in $MODEL_DIR (no download)"
  python - <<EOF
import json, pathlib
d = pathlib.Path("$MODEL_DIR")
want = set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())
missing = sorted(f for f in want if not (d / f).is_file())
print("missing shards:", missing if missing else "none")
raise SystemExit(1 if missing else 0)
EOF
  [ $? -eq 0 ] || exit 1
else
  section "download $BRANCH -> $MODEL_DIR"
  python -c "from huggingface_hub import snapshot_download as d; \
d('turboderp/DeepSeek-V4-Flash-0731-exl3', revision='$BRANCH', local_dir='$MODEL_DIR')"
fi
section "MTP drafter $DRAFTER (3.5 GB, the Hugging Face cache)"
python -c "from huggingface_hub import snapshot_download as d; d('$DRAFTER')"
du -sh "$MODEL_DIR"

section "tensorfold info"
tensorfold info "$MODEL_DIR" 2>&1 | tail -20

section "serve (log: exl3-serve.log)"
tensorfold serve "$MODEL_DIR" --port "$PORT" --drafter "$DRAFTER" > exl3-serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "server exited"; tail -60 exl3-serve.log; exit 1; fi
  sleep 5
done
grep -E "deepseek_v4|exact window|memory|GiB|wired|Error|error" exl3-serve.log | tail -20
NAME=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")

section "greedy sample"
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\": \"$NAME\", \
\"max_tokens\": 120, \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"In three sentences, why is \
the sky blue?\"}], \"chat_template_kwargs\": {\"enable_thinking\": false}}" | python -c \
"import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content']); print(r.get('usage'))"

section "decode speed"
python tools/bench_openai.py "http://127.0.0.1:$PORT" "$NAME" --tokens 256 --reps 3 --output exl3-bench.json
grep -E "Error|Traceback" exl3-serve.log | tail -10
section "done"
