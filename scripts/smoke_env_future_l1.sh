#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUTURE_L1_ROOT="${FUTURE_L1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PYTHONPATH="${FUTURE_L1_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Future-L1 environment smoke"
echo "repo_root=${FUTURE_L1_ROOT}"
echo "python=$(command -v python || true)"
python --version 2>&1 || true

python - <<'PY'
import importlib
import platform
import sys

print("platform", platform.platform())
print("python_executable", sys.executable)

required = [
    "torch",
    "transformers",
    "datasets",
    "decord",
    "qwen_vl_utils",
    "PIL",
]
optional = ["accelerate", "deepspeed", "flash_attn", "ray", "vllm"]
source_modules = [
    "src.constants",
    "src.dataset.data_utils",
    "src.model.projection_head",
]

failed_required = []
for name in required + optional + source_modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "n/a")
        category = "required" if name in required else ("optional" if name in optional else "source")
        print(f"PASS import {name} category={category} version={version}")
    except Exception as exc:
        category = "required" if name in required else ("optional" if name in optional else "source")
        print(f"FAIL import {name} category={category} error={exc!r}")
        if name in required or name in source_modules:
            failed_required.append(name)

try:
    import torch

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("torch_cuda", torch.version.cuda)
    print("cuda_device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"cuda_device_{idx} name={props.name!r} "
                f"memory_gib={props.total_memory / (1024 ** 3):.2f}"
            )
except Exception as exc:
    print("torch_diagnostics_failed", repr(exc))

if failed_required:
    print("OVERALL FAIL missing_or_broken=" + ",".join(failed_required))
    raise SystemExit(1)
print("OVERALL PASS")
PY
python_status=$?

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "SKIP nvidia-smi: command not found"
fi

exit "${python_status}"
