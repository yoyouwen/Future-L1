#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUTURE_L1_ROOT="${FUTURE_L1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPORT_DIR="${FUTURE_L1_ROOT}/reports"
REPORT_PATH="${REPORT_DIR}/grep_latent_paths.txt"
mkdir -p "${REPORT_DIR}"

if ! command -v rg >/dev/null 2>&1; then
    echo "ERROR: ripgrep (rg) is required for this source-map smoke." >&2
    exit 1
fi

cd "${FUTURE_L1_ROOT}"
{
    echo "Future-L1 latent path source map"
    echo "repo_root=${FUTURE_L1_ROOT}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo
    echo "## SFT latent tokens, targets, injection, recursion, and losses"
    rg -n --glob '*.py' --glob '*.sh' \
        'latent_start|latent_end|latent_target_embeds|pixel_values_latent|image_out_mask|latent_hidden_state|masked_scatter|mse_loss|latent_lambda' \
        src scripts || true
    echo
    echo "## RL rollout, latent likelihood, DePO, and rewards"
    rg -n --glob '*.py' --glob '*.yaml' --glob '*.sh' \
        'R_ctr|R_div|latent_ctr|latent_div|future_l1_depo|latent_log_probs|latent_clip|latent_mask|LatentRecorder|latents_array' \
        RL_v2 || true
    echo
    echo "## Evaluation and latent export"
    rg -n --glob '*.py' --glob '*.yaml' --glob '*.sh' \
        'future_l1|latent_states|latent_mask|latent_export|mirage_export' \
        lmms-eval/examples lmms-eval/lmms_eval/models lmms-eval/lmms_eval/tasks lmms-eval/tools || true
    echo
    echo "## Dataset schemas and collators"
    rg -n --glob '*.py' \
        'Expected JSON fields|reasoning_image|source_format|TwiFFDataCollator|FutureL1DataCollator|make_supervised_data_module' \
        src/dataset || true
} > "${REPORT_PATH}"

echo "PASS wrote ${REPORT_PATH}"
echo "lines=$(wc -l < "${REPORT_PATH}" | tr -d ' ')"
