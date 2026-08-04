# Future-L1 smoke status

Repository revision: `a91add2acc95bd0bdd609219da43df0af6e979db`.

This report distinguishes source-level checks runnable on the current macOS workspace from GPU/model/data checks that must run in the target RunPod environment.

| Smoke | Status | Exact command | Result / next fix |
|---|---|---|---|
| Repository checkout | PASS | `git clone --depth 1 https://github.com/OpenGVLab/Future-L1.git Future-L1` | Official source checked out at the revision above. |
| Repository/code map | PASS | Source inspection with `rg`, `sed`, and `nl` | SFT, RL, eval, latent alignment, recursion, reward, and data paths mapped in `reports/future_l1_repo_map.md`. |
| Script syntax/compile | PASS | `bash -n scripts/smoke_env_future_l1.sh scripts/smoke_grep_latent_paths.sh && python -m py_compile scripts/smoke_dataset_sft.py scripts/smoke_forward_sft.py` | Both shell scripts parse and both Python scripts compile. |
| Latent path grep | PASS | `bash scripts/smoke_grep_latent_paths.sh` | Wrote `reports/grep_latent_paths.txt` (503 lines). |
| FutureBench 8-sample eval replay | PASS: RUNPOD | `python -m lmms_eval --model qwen3_vl ... --tasks futurebench_pilot8` | Qwen3-VL-8B-Instruct scored 5/8 = 62.5%; all eight videos were readable and all responses parsed as A/B/C/D. |
| FutureBench 80-sample eval replay | PASS: RUNPOD | `python -m lmms_eval --model qwen3_vl ... --tasks futurebench_pilot80` | Deterministic balanced subset (20/type): 51/80 = 63.75%, Wilson 95% CI 52.8–73.4%. Paper balanced macro is 63.0%; delta +0.75 pp. Per type: hop1 50%, hop2 75%, hop3 65%, hop5 65%. |
| Environment/import smoke | FAIL: ENVIRONMENT | `bash scripts/smoke_env_future_l1.sh` | Current macOS Python 3.13.5 lacks torch, transformers, datasets, decord, qwen-vl-utils, and CUDA. Re-run in the target SFT environment; no installation was attempted locally. |
| Dataset schema-only smoke | PASS | `python scripts/smoke_dataset_sft.py` | Printed the expected TwiFF schema and cleanly marked data/collator checks SKIP. |
| Tiny TwiFF smoke-data generator | PASS: SOURCE | `python scripts/make_tiny_twiff_from_futurebench.py --help` | Added a deterministic generator that uses a real local FutureBench video with synthetic TwiFF supervision. It validates frame indices, video presence, options, and answer format. |
| Synthetic TwiFF video generator | PASS: SOURCE SYNTAX; RUNPOD PENDING | `python scripts/make_tiny_twiff_synthetic.py --output-dir /tmp/future_l1_tiny` | Generates a moving-square MP4 plus one TwiFF record without benchmark downloads. The script compiles locally; runtime validation is deferred to RunPod because local macOS lacks OpenCV. |
| Dataset normalized-load smoke | BLOCKED: INPUT | `python scripts/smoke_dataset_sft.py --data-path /path/to/tiny.json` | Needs a real tiny TwiFF JSON plus referenced video. |
| Dataset collator smoke | PASS: RUNPOD | `python scripts/smoke_dataset_sft.py --data-path /workspace/data/future_l1_tiny/train.json --model-path /root/local-checkpoints/Qwen3-VL-8B-Instruct` | Future-L1 chat template restored assistant-side image placeholders. Observed and teacher tensors both had shape `(256, 1536)`; latent token count and `image_out_mask` count both equaled 64. |
| SFT forward/loss smoke | PASS: RUNPOD | `python scripts/smoke_forward_sft.py --model-path /root/local-checkpoints/Qwen3-VL-8B-Instruct --data-path /workspace/data/future_l1_tiny/train.json` | Direct-to-GPU load from container-local checkpoint completed in about 2 seconds. Finite losses: total `4.6841321`, CE `3.5513196`, latent MSE `5.65625`, with latent weight `0.2`. No Trainer or save path. |
| SFT backward smoke | PASS: RUNPOD | `python scripts/smoke_forward_sft.py --model-path /root/local-checkpoints/Qwen3-VL-8B-Instruct --data-path /workspace/data/future_l1_tiny/train.json --backward` | Backward completed with 399 finite gradient tensors and 0 non-finite gradient tensors. No optimizer step and no checkpoint save. |
| One-step Trainer/update smoke | PASS: RUNPOD | `bash scripts/smoke_train_step_1gpu.sh` | One stateless-SGD update completed on one A100 in 1.42 s: loss `4.7507`, grad norm `137.0`, learning rate `1e-5`, epoch `1.0`. Final full-model saving was skipped as intended. This validates Trainer/update plumbing only; it is not the paper's AdamW + multi-GPU ZeRO configuration. |
| Dense video-attention scaling profile | PASS: RUNPOD | `python scripts/profile_video_attention.py --model-path /root/local-checkpoints/Qwen3-VL-8B-Instruct --video-path /workspace/data/future_l1_tiny/moving_red_square.mp4` | At fixed 224x224 resolution, 4/8/16/32 frames produced 98/196/392/784 video tokens and 141/255/483/939 total sequence tokens. Relative theoretical dense-attention pairs grew 1.00/3.27/11.73/44.35x. Peak forward allocation deltas were 0.020/0.040/0.079/0.158 GiB. Single-run prefill timings were noisy at this small scale (0.067–0.169 s), so they should not yet be interpreted as a scaling curve. |
| Temporal latent-memory mechanism smoke | PASS: RUNPOD | `python scripts/smoke_temporal_latent_memory.py --steps 500 --batch-size 64 --eval-samples 2048` | With 16 global tokens per method, hierarchical memory reached 100.0% held-out synthetic accuracy versus 31.88% for uniform token selection; shuffling clip memory reduced accuracy to 34.88%. Gain over the uniform baseline was +68.12 pp. This establishes feature-level mechanism learnability, not real-video performance. |
| Frozen-Qwen visual-feature memory smoke | PASS: RUNPOD | `python scripts/smoke_temporal_memory_qwen_features.py --model-path /root/local-checkpoints/Qwen3-VL-8B-Instruct --steps 500 --batch-size 32 --eval-samples 1024` | On frozen 4096-d Qwen3-VL features reduced to 128-d, hierarchical memory reached 76.76% held-out temporal-composition accuracy versus 40.14% for equal-16-token uniform selection. Shuffling clip memory reduced accuracy to 39.61%; gain over uniform was +36.62 pp. Train/test reused 20 rendered visual prototypes, so natural-video/domain generalization remains untested. |
| Frozen-Qwen decoder memory injection | RETRY: UNDERTRAINED WARMUP | `python scripts/smoke_temporal_memory_qwen_decoder.py --model-path /root/local-checkpoints/Qwen3-VL-8B-Instruct` | First run stayed at chance: normal 26.37%, shuffled 28.12%, zero-memory 24.41%. Its prerequisite warmup loss was `1.3203` (near random `ln(4)=1.3863`), versus `0.6527` in the passing visual-feature test. Retry separates a 500-step batch-32 warmup from batch-8 decoder training, requires >=60% held-out warmup accuracy, and exposes a question-conditioned read slot to the frozen decoder. |
| Final-save guard | PASS: SOURCE | `FUTURE_L1_SKIP_FINAL_SAVE=1 ...` | Added a minimal guard in `src/train/train.py`; trainer state still saves, while final full-model serialization is skipped. |
| `R_ctr` source-level smoke | PASS | Inline Python assertions against `outcome_contrastive_latent_reward.py` | Identity similarity, mixed positive/negative reward, and UID-grouped batch reward passed; sample rewards were `[0.8808, 0.8808, 0.1192]`. |
| `R_div` source-level smoke | PASS | Inline Python assertions against `_temporal_latent_diversity_stats` | Two orthogonal two-token blocks produced one adjacent pair and mean cosine-squared penalty `0.0`. |
| Full SFT training | DEFERRED | `bash scripts/train_twiff.sh` | Official launcher assumes 8 GPUs. Create a separate one-GPU tiny launcher only after forward/backward pass and set `FUTURE_L1_SKIP_FINAL_SAVE=1`. |
| RL component smoke | DEFERRED | Targeted reward/rollout tests under `RL_v2` | Start with reward unit tests and imports; do not launch Ray/FSDP/vLLM training yet. |
| Full LA-DAPO/DePO training | DEFERRED | `cd RL_v2 && bash train.sh depo_ctr` | Checked-in config assumes 8 GPUs, rollout `n=8`, and an optional external judge. |

## Target RunPod sequence

FutureBench replay is complete. Resume with the TwiFF SFT component path:

```bash
cd /workspace/Future-L1
bash scripts/smoke_env_future_l1.sh | tee reports/env_report_runpod.txt
bash scripts/smoke_grep_latent_paths.sh
python scripts/make_tiny_twiff_from_futurebench.py \
  --futurebench-json /workspace/data/V1-33K/futurebench_pilot80.json \
  --video-root /workspace/data/V1-33K \
  --output /workspace/data/future_l1_tiny/train.json
python scripts/smoke_dataset_sft.py \
  --data-path /workspace/data/future_l1_tiny/train.json \
  --model-path /workspace/checkpoints/Qwen3-VL-8B-Instruct \
  | tee reports/dataset_smoke_runpod.txt
python scripts/smoke_forward_sft.py \
  --data-path /workspace/data/future_l1_tiny/train.json \
  --model-path /workspace/checkpoints/Qwen3-VL-8B-Instruct \
  | tee reports/forward_smoke_runpod.txt
```

If forward passes, repeat the last command with `--backward`. Do not enable model downloads, Trainer construction, DeepSpeed, W&B, evaluation, or checkpoint saving during these first checks.
