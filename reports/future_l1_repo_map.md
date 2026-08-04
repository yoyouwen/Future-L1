# Future-L1 repository map

Inspected revision: `a91add2acc95bd0bdd609219da43df0af6e979db` (2026-06-12).

## Repository structure

- `src/`: SFT datasets, collators, model wrappers, monkey-patched forward functions, trainer, and launch-time configuration.
- `scripts/`: SFT launchers and DeepSpeed configurations.
- `RL_v2/`: EasyR1/verl-based GRPO, DAPO, and DePO training; Future-L1 vLLM latent rollout patches; reward functions.
- `lmms-eval/`: Future-L1 model adapter, FutureBench/TwiFF-Bench tasks, evaluation launchers, and latent visualization/export tools.
- `chat_template.json`: Qwen chat template expected by the SFT and RL paths.

## Entrypoints

### SFT

- Launcher: `scripts/train_twiff.sh`
- Python entrypoint: `src/train/train.py`
- Official defaults: Qwen3-VL-8B-Instruct, 8 local processes, batch size 1/device, global batch 128, bf16, DeepSpeed ZeRO-2, 16 observed video frames, 32 latent tokens, MSE latent loss with weight 0.2.
- Official freeze policy: vision tower frozen, merger frozen, LLM trainable. The optional projection head is disabled unless explicitly requested.
- The upstream revision saved the full model unconditionally. This smoke branch adds a minimal `FUTURE_L1_SKIP_FINAL_SAVE=1` guard around only the final full-model save; `trainer.save_state()` remains active. The isolated forward smoke avoids constructing a Trainer and never enters either save path.

### RL

- Launcher: `RL_v2/train.sh MODE`, where `MODE` is one of `grpo`, `dapo`, `depo`, `grpo_ctr`, `dapo_ctr`, or `depo_ctr`.
- Python entrypoint: `RL_v2/verl/trainer/main.py`.
- Default configuration: `RL_v2/examples/config_future_l1.yaml`.
- Reward entrypoint: `RL_v2/examples/reward_function/future_l1_reward_function.py:compute_score`.
- The checked-in config requests 8 GPUs and rollout `n=8`; this is not an appropriate first smoke on one A100.

### Evaluation

- FutureBench: `lmms-eval/examples/eval_futurebench_future_l1.sh` → task `futurebench_future_l1`.
- TwiFF-Bench: `lmms-eval/examples/eval_twiffbench_future_l1.sh` → task `twiffbench_future_l1`.
- Model adapter: `lmms-eval/lmms_eval/models/future_l1.py`.
- Both official launchers default to 8 processes. FutureBench uses rule-based task metrics; TwiFF-Bench can use an OpenAI-compatible judge configured through `lmms-eval/.env`.

## Stage-1 latent implementation

### Tokens

`src/constants.py` defines:

- `<|latent_start|>`
- `<|latent|>`
- `<|latent_end|>`

`src/train/train.py:190-208` adds the three strings to the tokenizer, stores their IDs in the model config, and resizes token embeddings when needed. The tokens are added with `special_tokens=False`.

### Teacher target and placeholder injection

The teacher is not a separate frozen encoder. Future-L1 reuses the model's own vision tower:

1. `TwiFFDataCollator` extracts question frames and assistant-side `reasoning_image` frames from the source video.
2. Question frames become `pixel_values`/`image_grid_thw`.
3. Future visual hints become `pixel_values_latent`/`image_grid_thw_latent`.
4. Assistant image placeholder interiors are rewritten to `<|latent|>` by `src/dataset/data_utils.py:647-724`.
5. In `src/train/monkey_patch_forward.py:472-493` (Qwen2.5) and the corresponding Qwen3/Qwen3.5 branches, `self.get_image_features` encodes the future frames and `masked_scatter` injects those embeddings into the latent-token input positions.

An offline alternative, `latent_target_embeds`, can bypass vision encoding, but it is mutually exclusive with `pixel_values_latent`.

### Alignment loss and shift

The Qwen2.5 implementation is at `src/train/monkey_patch_forward.py:676-699`; Qwen3 is at `1145-1168`; Qwen3.5 is at `1277-1299`.

For the latent mask `M`, it computes:

```text
prediction = projected_or_raw_hidden_states[:, :-1][M]
target     = injected_input_embeddings[:, 1:][M]
L_latent   = MSE(prediction, target)          # or 1 - cosine similarity
L_total    = L_CE + latent_lambda * L_latent
```

This is the expected autoregressive shift: hidden state `h_t` predicts the next injected future-frame embedding `e_{t+1}`.

The target slice is **not explicitly detached**. Under the official launcher the vision tower and merger have `requires_grad=False`, so the teacher encoder does not update. If those modules are unfrozen, latent-loss gradients can reach the target branch; this should be treated as an intentional configuration choice or changed explicitly for a fixed-teacher experiment.

Text CE labels retain assistant text after `<|im_start|>assistant` while masking latent filler tokens (`src/dataset/data_utils.py:741-782`). Thus the default CE includes textual reasoning and answer text, not only the final answer.

### Inference recursion and diagnostics

- Generation state machine: `src/model/future_l1.py:_future_l1_sample`.
- While inside a latent span, the previous projected/raw hidden state replaces the next `<|latent|>` input embedding (`src/train/monkey_patch_forward.py:437-439` and corresponding Qwen3 branches).
- Generated latent trajectories and masks are already attached to return-dict generation outputs by `src/model/future_l1.py`.
- lmms-eval can export latent embeddings from `lmms-eval/lmms_eval/models/future_l1.py`; plotting tools include `plot_future_l1_latent_umap.py`, `plot_future_l1_latent_blocks.py`, and `plot_future_l1_mirage_fig7.py`.

This means the proposed first diagnostic hooks—span count, span length, hidden dimension, and trajectory export—largely exist already and should be validated before adding new instrumentation.

## Dataset schema

`TwiFFSFTDataset` accepts two source formats.

### `twiff_frames`

```json
{
  "conversations": [
    {"from": "human", "value": "<image> What happens next?"},
    {"from": "gpt", "value": "THOUGHT 1: ... <image> ..."}
  ],
  "video": "relative/or/absolute/video.mp4",
  "image": [1],
  "reasoning_image": [2],
  "answer": "final answer",
  "cot": "optional text-only CoT"
}
```

`image` and `reasoning_image` are 1-based indices into TwiFF's uniformly sampled frame pool. Relative video paths are resolved relative to the JSON file. Each `<image>` in the human turn consumes a question frame; each `<image>` in the GPT turn consumes a future/reasoning frame.

### `chat_video_distill`

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<video> ..."},
    {"role": "assistant", "content": "..."}
  ],
  "videos": ["video.mp4"],
  "images": []
}
```

This format supplies ordinary video distillation data and need not contain latent future-frame supervision.

The latent-supervised collator emits:

- `input_ids`, `attention_mask`, `labels`
- question evidence: `pixel_values`, `image_grid_thw`
- teacher/future evidence: `pixel_values_latent`, `image_grid_thw_latent`
- `image_out_mask` identifying latent targets
- processor-specific multimodal fields when applicable

## LA-DAPO, DePO, and reward map

### Rollout and continuous latent likelihood

- vLLM state machine: `RL_v2/future_l1_rl/vllm_runner/future_l1_gpu_model_runner.py`.
- Per-step transport/recording: `latent_hook.py` and `latent_recorder.py`.
- Rollout integration: `RL_v2/verl/workers/rollout/vllm_rollout_spmd.py:315-410`.
- Actor latent mask and vMF-style log probability: `RL_v2/verl/workers/actor/dp_actor.py:142-198` and its actor forward path.
- `future_l1_depo` is the latent-aware rollout mode used by GRPO, DAPO, and DePO modes; DePO-specific loss splitting remains separately gated.

### Decoupled policy loss

`RL_v2/verl/workers/actor/dp_actor.py:613-660` splits response positions into text and latent masks, applies ordinary clipping to text positions and tighter latent clipping to latent positions, then combines:

```text
L_policy = L_token + latent_loss_alpha * L_latent
```

Launcher defaults for DePO are latent clip low/high `0.1/0.1`, dual clip `3.0`, and latent loss weight `0.5`. Optional latent vMF KL is implemented in `RL_v2/verl/trainer/core_algos.py:622-650` and applied by `dp_actor.py:698-715`.

### `R_ctr`

- Definition: `RL_v2/examples/reward_function/outcome_contrastive_latent_reward.py`.
- It groups rollouts by `uid` (fallback: problem + ground truth), normalizes trajectories, compares time-aligned latent steps, and rewards similarity to correct trajectories relative to wrong trajectories.
- It is composed into the batch reward at `future_l1_reward_function.py:505-535`.
- `*_ctr` launcher modes default to coefficient `1.0` and temperature `0.5`, unless overridden.

### `R_div`

- Definition/composition: `RL_v2/examples/reward_function/future_l1_reward_function.py:350-394,527-535`.
- Each completed latent block is treated as a keyframe, mean-pooled, and compared with its adjacent block. The penalty is the mean squared cosine similarity, so minimizing it encourages temporal diversity.
- The checked-in YAML default is disabled (`latent_div_lambda: 0.0`). It can be enabled through reward overrides or `FUTURE_L1_LATENT_DIV_LAMBDA`.

Overall reward in code is:

```text
(1-format_weight)*accuracy
+ format_weight*format
- length_penalty_weight*length_penalty
- latent_div_lambda*mean_adjacent_block_cosine_squared
+ latent_ctr_lambda*R_ctr
```

## Immediate adaptation implications

1. The existing latent spans are frame/keyframe aligned, making them a useful baseline for temporal latent diagnostics.
2. The current compressor is the backbone vision encoder plus its merger, not a dedicated spatio-temporal compressor.
3. Latent export already supports the first proposed visualization experiments.
4. A long-video extension should first preserve and annotate span-to-frame/clip correspondence, then introduce explicit temporal windows or operation types.
5. Persistent memory is not present in this code map; it remains a later isolated prototype rather than part of the reproduction smoke.
