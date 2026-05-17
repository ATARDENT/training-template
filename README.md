# Training Template

A standardised template for fine-tuning ML models — from dataset ingestion through training, evaluation, and versioned model storage. Supports Colab, ThunderCompute, and AWS spot as compute backends, in that priority order.

---

## Repository structure

```
.
├── scripts/
│   ├── main.py               # Main training script — called by the pipeline
│   └── stratify.py           # Custom stratification logic for data splits
│
└── manifest.yaml             # Pipeline control (see below)
```

---

## Configuration

```yaml
version: "1.0.0"

model:
  name: "meta-llama/Llama-3.2-3B"   # HF hub ID, local path, or null to train from scratch
  task: "sft"                         # sft | dpo | ppo | classification | regression | ...
  trust_remote_code: false
  attn_implementation: "sdpa"         # eager | sdpa | flash_attention_2
  quantization:
    enabled: false
    bits: 4                           # 4 | 8
    type: "nf4"                       # nf4 | fp4
    double_quant: true
    compute_dtype: "bfloat16"

tokenizer:
  name: null                          # null = inherit from model.name
  max_length: 2048
  padding_side: "right"
  chat_template: null                 # "llama3" | "chatml" | null

dataset:
  deputy: "<dataset-deputy-uuid>"     # Deputy UUID of the compiled dataset
  format: "jsonl"                     # jsonl | csv | parquet | huggingface | custom
  columns:
    input: "prompt"
    target: "response"
    label: "category"
  preprocessing:
    template: null
    filter_script: null
    max_samples: null
    shuffle_seed: 42

split:
  train: 0.8
  val:   0.1
  test:  0.1
  stratified:
    enabled: true
    label_column: "category"
    num_bins: 5
    custom:
      classify_script: "scripts/stratify.py"

training:
  max_epochs: 10
  max_steps: null
  save_mode: ["checkpoint", "lora"]   # list — one or both
  run_test_after: true
  precision: "bf16"                   # fp32 | fp16 | bf16
  gradient_checkpointing: true

  hyperparameters:
    learning_rate: 0.0002
    batch_size: 4
    gradient_accumulation_steps: 8
    optimizer:
      type: "adamw_torch"
    scheduler:
      type: "cosine"
      warmup_ratio: 0.03

  checkpoints:
    enabled: true
    top_k: 3
    monitor: "val_loss"
    lora:
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: [q_proj, v_proj, k_proj, o_proj]

  early_stopping:
    enabled: true
    monitor: "val_loss"
    patience: 3

  monitoring:
    backend: "tensorboard"            # tensorboard | wandb | comet | mlflow | none
    project_name: "my-training-project"

backend:
  primary:
    - name: colab
      config:
        instance_type: "A100"
      max_runtime_hours: 12

  fallback:
    - name: thundercompute
      config:
        instance_type: "A100"
        num_gpus: 1
        spot: true
      max_runtime_hours: 12
      max_cost_usd: 20

    - name: aws
      config:
        instance_type: "g4dn.xlarge"
        region: "us-east-1"
        spot: true
      max_runtime_hours: 12
      max_cost_usd: 50

  maximum_runtime_hours: 24
  retry_on_failure: 1

output:
  name: "my-model"
  destination: "idrive_e2e"          # idrive_e2e | gdrive | huggingface
  config:
    idrive_e2e:
      path: "models/"
    gdrive:
      path: "My Drive/models/"
    huggingface:
      repo_name: "my-username/my-model-repo"
      private: true

tag:
  enabled: true
  prefix: "model"                    # tag format: <prefix>-v<version><suffix>
  suffix: ""
  github_branch: "main"
  dvc_file: "model.dvc"

notifications:
  enabled: true
  events:
    on_success: true
    on_failure: true
    on_early_stop: true
    on_backend_fallback: true
  channels:
    - type: email
      config:
        recipients:
          - "you@example.com"
    # - type: slack
    # - type: discord

reproducibility:
  deterministic: false
  benchmark: true
  global_seed: 42

variables:
  experiment_id: "exp-001"
  notes: "Baseline run"
```

---

## Pipeline

![alt text](image.png)

### Steps

**Step 1 — Clone dataset**
Pull the compiled dataset using the configured `deputy` UUID.

**Step 2 — Validate dataset**
Integrity check — schema, row counts, no file corruption. Aborts on failure.

**Step 3 — Split data**
Partition into train / val / test sets using the ratios in `manifest.yaml`. Supports optional stratified splitting via `split.stratified`, with a custom classify script at `scripts/stratify.py`.

**Step 4 — Setup environment**
Provision the compute backend (in priority order: Colab → ThunderCompute → AWS). Install dependencies and verify GPU availability. Falls back to the next backend after `max_runtime_hours` or `max_cost_usd` is reached, with `retry_on_failure` transient retries per backend.

**Step 5 — Resume check**
Look for an existing checkpoint matching the current `version`. If found, training resumes from it rather than starting over.

**Step 6 — Train**
Run `scripts/main.py` on the train + val split. Loops until `max_epochs` or `max_steps` is reached. Supports early stopping via `training.early_stopping`.

**Step 7 — In parallel: log metrics + wait for approval**
While training runs, metrics (loss, eval, learning rate) are streamed to the configured monitoring backend. Simultaneously, the pipeline waits for a user approval decision:

| Outcome | Behaviour |
|---|---|
| 7.1 Training finishes naturally | Auto-success, proceed to Step 8 |
| 7.2 User approves | Interrupt training → save checkpoint → proceed to Step 8 |
| 7.3 User rejects | Abort. Nothing is saved or tagged. |
| 7.4 Spot revocation (ThunderCompute / AWS) | Emergency checkpoint save → skip Steps 9–10 → proceed to Step 11 |

**Step 8 — Tag & store model**
Save the model artefact (full checkpoint, LoRA adapter, or both per `save_mode`) to the configured destination.

**Step 9 — Run test data**
Evaluate the stored model on the held-out test split. Skipped if `run_test_after: false` or if triggered by spot revocation.

**Step 10 — Store results**
Push the evaluation metrics report alongside the model artefact.

**Step 11 — Tag version**
Create a Git tag in the format `<prefix>-v<version><suffix>` (e.g. `model-v1.0.0`) on the configured branch. Marks the artefact as permanently reproducible.

---

## CI behaviour summary

| Event | Steps run |
|---|---|
| Push / pull request | Steps 1–2 only (clone + validate) |
| Merge to main | Full pipeline (Steps 1–11) |
| Spot revocation | Steps 1–8, 11 (test skipped, model still saved) |

---

## Implementing this template

1. **Point at your dataset** — set `dataset.deputy` to the UUID of your compiled dataset deputy, and `dataset.format` to match its format.
2. **Configure your model** — set `model.name` to a HF hub ID or local path, and `model.task` to match your training paradigm (`sft`, `dpo`, `classification`, etc.).
3. **Configure your split** — adjust `split` ratios (must sum to 1.0). Enable `split.stratified` and point `classify_script` at `scripts/stratify.py` if you need stratified sampling.
4. **Write `scripts/main.py`** — the script receives split paths and config values as arguments. It should checkpoint regularly and handle `SIGTERM` for spot revocation.
5. **Tune hyperparameters** — fill in `training.hyperparameters` including optimizer, scheduler, and LoRA config under `training.checkpoints.lora` if using LoRA.
6. **Choose your compute** — fill in the relevant `backend.primary` and `backend.fallback` blocks. The pipeline tries each backend in priority order, respecting `max_runtime_hours` and `max_cost_usd` caps.
7. **Set `save_mode`** — use `["lora"]` to store only the adapter weights, `["checkpoint"]` for the full model, or both.
8. **Configure output** — set `output.destination` to `idrive_e2e`, `gdrive`, or `huggingface` and fill in the matching sub-block under `output.config`.
9. **Set up notifications** — add recipients or webhook URLs under `notifications.channels`.
