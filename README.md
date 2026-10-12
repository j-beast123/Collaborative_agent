# SHERPA online — anchored group learning

## Prompt optimization on this machine

Use `run_prompt_optimization.sh` for the current offline generator-prompt search.
The Python environment is `.venv`, inheriting the host's B200-compatible NVIDIA
PyTorch; the additional pinned packages are in `requirements-prompt.txt`.
The local Qwen3-8B model is loaded from `../vllm/qwen3` via
`policy.local_model_path`, while its logical model identity remains
`Qwen/Qwen3-8B`. No model download is needed.

Fill `OPENAI_API_KEY` and `OPENROUTER_API_KEY` in `.env` locally, then run:

```bash
bash run_prompt_optimization.sh                     # local preflight, no API calls
bash run_prompt_optimization.sh --launch             # base rollouts, then prompt search
```

Optional one-case smoke run and stage-specific resume:

```bash
bash run_prompt_optimization.sh --launch --phase base-rollouts --max-cases 1
bash run_prompt_optimization.sh --launch --phase base-rollouts
bash run_prompt_optimization.sh --launch --phase optimize-prompt
```

The default visible GPUs are `0,1`; override `CUDA_VISIBLE_DEVICES` to select a
different pair. Base collection now uses **vLLM 0.31.0**, one original Qwen3-8B
model per GPU, with `--gpu-memory-utilization 0.9`, tensor parallel size 1,
`--max-num-seqs 128` per server and a **global 128-request limit**. Four cases
(each with 32 branches) overlap; vLLM continuously batches waiting requests.
Servers bind only to localhost, start automatically, and stop after collection.
Logs are `vllm_gpu0.log` / `vllm_gpu1.log` in the experiment output directory.
The server environment `.venv-vllm` is isolated from the existing GRPO `.venv`:

```bash
uv venv --python /usr/bin/python3 .venv-vllm
uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt
```

Doctor is now `gpt-4o-mini`, with no `reasoning` parameter in the API request
(non-reasoning model). Doctor temperature stays 0.4 and its global API limit is
50. Judge remains `gpt-6-luna`; the rubric generator remains Gemini.
The new experiment writes to `outputs/sherpa_prompt_auc_gpt4o_mini_vllm_v1`.
Previous Luna-doctor results in `outputs/sherpa_prompt_auc_v1` are preserved;
changing Doctor starts fresh, without importing those outcomes.

Prompt phases set `rate_limit_retries: 0`: rate-limit errors propagate without
sleeping or resending. Other transient errors retain their retry policy.
Restart with the same command and output directory to reuse completed cases,
successful calls inside unfinished cases, and cached prompt evaluations.
Concurrency and retry settings do not invalidate these caches. Engine migration
can reuse validated original-base evidence, including the legacy disabled-adapter
cache namespace. New vLLM traces record `vllm_seeded_v1`: equal seeds do not imply
identical outputs across engines. This new Doctor experiment starts entirely
with vLLM. GRPO training continues to use Transformers/PEFT and its full traces.

Local-only GPU verification (no paid APIs, refuses GPUs locked by another run):

```bash
.venv/bin/python scripts/online_rl/smoke_vllm.py
```

After collection, both GPU models are released. Prompt search uses remote APIs
only, evaluates four candidates concurrently with four cases per candidate,
and shares the run-wide limits of
10 generator calls and 20 judge calls. It reuses 24 fixed mixed-outcome cases,
evaluates the initial prompt plus four candidates in each of three iterations,
then independently regenerates and rescores the initial and winning prompts once.
Confirmation uses the same cases, is not held-out evaluation, and never changes
the selected winner. Generator temperature is 0.2; rubric behaviors must belong
to the assistant, with partner state allowed only as a condition.
All cases share one judge worker pool; case parallelism does not multiply the
20-request judge limit. Cached case results, aggregation order and tie rules
remain unchanged. Changed runtime settings take effect after restart.

To compare the older Transformers sampler configurations on an idle GPU:

```bash
.venv/bin/python scripts/online_rl/benchmark_prompt_gpu.py --gpu 0 --trace-mode both
```

This uses the same batch sampler with 512 synthetic input tokens and 128 output
tokens, EOS disabled, one warm-up and three measured repetitions per setting.
Override `--input-tokens` and `--output-tokens` to match another benchmark.
It compares 1 model × 4/8/16 requests and 2 models × 4/8 requests on one GPU,
and writes median output tokens/s to `outputs/prompt_benchmark/`. It refuses to
run while that GPU's SHERPA lock is held and never changes production settings.
Synthetic throughput does not measure end-to-end dialogue/API latency.

Monitor an existing run without restarting it:

```bash
bash monitor.sh                         # automatically select prompt optimization
bash monitor.sh --mode optimization --details
bash monitor.sh --mode optimization --json
# Explicit directory, if a different output directory was used:
bash monitor.sh --mode optimization --output outputs/sherpa_prompt_auc_gpt4o_mini_vllm_v1
```

The monitor reads existing artifacts and incrementally tails completed API/cache
records. It shows case/FINAL progress, mixed-case selection, reflections,
candidate scores and partial judge counts, confirmation AUC differences, API
usage, and server GPU utilization. GPU utilization includes other processes.
`--mode training` retains the training monitor. Ctrl+C stops only the monitor.

Results and per-phase API usage logs are under `outputs/sherpa_prompt_auc_gpt4o_mini_vllm_v1/`.
The principal outputs are `final_prompt.txt`, `final_prompt.json`, `history.json`
and `confirmation.json`. Completed case/candidate evidence is reused on restart.
At least 24 of the 120 base cases must have both successful and failed trajectories.
This workflow does not run GRPO training or upload model checkpoints.

To recreate the environment on this NVIDIA image without replacing host PyTorch:

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python --no-deps -r requirements-prompt.txt
bash run_prompt_optimization.sh
```

## Ours: optimized-rubric GRPO

The independent `configs/ours.json` now uses a frozen **Qwen/Qwen3-8B Doctor**.
GPU 0 runs the trainable assistant (Transformers/LoRA, batches up to 64 replies,
global queued/executing assistant request limit 128); GPU 1 runs the separate
original-base Doctor through vLLM with memory utilization 0.9. Doctor thinking
is disabled, temperature is 0.4, and discussion, FINAL and reflection all use
this same frozen Doctor, with a global limit of 128 local Doctor requests. Its weights never receive assistant GRPO updates.
The run writes to `outputs/ours_qwen3_doctor_256_compact_luna_rubric_v4`.

Doctor changes require a matching new prompt optimization experiment. The
existing GPT-4o-mini run/config/output is preserved. Start the Qwen experiment:

```bash
bash run_prompt_optimization.sh --config configs/sherpa_qwen_doctor.json --launch
bash monitor.sh --config configs/sherpa_qwen_doctor.json --mode optimization
```

This collects new Qwen-Doctor rollouts into
`outputs/sherpa_prompt_auc_qwen3_doctor_256_compact_luna_v4`, then optimizes the generator prompt.
Add `--no-cache` to regenerate calls and completed results in a new sibling
directory ending in `_nocache_<unique-id>`. Existing evidence is preserved.
With `--phase optimize-prompt --no-cache`, only completed base rollout data is
copied as search input; reflections, rubric generation, judging and proposals
are generated again. Successful call caches are neither read nor written;
failure audits and final result artifacts are still saved. The printed
`run_config.json` path can be passed to `monitor.sh --config` and contains the
new `final_artifact` path needed by any subsequent training config. This option
does not change seeds, the minimum mixed-case requirement or retry settings.
Both stages automatically start/stop the local Doctor server on GPU 1; even the
prompt-search phase uses this GPU for Doctor reflections. The assistant's base
rollouts use vLLM on GPU 0; GRPO uses the trainable Transformers model on GPU 0.
Generator, prompt proposer and judge all use gpt-6-luna through the direct OpenAI Responses API, sharing a 100-request limit and a 100-worker judge pool.
The v4 initial prompt derives case-specific binary collaboration items from
success/failure examples and Doctor reflections. Proposal feedback includes
misranked or tied conversation pairs and per-item outcome satisfaction rates.
The current prompt and the top two proposals per iteration are compared by
three evaluation passes; only finalists receive the two additional passes.
Three independent confirmation passes report initial/final differences on the
same 20 optimization cases without changing the selected winner. These are
repeatability checks, not held-out evaluation. The existing 120 base cases are
reused; v3 prompt scores are preserved in their original directory.

Each scheduled step overlaps four cases. Each case generates one Doctor opening
and branches into 32 assistant replies, then continues 32 conversations with the
current policy. All assistant generations use temperature=1, top_p=1, top_k=0.
The final optimized **generator prompt** is fixed for the entire run; each case
still gets fresh reflections and 5–6 case-specific rubrics on its current rollouts.
The reward is the mean of the binary rubric scores. The Doctor outcome is used
for reflection/diagnostics, not added to the GRPO reward. Advantages are z-scores
within each group of 32, and all four scored groups reach one update barrier.
Only assistant tokens receive the trajectory advantage; the base policy supplies
the KL reference. The trainable parameters are LoRA adapters, not the full base.

The pinned 120-case order is repeated three times: 30 four-case steps per
iteration, 90 scheduled steps total (11,520 trajectories). A batch with no reward
variation in any group is recorded but skips the optimizer because no advantage
signal exists. GPU replicas are explicitly synchronized before every step,
including after restoring a checkpoint. Recovery checkpoints allow interrupted
steps to reuse their saved calls without replaying a committed update.

```bash
bash run_ours.sh                          # preparation report, no GPU/API work
bash run_ours.sh --launch --max-steps 1    # optional first real training step
bash run_ours.sh --launch                 # resume/run to step 90
bash monitor.sh --config configs/ours.json --mode training
```

Launch requires the completed optimization artifact
`outputs/sherpa_prompt_auc_qwen3_doctor_256_compact_luna_v4/final_prompt.json`; it never falls back
to an initial/intermediate prompt. Its hash and generator/judge/Doctor identity
are validated, and the exact prompt is copied into the training output.
Do not start the real run while prompt optimization is still writing its final
artifact. The existing prompt optimization process is independent of this setup.

Every 10 steps a local checkpoint is saved and an immutable full recovery bundle
(LoRA adapter, optimizer/RNG state, metadata and frozen generator prompt) is queued
for a detached Hugging Face uploader. Fill `HF_TOKEN` and `HF_REPO_ID` in `.env`;
the target must be an existing model repository you can write to. The base model
weights are not re-uploaded. Upload status and errors live under `hf_uploads/`;
a committed training step is preserved even if an upload fails. Restart catches
up pending ten-step checkpoints. To run locally without uploads, explicitly set
`checkpoint_upload.enabled=false` in `configs/ours.json` before the first launch.

Local-only assistant test of 128 short synthetic replies and one GRPO update:

```bash
.venv/bin/python scripts/online_rl/smoke_ours.py
```

For a local-only test of structured Doctor turns, FINAL and reflection:

```bash
.venv/bin/python scripts/online_rl/smoke_local_doctor.py
```

The assistant test acquires its GPU locks and makes no Doctor/generator/judge calls. It
checks one optimizer step and configured replica synchronization; it is not a full dialogue
benchmark or a peak-memory test for maximum-length generations.

## Earlier anchored-group workflow notes

The notes below describe the earlier anchored-group experiments. The current
`run.sh` defaults to `configs/sherpa.json`; use the prompt workflow above for
offline prompt optimization.

The default configuration is `configs/group_a100_partner_state.json`.
Doctor uses `gpt-6-luna` on the OpenAI platform, with a shared run-wide limit
of 50 concurrent Doctor requests. Assistant uses `Qwen/Qwen3-8B`, sampled at
`temperature=1.0`. GPU 0 hosts the assistant and GRPO updater. The default
training uses 120 cases, eight-case batches and checkpoints every 32 cases.

```bash
bash run.sh                              # preparation checks only
bash run_a100.sh --launch                # start training
bash monitor.sh                          # monitor the current run
runtime/python scripts/online_rl/inspect_group_signal.py --batch 1
```

## Current learning loop

1. Generate one shared opening with Doctor `gpt-6-luna` through the OpenAI
   Responses API. Sample 32 assistant replies with `Qwen/Qwen3-8B` and
   `temperature=1.0`; continue each rollout independently without further forks.
   Each case produces one opening group and 32 natural trajectories. Existing
   halting rules and the four-turn limit apply independently to each path.
2. The partner privately records a hypothesis, favored option and confidence
   in [0,1] at every discussion turn, including the opening. Discussion uses an
   OpenAI Structured Outputs schema with separate `reply`, `status` and `private_state` fields, with
   dynamic option labels from each case. Confidence is constrained by a numeric
   enum to 0–1 in increments of 0.01, Only `reply` enters the dialogue. A
   malformed structured discussion gets one seeded format retry; missing states
   remain an error and are never fabricated. The visible reply is requested to
   stay within 60 words so the hidden fields fit the existing 256-token budget.
   These fields never
   enter the assistant's visible transcript. For every group the partner compares
   the shared prefix, its recorded state, 32 replies and their own natural
   outcomes. Reflection still runs when all outcomes are the same. Optional gold
   is controlled by `group_learning.show_gold_to_reflection` (default false).
   The partner writes natural first-person prose in one `reflection` string.
   OpenAI's per-call output schema enforces this single-field JSON object;
   the partner does not have to match source IDs or extract exact quotations.
   Replies are labeled Reply 1–32, separately from medical answer choices.
   A format/empty-output failure gets at most one retry with a different seed;
   raw attempts and errors are saved. No valid reflection is rewritten.
3. One generator call per group selects ONE sibling reply whose observed
   behavior was most decisive for the partner's final decision and describes
   that behavior as b. It does not classify it as helpful/harmful. A shared
   reinforcement can be represented by one reply when the partner reports no
   unique difference. The prompt requires reflection/outcome consistency:
   helpful behaviors must come from successful conversations, and hindering
   behaviors from failed ones. If none qualifies, return `{"candidates": []}`;
   this is a valid no-candidate result, not a format failure or verification
   rejection. It does not trigger regeneration, verification, or judging, and
   the group is logged as `no_candidate_rubric`. Unresolved outcomes cannot
   anchor a candidate. The generator supplies state,
   mutually exclusive observed/middle/contrast criteria, opposite behavior
   b_star, and a quote. CODE sets b's endpoint from that source trajectory:
   success -> 2, failure -> 0. The contrast gets the opposite endpoint and the
   middle gets 1. Numeric criteria and the internal direction label are therefore
   assigned by code. Interpretation of the prose judgment remains with the
   generator; code checks the source, quote, and outcome-based endpoints.
   A schema/field/ID/quote validation failure gets at most ONE regeneration,
   with the previous output and error supplied only to the generator. All raw
   attempts are saved. Verification failures do not trigger regeneration.
   Quotes must appear in the specified reply;
   validation allows Unicode normalization, case, whitespace and punctuation
   differences, without fuzzy word-overlap attribution or source-ID repair.
4. Generate one contrast reply per candidate at the original shared prefix;
   reuse that fixed reply in exactly three independently seeded continuations.
   `execution.rerun_workers=3` runs these three continuations concurrently;
   their results are stored in rerun-index order. Verification ends at that
   Doctor reply; it does not create further assistant replies or a FINAL.
   The default `next_partner_state_v1` verification checks the first Doctor
   private choice after the contrast reply. Keep the rubric when at least two
   of three trials choose the required correct/incorrect endpoint. Unknown outcomes are unresolved and receive no numeric s.
5. For each group use only its verified anchored rubrics. One blind judge call
   ranks all 32 replies against these rubrics from best to worst. Response order is
   shuffled with a reproducible seed and anonymous IDs. Public rubrics include
   only name, state and criteria; source IDs, outcomes, b and b_star are omitted.
   There is no applicability call. The judge explains rubric-grounded differences
   and permits ties only when it cannot meaningfully distinguish the responses.
   With multiple rubrics it gives each equal weight. `r(a)=(32-rank(a))/31`; first place gets 1 and last place gets 0. Ties receive the average of
   their occupied ranks (a tie for 2nd and 3rd gives each rank 2.5).
   32 indistinguishable responses all receive 0.5 and zero advantage.
   Ranking groups must cover all 32 shuffled IDs exactly once, and have a
   rubric-grounded reason. Invalid rankings exclude that group from learning.
   Groups without a verified rubric, or without a complete valid judge result,
   are excluded from the update.
6. Collect the batch's scored groups and compute `(r-mean)/(std+1e-8)` within each
   32-response group. Equal rewards yield zero advantage. The loss includes
   only each anchor reply's tokens, with its shared prefix as input. The group
   configuration samples one visible reply per action. Continuation and contrast
   replies never enter the loss. A batch with no scored groups skips the optimizer.

An assistant's `Final answer:` text alone does not stop the dialogue.
`mutual_conclusion` is disabled: conclusion phrases without a matching choice
are insufficient. Explicit `mutual_agreement` requires matching parsed choices
and conclusion signals from both participants. STATUS-based `doctor_halt` now
also requires the latest assistant's parsed option to equal the current Doctor
option, in addition to stable valid STATUS and all unresolved flags being false.
If the assistant's option is absent or different, this path continues. The same
rules apply to natural rollout and verification. The repeated-reply rule remains
at turn four, alongside the four-turn limit. These changes can extend dialogues;
matching, resolved discussions can still finish after two turns.
The default update batch contains eight cases and the learning rate remains `1e-5`.

### Private reflection boundary

Reflection uses a separate stateless partner `_chat` request with its own system
prompt and response schema. Its result goes to the rubric generator and private
case logs. Neither raw reflection nor structured observations are appended to
the visible dialogue, assistant system/user prompts, continuations, or GRPO
input traces. The schema is scoped to that call and does not persist into later
Doctor discussion or FINAL. Verification supplies only the candidate's `b_star`
behavior instruction to the one replacement reply. The blind judge receives
public rubric state/criteria and visible replies, without reflection or source
provenance. JSON schema constrains output structure; routing enforces this boundary.

Each case has 32 natural trajectories, one partner reflection, one generator
call, at most one contrast reply, three verification trials and at most one
reward judge call. Cached calls keep the observed API results and protocol identity.

## Group logs

Each case saves `group_rollout.json`, incremental `group_progress.json`,
`group_metrics.json` and its final case record. The latter contains every group's
source outcomes, raw partner reflection and its attempts, selected behavior,
code-assigned direction/endpoint, generation attempts, fixed contrast reply, three continuations,
s, KEEP/REMOVE result, anonymized judge request, scores and advantages.

The parent saves `group_metrics/batch_*.json` and `rubric_books/batch_*.json`, each
bound to its frozen source policy. Metrics include rubric group fraction,
zero advantage fraction among scored groups, verification pass rate among
resolved candidates, raw s rows and failure statuses, response token/character
lengths, initial partner state/correctness, reference outcome, and rescue/break
flags. Raw s rows can be used directly for the verification analysis figure.
Partner gold correctness is logged without adding outcome to the training reward.
Group metrics also include reference turn count and reflection failure fraction.
Each group stores stage durations and readiness/start/completion times, so overlap
can be audited. Group work durations overlap and must not be summed as batch wall
time. `batches/batch_*/timing.json` stores elapsed batch time through update/commit,
plus `projected_24_batches_seconds`. The inspection command prints that estimate.


## Runtime and credentials

Set `OPENAI_API_KEY` in the environment or `.env` selected by `SHERPA_ENV_FILE`.
Other configured API roles keep their own providers and credentials. Doctor
requests use a dedicated `doctor` API slot pool, separate from Luna judge slots.
Seeds identify cached Doctor observations; OpenAI requests do not receive a seed.

The launcher selects `runtime/python`, `.venv/bin/python`, `venv/bin/python`,
then Python on PATH. `SHERPA_PYTHON` overrides this choice. Hugging Face assets
use `HF_HOME` and `HF_HUB_CACHE`, with `runtime/huggingface` preferred when present.
Only the Qwen assistant model needs local weights. To prepare them:

```bash
runtime/hf-cli/bin/python scripts/online_rl/prepare_models.py --http
.venv/bin/python scripts/online_rl/download_progress.py --watch
```

The remaining configuration variants use the same Doctor and initial rollout
layout while retaining their respective learning settings. Their output roots
are declared in their configuration files. Protocol changes require a new run;
previous cached rollouts and checkpoints are not imported as current observations.

Tests, training and API verification have not been run for these changes.

## Plan B: mixed-case MedMCQA training data

`configs/plan_b.json` prepares an independent 2,000-case random **train-only**
pool. It excludes the current 120 training questions, calibration IDs, protected
evaluation IDs/questions, and duplicate questions. No GPU inference runs during
`prepare`. The pool is already prepared at
`outputs/plan_b_medmcqa_mixed_v1/pool.jsonl`.

After the current GPU job finishes:

```bash
bash run_plan_b.sh --phase all --launch
```

Filtering uses the original Qwen3-8B base model **without LoRA** through vLLM on
GPU 0 and the same frozen Doctor on GPU 1. It calls the shared training rollout
function with the same prompts, temperature, final outcome and stopping rules,
changing only the fanout to eight. The engine is vLLM rather than the training
Transformers engine; identical seeds are not claimed to produce identical text.
Both model queues are bounded at 128 requests. A sliding queue keeps up to 16
cases active, replacing finished cases without waiting for a whole batch.
Existing physical GPU locks are checked before either model is loaded, so an
occupied GPU aborts filtering without stopping the active training process.

The default accepts 2–6 successes out of eight and stops dispatching once the
fixed randomized pool prefix contains 120 mixed cases. In-flight cases finish,
and the first 120 eligible cases in pool order are selected, preventing a bias
for fast-finishing conversations. `--selection random120` screens the entire
pool and samples 120 eligible cases with a fixed seed instead. `--min-successes
1` accepts 1–7 successes. Set these choices before screening; changing an
already-cached screening protocol requires a fresh output directory. Pool size
and case concurrency are configurable with `--pool-size` and `--case-workers`.

Screening summaries store eight outcomes and the Doctor opening; shared call
caches preserve resumable evidence. No rubric, judge call or optimizer is used
in filtering. The script writes a new pinned `train120.jsonl`, case order,
24 randomly selected prompt optimization IDs, and independent training/prompt
configs. It then collects **32 additional trajectories only for those 24 cases**,
reusing the screening Doctor opening but using fresh continuation seeds. The
screening eight outcomes are never included in the AUC observations. The 24
cases are fixed before this collection; undefined AUC is not fabricated or
silently replaced with a selected favorable case.

Phases can also be run separately, and completed calls/cases resume:

```bash
bash run_plan_b.sh --phase prepare --launch
bash run_plan_b.sh --phase screen --launch
bash run_plan_b.sh --phase auc32 --launch
bash run_plan_b.sh --phase optimize --launch
bash run_ours.sh --config outputs/plan_b_medmcqa_mixed_v1/ours_plan_b.json --launch
bash monitor.sh --config outputs/plan_b_medmcqa_mixed_v1/ours_plan_b.json --mode training
```

The Plan B trainer reads its own optimized `final_prompt.json` and initializes
fresh LoRA weights. Its output and HF checkpoint prefix are separate from the
current experiment. Running without `--launch` only describes the planned phase.
