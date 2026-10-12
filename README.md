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

Evaluation uses all 622 English CasiMedicos cases for the original base model
and each iteration-end checkpoint (steps 30, 60 and 90). There is no 200-case
subset evaluation. Step 90 remains the main result. Run the evaluation phase
with GPUs free; it evaluates available completed iteration checkpoints and
resumes saved evaluation calls on subsequent launches:

```bash
.venv/bin/python scripts/online_rl/run_sherpa.py --config configs/ours.json --phase evaluate --launch
```

Evaluation runs separately from training. Changing only the `evaluation`
configuration preserves the existing training recovery identity and cached steps.

The first-iteration rubric audit has an isolated follow-up configuration,
`configs/sherpa_qwen_doctor_v5.json`. It separates private reflection from the
Doctor discussion instructions, tightens whole-conversation judging of repeated
agreement and unrepaired assumptions, and compares a stronger seed generator
prompt with the current optimized prompt. Selection uses three generation passes
and requires a mean-AUC gain greater than 0.02 before replacing the parent.
This threshold is an operational guard, not a statistical significance test.
Old training continues to use its existing frozen artifact.

Prepare the v5 input after step 30; this only copies the 120 immutable original-
base rollout files. It does not copy old reflections, rubrics, or scores. With
the Doctor GPU free, run the new optimization separately:

```bash
.venv/bin/python scripts/online_rl/prepare_prompt_revision.py
bash run_prompt_optimization.sh --config configs/sherpa_qwen_doctor_v5.json --phase optimize-prompt --launch
```

The new judge/reflection prompts need real calibration and optimization before
claiming an improvement. They retain the same models and rollout/Doctor rules.
No new model or API inference occurs during preparation.

The preliminary behavior audit annotates all 3,840 saved original-base dialogues
with Luna, eight dialogues per request. It hides outcome labels and the recorded
FINAL messages. Every positive tag selects an assistant passage ID; code extracts
the exact source quote. The analysis saves multi-label prevalence, tagged/untagged
success rates, within-case success-rate differences, and case-bootstrap intervals.
The main within-case comparison uses mixed cases with both tagged and untagged
dialogues; each behavior has its own eligible-case denominator. These are
observational associations, not causal effects or a trained-model evaluation.

```bash
.venv/bin/python scripts/online_rl/analyze_base_behaviors.py --launch
.venv/bin/python scripts/online_rl/verify_behavior_tags.py --launch
```

Requests default to 64 concurrent calls during the training `agent_update` stage
or after training exits. They wait during training rollout/API stages, use no
GPU, and resume validated batches. The second command reviews only positive
contradiction and unverified-case-fact candidates, saves corrected results under
`outputs/base_behavior_luna_v2/verified/`, and preserves the original annotations.
That review targets false positives; it does not establish recall or human-level
annotation accuracy. The initial request uses seven descriptive tags, including
an explicit agreement-only category. General medical knowledge assertions are
excluded from the unverified-case-fact tag even when their correctness is dubious.

For a small corrected Doctor-reflection pilot on the same saved dialogues:

```bash
.venv/bin/python scripts/online_rl/pilot_doctor_reflections.py --grounded --full-turn-evidence --output-dir outputs/doctor_reflection_pilot_v4 --launch
```

The default sample is four mixed base cases, one successful and one failed
dialogue per case. It reuses the existing frozen local Doctor server and the
original reflection seeds, uses a standalone private-reflection system prompt,
and collects helpful/hindering observations plus a wished-for assistant action.
Assistant evidence is selected by whole-turn ID, and the exact source text is
extracted by code. The dialogue text is never fragmented in this mode. Completed
outputs can be reopened without a running server.
New inference uses only the first 40 seconds of an active training update stage;
no new rollout, server launch, training-cache change, or remote API call occurs.
Grounded quotations establish source attribution, not the truth of the physician
model's retrospective assessment or its hypothetical wishes.

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

## Mean-only case-level trajectory advantages

`configs/ours_group_mean.json` inherits the unchanged `ours.json` experiment,
selecting advantage mode `group_mean`, the explicit user-revised generator
`trajectory_checklist_generator_v4.txt` and judge `trajectory_checklist_judge_v3.txt`,
and assigning independent output and HF checkpoint paths. The existing running z-score job is not restarted or
modified. The new run starts from the same fresh base/LoRA initialization; a
mode change cannot resume an old run's optimizer/cached step records.

For each case's 32 trajectories, compute `A_i = R_i - mean(R)` over the **full**
group. There is no standard-deviation divisor, epsilon normalization, extra
scale or global/batch centering. Every assistant token across every turn of a
trajectory receives that same advantage. After centering, exact-zero advantage
trajectories are excluded from both policy and KL loss. A flat-reward group is
excluded entirely. Legacy `group_zscore` and `mean_center` behavior is retained,
including their previous KL inclusion. Plan C also selects `group_mean`, with
shared-history turn groups rather than case-level trajectory groups.

The binary reward calculation, rubric schema and input visibility,
rollout/Doctor/termination rules, optimizer, KL coefficient, clipping, LoRA,
case ordering, seed and 4 x 32 / 90-step schedule are inherited unchanged.
The generator and judge text are exactly the new user-provided revisions.
`rubric_pipeline.generator_prompt_file` explicitly selects the revised generator.
The immutable optimized source artifact remains intact; the run saves a
`sherpa_user_prompt_revision_v1` prompt receipt with both source and revised hashes,
`prompt_source=explicit_user_revision` and `auc_validated=false`. It does not
reuse the old prompt's AUC. Judge/reflection backend hashes participate in
restart authentication for revised-prompt runs. Old rubric/score caches are
not imported into the new output directory. The resolved configuration is saved in `config.json`
and `run_manifest.json`; startup logs, step metrics and updater telemetry record
the actual mode, exact-zero exclusion and fixed scale 1.0.

Only inspect readiness now (no GPU/API training work):

```bash
bash run_ours.sh --config configs/ours_group_mean.json
```

When explicitly ready to launch a separate experiment with free GPUs, use
`bash run_ours.sh --config configs/ours_group_mean.json --launch`. Add
`--max-steps 2` to run only the first two new steps. Repeating the same command
resumes committed steps; `--max-steps` counts additional steps in that invocation.

Each step saves `learning_signal.json` and `signal_by_group_type.csv` alongside
`metrics.json`. They record overall/used advantage distributions, exact-zero
trajectory and group exclusions, mixed/all-success/all-failure counts, each
type's sum/share of absolute advantages, used assistant tokens, pre-clip gradient
norm and KL. The mean-only mode also records a **diagnostic** z-score distribution
computed from the same sampled rewards; it never affects the loss. Distribution
`all` includes zero advantages for a comparable denominator; `used` describes
only loss trajectories. Absolute-advantage mass is not measured gradient attribution.

`learning_signal_monitor` reads same-step metrics from the old run without writing
there. It checks matching model/data/rollout/optimizer settings. The revised configuration
explicitly permits a reference with different generator/judge prompts and marks
that comparison as descriptive, not a controlled normalization-only comparison.
A gradient
norm below 0.1 of the reference produces a `learning_signal_warning`. KL within
1e-8 for three consecutive optimizer steps also produces a warning. The first
step's zero KL is expected because the fresh adapter matches the frozen base;
one/two steps alone cannot trigger the three-step warning. Alerts are stored in
step metrics and logs. They never increase learning rate or introduce a multiplier;
any future fixed advantage scale must be applied consistently across compared runs.

Rebuild the first 30 steps' tables, or compare the first two steps once available:

```bash
.venv/bin/python scripts/online_rl/analyze_advantage_signal.py \
  --run-dir outputs/ours_qwen3_doctor_256_compact_luna_rubric_group_mean_prompt_v2 \
  --reference-dir outputs/ours_qwen3_doctor_256_compact_luna_rubric_v4 \
  --max-step 30 \
  --output-dir outputs/advantage_signal_comparison_v1
```

The report includes per-group/per-step/type CSVs, overall type summaries with AUC
and reward variance, gradient/KL curves and JSON. Only committed steps are read.
For historical logs without outcome types it reads the original saved case
records and checks exact reward/advantage alignment. It performs no new rollout,
judge/API call or optimization and cannot write into either source run directory.

## Plan C: counterfactual replies at each main-dialogue turn

Plan B is retired (`configs/plan_b.json` has `retired: true`; its default runner
rejects execution). Historical code and generated datasets remain for provenance.
The active `ours` run and its configuration/checkpoints are unchanged.

`configs/plan_c.json` is an independent fresh-LoRA experiment. It retains the
120 training cases, four cases per batch, 30 steps per iteration, three iterations,
assistant temperature 1, frozen Qwen3-8B Doctor and existing termination rules.
GPU 0 owns the Transformers trainable assistant; GPU 1 owns the frozen vLLM
Doctor. Both request limits remain 128; Luna generates and judges rubrics.

For each case, first generate **one main conversation** to Doctor FINAL. For
every assistant turn actually reached by this conversation:

1. Freeze the complete visible history immediately before that response and
   preserve the Doctor status/private-state snapshot for identity and halting.
2. Reuse the main response and its original continuation. Sample 31 alternative
   responses from the **same policy prompt/history**, and continue each naturally
   to FINAL under the same absolute turn limit and stopping rules. Branches do
   not become the history of later main turns. This is a spine with siblings,
   not recursive exponential branching.
3. Select up to four successes and four failures (four examples total for a
   uniform-outcome group), and ask the Doctor for standalone, private reflection
   about the **marked response only**. Ground citations in the full target turn;
   unsupported observations must explicitly use `none`.
4. Generate five or six **turn-specific** binary rubrics from these observations.
   The user-revised `trajectory_checklist_generator_v4.txt` supplies the base
   instructions, followed by `prompts/plan_c/turn_generator_v2.txt` enforcing
   target-turn scope. The judge uses `prompts/plan_c/turn_judge_v2.txt`, adapted
   from the revised sentence-evidence/anti-repetition judge: it evaluates only the
   marked response, including avoidance criteria. Source, contract and effective
   prompt hashes are stored alongside the original optimized artifact provenance.
   This revised prompt has **not** been optimized or AUC-validated for Plan C.
5. The blind judge sees only the clinical case, shared pre-response history,
   marked response and that turn's rubrics. FINAL, outcomes, later continuation,
   reflection, private state and correct-answer labels are excluded. Earlier
   actions provide context but cannot earn this response's credit.
6. Average the binary scores for each target response. With `group_mean`, compute
   `A_i = R_i - mean(R)` over all 32 siblings at this exact history, without
   dividing by standard deviation. Exclude exact-zero advantage responses from
   both policy and KL loss, and exclude an entirely flat-reward group. Pass **one target trace per
   candidate** to GRPO, excluding prefix and continuation action traces.

All turn groups from the four cases form **one optimizer update** after collection
and scoring finish with frozen policy weights. A zero-variance group contributes
no advantage. The selected mode, fixed scale 1.0 and target-only credit scope
are saved in the resolved config, startup log and step metrics. Rubric failures stop the batch without inventing rewards. AUC uses
half-credit ties and remains undefined for uniform outcomes. It measures an
observed association with the eventual outcome, not a causal effect of the
single response: downstream policy/Doctor sampling also varies.

A case reaching four turns produces four groups of 32 candidates (128 target
responses), so a four-case step has up to 16 groups / 512 target responses.
This increases work relative to the old 32 full dialogues per case. With four
turns and no early stopping, the assistant samples 4 main actions plus
31 x (4 + 3 + 2 + 1) branch actions = 314 actions per case, versus 128 before.
The concurrency limit 128 bounds requests in flight, not the total sample count.
When a turn's branches complete, its reward pipeline starts immediately and
can overlap the next turn's branch collection. There is no cross-step prefetch
with stale adapters.

Inspect the configuration without model loading, API calls, or training:

```bash
bash run_plan_c.sh
bash run_plan_c.sh --responses-per-turn 8
```

Implementation is prepared only; no real Plan C experiment has been launched.
When explicitly ready to run it after the current job, the future entrypoint is
`bash run_plan_c.sh --launch` (optional `--max-steps 1` for a single batch).
Acquire both physical GPU locks before model loading; an occupied GPU causes
an error without stopping the existing training process.

Output is isolated at `outputs/plan_c_turn_counterfactual_group_mean_prompt_v2`. Restart with the
same command/config to reuse completed calls, turn observations and committed
steps. Every step publishes a durable recovery checkpoint; every ten steps also
publishes a checkpoint and queues HF upload under `plan_c_turn_counterfactual_group_mean_prompt_v2`
in `zzu819/SHERPA`. Changed config, prompt/backend identity or policy hashes
cannot silently import stale cached rewards/checkpoints.

Each step stores `metrics.json`, `learning_signal.json` and
`signal_by_group_type.csv` with per-turn AUC, reward variance, overall/used
advantage distributions, mixed/all-success/all-failure absolute-advantage sums
and shares, gradient norm, KL, update diagnostics and timing. Groups are typed
from all sibling continuations, not from the main conversation alone. A KL
near-zero alert is emitted after three consecutive optimizer steps. No real
old Plan C run exists for a same-step gradient comparison; the diagnostic
z-score distribution from the same rewards remains available.
`analyze_advantage_signal.py` supports these dynamic turn groups as well as
case-level runs, and records the group scope explicitly. Each case stores `spine.json` and
`turn_XX/{rollouts,reflections,rubric_scores}.json`. The progress event stream
includes `turn_rollouts_completed`, `turn_reward_completed`, `agent_update` and
`batch_committed`; existing initial-fanout dashboard assumptions should not be
used to interpret Plan C's dynamic sample count.
