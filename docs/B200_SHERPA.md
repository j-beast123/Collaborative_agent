# SHERPA: terminal reflection, fixed accuracy, linear verification

This version is implemented in branch `sherpa-jh`. Implementation and CPU checks
do not start training, inference, API requests or Ollama services. Run the commands
below yourself on the B200 server after pulling this branch.

## Scientific configuration

Use `configs/experiments/train120_max4_rubric_only_b200.json` with
`scripts/online_rl/launch_b200.sh`. The old five-GPU launcher is not used.
The standalone configuration records its source config and hash under `notes`.

| Item | New run |
|---|---|
| Dataset/order | Same pinned train120 and separate order file; no resampling |
| Seed | Run 0; LoRA initialization 20261006 |
| Agent | Local Qwen3-8B, BF16; rsLoRA r16/alpha32/dropout0, q/k/v/o projections |
| Agent protocol | Existing `precommit_v1`, private120 + visible400; both remain in the GRPO action |
| Agent decoding | Existing branch values: temperature1, top-p1, top-k0; per-row RNG |
| Doctor | Fresh gemma3:12b-it-q8_0 download, Ollama0.12.5; actual digest pinned at preparation |
| Dialogue | Maximum4 Agent turns, branching4, minimum2; existing halt/stop rules, repeated-reply check from turn4 |
| Batch/update | Five cases on the same frozen policy; all original sibling groups then at most one optimizer step |
| GRPO | lr1e-5, beta.04, clip.2, gradient clip1, mean centering; existing pi_old/pi_ref/group filters |
| Reward | Outcome0, rubric1; equal-weight normalized rubric scores |
| Generator | Fixed posthoc-only P0; no prompt optimization or P1 |
| Checkpoints | Initial, every30 cases, final120; recovery checkpoint every5-case batch |
| Evaluation | No base or intermediate test; explicit final120 checkpoint on test622 once |

Generator, consolidate, gate and scorer use **direct OpenAI `gpt-6-luna`**.
This changes the gate model from gpt-5.6-luna and moves all four API roles off
OpenRouter. Only `OPENAI_API_KEY` is read; no `.env` key fallback or provider
substitution is used. Doctor reflection remains local Gemma.

Requests use `https://api.openai.com/v1/chat/completions`,
`max_completion_tokens`, and (generator/consolidate) `reasoning_effort: "none"`.
Generator/consolidate omit temperature; gate/scorer retain provider-default
reasoning and their existing prompts, structured schemas and token budgets.
Provider/model/settings enter manifests and cache identities. Old provider caches
are not reusable. Existing committed runs cannot resume with a changed config.

See the [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [Chat Completions request schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
The returned model must be the requested alias or a date-suffixed snapshot of
that same model. The actual response model and request ID are audited. Unsupported
settings/models fail explicitly; real account access has **not been live tested**.
All retries share the run's 50-slot limiter; retry/backoff waits release slots.
Generator/consolidate reasoning tokens despite explicit `none` are rejected.
Existing bounded JSON repair rules are retained; errors never become reward zero.

The initial adapter parameter hash must equal
`9018ce3bbacd319049576e8ace88afb4777621a8769919bf46676fca8a85b17c`.
Initialization seed and LoRA settings are fixed; startup checks the actual hash.
The local base snapshot's JSON/tokenizer/weight file hashes are also recorded and
checked on resume. Matching adapter initialization alone does not prove that a
different server's base weights match; reuse the original Qwen snapshot there.

## Execution order and prompts

1. Build the ordinary tree with `prompts/user_simulator_v4_final_label.txt`.
   Doctor replies are plain public text plus the original hidden STATUS. No
   prospective private need is requested, parsed, cached or shown to any role.
2. Four Agent siblings share the same public Doctor prefix. Agent responses keep
   `prompts/agent.txt` and the existing private-assessment/visible-reply protocol.
   This Agent private assessment is unchanged and is not a scorer input.
3. After each eligible terminal FINAL, the same local Doctor reflects using
   `prompts/sherpa/doctor_post_final_reflection_v1.txt`. The two-field JSON has
   `helpful` and `unresolved` arrays, each with zero or one reply-linked experience.
   The original512-token budget and one bounded format repair apply. No
   `need_assessment` or prior private need is supplied or required. Reflection
   sees the visible case/discussion/recorded FINAL, never correctness or gold.
4. Independently generate candidates with the fixed
   `prompts/sherpa/trajectory_rubric_generator_v4_posthoc.txt`. Audit every
   retrospective claim against public dialogue. Missing prospective records are
   not a rejection condition. Keep the public clinician-side/assistant-side
   applicability, response-only anchors and fixed-accuracy duplicate exclusion.
   Candidate scheduling remains chunks of8 across the same five frozen cases.
5. Sort candidates by stable ID and consolidate once per case using
   `prompts/sherpa/consolidate_partner_bank_posthoc_v1.txt`. No private need IDs
   are required. Keep original provenance/anchors, success/failure strata,
   verbatim canonical representatives and deterministic source selection.
   Gate each canonical rubric with `prompts/sherpa/luna_gate_posthoc_v1.txt`.
   Public evidence must establish applicability; hidden preferences cannot.
6. Verify each valid canonical partner rubric once on its **creator trajectory**.
   This is a deterministic source-selection rule, not a random search: source seed
   is explicitly null, rule and run seed are recorded. Rubric reuse cannot search
   for another source after seeing verification outcomes. All three trial seeds,
   source, anchor, guidance and policy identity are cached before starting trials.
7. Generate a single linear continuation per trial, beginning at the anchored
   Agent response. Use normal precommit; add the contrasting behavior guidance
   from `prompts/sherpa/verification_linear_v1/score0.txt` or `score2.txt` only to
   the anchored visible response. Private assessment and later responses use their
   normal prompts. Failure sources get level2, success sources level0. Existing
   prefix plus generated turns totals at most4 Agent turns; normal early stops apply.
   Follow-up Doctors use the same ordinary public/STATUS protocol. Verification dialogues
   are audit records only: no reflection/generator/scorer is run on this dialogue,
   and no verification action is added to GRPO.
8. Run planned trials0 and1 concurrently, collect them in index order. Two flips
   KEEP; two valid nonflips DROP; otherwise run trial2. At least two flips KEEP,
   at least two valid nonflips DROP. Incomplete evidence is separately INCOMPLETE.
   This agrees with fixed-three majority for three valid binary outcomes. A
   two-trial decision stores exactly two results and a planned-but-unexecuted third
   seed, never an estimated third outcome.
9. Add `prompts/sherpa/fixed_medical_accuracy_v1.json` exactly once to the case
   scoring book. It is outside reflection generation, gate and verification.
   Score with the same response scorer and
   `prompts/sherpa/scorer_need_accuracy_v1.txt`.
10. When all five cases are complete and all generation/scoring futures have
    returned, use their ordinary tree responses for one existing GRPO update.
    Publish checkpoint, diagnostics and progress, then start the next batch.

## Parallel bank and global API capacity

Set `rubric.candidate_batch_size` (default8) and
`execution.global_api_max_in_flight` (default50) in
`configs/experiments/train120_max4_rubric_only_b200.json` **before starting**.
The existing `five_gpu.api_workers` controls ready scoring/feedback tasks; it
is a worker count, not another API allowance. Five case coordinators can generate
up to40 candidate tasks together at the default chunk size. Consolidation is
sequential within each case and can overlap across cases. No dummy requests are
created to reach50. Generation batch size means scheduling chunk size; it does
not change the five-case optimizer batch or combine terminals into one prompt.

All roles and retries share ONE process-wide semaphore at the physical JSON HTTP
transport. The B200 launcher performs these calls in one process; case/role
threads cannot create independent limits. The limiter conservatively covers
local Doctor HTTP attempts too, whose existing concurrency remains1. It does
not impose a machine-wide/account-wide cap on separately launched experiments.
Physical requests hold a slot until response reading completes or raises. Retry
waiting occurs after release. Existing bounded budgets remain:6 transient and11
rate-limit retries, Retry-After where available (capped300s), exponential jittered
backoff and rate-limit wait budget1200s. No retry changes model or settings.

Per-case `calls.jsonl` stores independent `rubric_generation` results,
`consolidate_input`, `case_consolidation`, `canonical_bank_plan` (with original
candidates, mapping, exclusions, chosen source/anchor), and `case_bank_complete`.
A single-flight owner finalizes each case bank. The launcher process/output/GPU
locks prohibit competing runs. Partial resume reuses successful calls and the
saved consolidation plan; completed verification plans/trials retain the
existing durable cache. Failed attempts are audit-only and may recur on resume.
A request interrupted before its response is saved can be repeated; this is not
an exactly-once guarantee at the remote provider. Completed cases and committed
updates use the existing five-case checkpoint/recovery mechanism.

Fixed medical accuracy is never passed to candidate generation/consolidation,
gate or verification. It remains a single scoring-book entry. Reward0/1,
max4, early majority verification, Doctor protocol and the frozen-policy barrier
are unchanged. Old output directories/caches are not imported.

Independent generation can cost more than assign-or-create because equivalent
trajectories generate candidates before deduplication. One consolidation request
per case adds cost; it receives all eligible candidate records, so worst-case
context/output size can be large. Provider context-limit errors stop with saved
candidates; there is no silent truncation or pruning. Changing scheduling chunk
size does not reduce that final consolidation input. No measured speedup or cost
reduction is claimed without a real run.

## Applicability and reward

Partner rubrics use `applicability.user_side_context` (clinician need, uncertainty,
judgment state), `applicability.agent_side_state` (what was already provided and
what remains unresolved), and situation-specific 0/1/2 behavior anchors.
Applicability is shared across siblings and depends only on the public prefix.
Different private assessments cannot split that shared applicability judgment.
Scorers receive case evidence, public prefix, visible rubric and response; they
receive no private needs, reflections, private assessment, future or final outcome.

* Different situation: N/A; no numeric score.
* Matching situation but missing behavior: 0, not N/A.
* Partial/sufficient behavior: 1/2, normalized to .5/1.
* Fixed medical accuracy: always applicable, same 0/1/2 scoring and normalization.

The fixed instruction is: **Provide medically accurate information and
recommendations consistent with the case evidence.** Its fixed anchors are in
the JSON file above. It does not require a particular answer letter, agreement or
forced certainty. An empty/unassessable answer receives the defined behavioral0;
an API/parsing failure does not become0.

For response n, `R(n) = mean(valid normalized scores of applicable KEEP partner
criteria + the one fixed accuracy criterion)`. The original sibling mean-centred
GRPO advantage follows. There is no outcome mixing, correctness bonus or extra
accuracy gate. If every partner criterion is N/A, valid accuracy alone supplies
the reward. If accuracy too has no valid score, reward=None; no outcome fallback.
Original filters still skip groups with too few valid responses or no variation.

## Errors, caches and resume

Existing bounded transport retry policies are retained (see `src/common/retry.py`
and `src/online_rl/doctor.py`). Authentication/nonretryable failures stop the batch.
The optional legacy Doctor need protocol (disabled in this run) gets one deterministic format repair, then aborts with
raw validation evidence. Reflection retains its one format repair; persistent
format errors abort. Existing semantic-invalid reflection exclusion remains.
FINAL parse failures remain None, are recorded and never count as an
incorrect answer or a successful flip; FINAL is not repeatedly sampled until valid.

Malformed/missing applicability or behavior results stop the batch after existing
backend repairs. An explicit valid `rubric_unscorable` is distinct: that rubric is
masked across the entire sibling group, preserving a common denominator, while
remaining valid criteria (including accuracy) may contribute. Errors are never
silently converted to N/A/0. If fixed accuracy is also masked and nothing remains,
the group supplies no training reward. See response scoring evidence and errors.

Successful Doctor outputs (public text and hidden STATUS), tree calls, source
plans, completed verification model calls/trials, rubric generation and scoring
are durable caches under the new output directory. Resume loads them rather than
re-generating completed calls. A process killed after a provider completed a request but
before it was saved can repeat that unsaved request; physical attempts are audited.

Checkpoint publication precedes progress publication. A crash in between restores
the saved policy/optimizer/step/RNG and does not apply the batch twice. Each resume
checks scientific configuration, source code, local model files, initial adapter,
current policy and Doctor identity. Keep the same paths/config/code for a run;
do not edit a running experiment or reuse caches from the previous protocol.
An interrupted initial checkpoint publication without run state is rejected for
inspection, rather than overwritten automatically.

## Two GPUs and expected differences

PhysicalGPU0 is dedicated to one Doctor replica on port11985. PhysicalGPU1 hosts
one Assistant replica for rollout, verification and update. No extra reference or
verification model is loaded. Five CPU case coordinators share an Agent generation
batcher of capacity4 and Doctor concurrency1. Two verification trial workflows
overlap; their Doctor generations serialize at this capacity. Results are assembled
by fixed trial and case index, independent of completion order. API state workers
remain3 per case, at most15 across five concurrent cases.

The five-case update barrier and number of training cases/updates do not change
with GPU worker count. Dynamic batching/padding and different GPU kernels may
still change floating-point outputs even with per-request seeds; bitwise dialogue
identity across five-GPU and two-GPU hardware is not promised. Need generation
also intentionally changes the Doctor output protocol. Memory and throughput on
the remote B200s have not been measured. No memory-based automatic change to
batch size, model, token budget or update semantics is made.

## Data and local resources

All three private files have been copied byte-for-byte into this worktree and
hash-checked. They are gitignored and **will not arrive through git pull**.
On B200 retain the same files or use the path overrides below.

| File | SHA256 |
|---|---|
| `data/interim/medmcqa/five_gpu_train120_seed20261008/train120_excluding_calibration_seed20261008.jsonl` | `3d34929094925c822062d03839b49de8f6fb43afba71fb573de409471bb5b8ab` |
| `data/interim/medmcqa/five_gpu_train120_seed20261008/train120_order_seed20261008.json` | `edf23262be5430d10a0d02dde3da80710bf9a91bd941d83234e9c98372752c41` |
| `data/interim/casimedicos/all_en_622.jsonl` | `e42077d006c4e8227564f99b06050b66e7df9989f9b2b17f5fc5d9dd083feb56` |

Read-only checks on the **current editing server**, not B200, found:

* Qwen: `/convei_nas2/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`.
* Ollama: `/convei_nas2/zzuhyun/ollama_baselines/opt/ollama-0.12.5/bin/ollama`.
* Baseline model store: `/convei_nas2/zzuhyun/ollama_baselines/models_gemma3_88c2e99b4a44`,
  manifest digest `88c2e99b4a44ef8a4c855315927ae99f6ea77a06f4ec24ad2d08929f818f5f00`.

The new preparation path downloads Ollama0.12.5 and the currently available
`gemma3:12b-it-q8_0` into a separate account-owned store. The original digest
`997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc` is retained
as provenance, not a blocker for this approved fresh download. No digest is
invented before downloading on B200.

The first successful pull verifies the service version, Q8_0 metadata, local
manifest digest against `/api/tags`, and every model blob's SHA256. It atomically
creates `.local/b200_doctor.json` (gitignored), recording the old/new digest,
executable hash, local paths and UTC time. The launcher automatically reads this
receipt and updates Doctor, reflection and opening-cache identities together.
Outputs/checkpoints/caches become
`outputs/sherpa_b200_max4_posthoc_only_v3/doctor_<actual_digest_first12>/`.
An old model's caches are not used. Keep this receipt for resume/evaluation;
it contains no API key. A subsequent `pull` command verifies the pin and does
**not** pull the mutable tag again or overwrite the receipt. Model/executable
changes fail checks. Preflight and resume retain identity checks.

Without a receipt the preserved source config still describes the original
Doctor. Complete the download path below before starting a fresh run. Do not
switch an already started experiment's Doctor in place.

The installer uses the official [v0.12.5 assets and published SHA256 values](https://github.com/ollama/ollama/releases/expanded_assets/v0.12.5),
with Linux x86_64/ARM64 archive selection. It performs no system-wide installation,
sudo, service registration or process termination. Pull only targets the dedicated
service started by this helper: PID, executable, model-store and port checks
prevent a pull into an unrelated existing service. Model identity uses the
[official list-models API](https://docs.ollama.com/api/tags).

Allow roughly **30 GiB free** for model files, downloaded installation archive
and extracted libraries; this is a planning allowance, not a measured size or
space for training checkpoints. The installer checks8GiB for the binary package
stage. For a larger disk set `SHERPA_OLLAMA_HOME` before all setup commands. The
default is `$HOME/.local/share/sherpa-b200`; model storage is its `models/` child.
Remote filesystem, credentials and live provider model availability remain
unverified on this editing server.

## Clone a new server and choose learning rate

See [the complete clone-to-evaluation command guide](B200_CLONE_RUN.md).
Set `SHERPA_LEARNING_RATE` or pass `--learning-rate`; CLI wins over environment,
then config. Default `1e-5` preserves the existing output path. A different value
adds `lr_<value>/` before `doctor_<digest>/`, separating output, cache and
checkpoints. The effective optimizer config enters the manifest/config hash.
Resume and evaluation require the original learning rate; this does not add a
scheduler or permit changing LR inside an existing run. Report supports the same
flag/environment. For example `SHERPA_LEARNING_RATE=5e-6` selects
`outputs/sherpa_b200_max4_posthoc_only_v3/lr_5e-06/doctor_<digest12>/`.

## Commands on B200

Use the existing CUDA/PyTorch/Transformers/PEFT environment compatible with this
repository; no dependencies, models or services were installed for the remote
server. Substitute the repository, Python and Qwen paths below. Paths and the
API key must be set in the terminal that launches training/evaluation.

```bash
cd /ABSOLUTE/PATH/TO/Collaborative_agent
export SHERPA_PYTHON=/ABSOLUTE/PATH/TO/EXISTING/ENV/bin/python
export SHERPA_BASE_MODEL=/ABSOLUTE/PATH/TO/EXISTING/Qwen3-8B/snapshot
unset SHERPA_OLLAMA_BINARY SHERPA_OLLAMA_MODELS SHERPA_DOCTOR_RECEIPT
# Optional: export SHERPA_OLLAMA_HOME=/PATH/ON/LARGE/DISK/sherpa-b200
export CUDA_VISIBLE_DEVICES=0,1
set +x
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

Do not put the key in a file or command-line argument. If the data paths on B200
already differ, set these optional overrides to those existing exact files:

```bash
export SHERPA_TRAIN_FILE=/ABSOLUTE/PATH/TO/train120_excluding_calibration_seed20261008.jsonl
export SHERPA_TRAIN_ORDER=/ABSOLUTE/PATH/TO/train120_order_seed20261008.json
export SHERPA_TEST_FILE=/ABSOLUTE/PATH/TO/all_en_622.jsonl
```

Download the pinned Ollama binary (no GPU inference or service startup):

```bash
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup install
```

In a dedicated second terminal, cd to the same repository and set the same
`SHERPA_PYTHON` and optional `SHERPA_OLLAMA_HOME`. Start the private Doctor server:

```bash
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup serve
```

This stays in the foreground on physicalGPU0 and `127.0.0.1:11985`, one replica,
one parallel request. It refuses an occupied port. Keep it running; do not run
this command a second time while that service is up. No model generation is
requested by setup.

Back in the training terminal, download Doctor and freeze its real identity:

```bash
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup pull
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup verify
bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_rubric_only_b200.json --action preflight
```

`verify` rechecks every blob without downloading. `preflight` checks config,
prompt/data/order hashes, local Qwen shards, binary/version and pinned manifest,
blob presence and key presence. It sends no paid requests, starts no service and
loads no GPU model. Startup also verifies the live Doctor before generation.
No manual binary/model-path exports are needed after the receipt is created.
All train/resume/evaluation commands below automatically load the same receipt.
If download is interrupted, rerun the same install or pull command; completed
model blobs and a partial binary download can be reused. A checksum-corrupt
archive must be removed from this private install directory before retrying.

First real batch (five cases; counted as the first batch of the full run):

```bash
bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_rubric_only_b200.json --action train --max-batches 1
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py
```

Expect completed_cases5/batches1. Inspect
`outputs/sherpa_b200_max4_posthoc_only_v3/doctor_<digest12>/diagnostics/batch_000001.json` for
need status/parse repair, fixed/partner score distributions, verification actual
trial counts, applicability and skip reasons. A batch can validly commit without
an optimizer step if no group has a usable nonzero advantage. This command runs
one **additional** batch if a run already exists.

Continue all120 cases; the first batch is reused, not restarted:

```bash
bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_rubric_only_b200.json --action train
```

After interruption, re-enter the same environment variables/key in the new
terminal and use the same command. No separate resume flag:

```bash
bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_rubric_only_b200.json --action preflight
bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_rubric_only_b200.json --action train
```

After completed_cases120/batches24, evaluate only the final checkpoint:

```bash
bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_rubric_only_b200.json --action evaluate
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py
```

The selected checkpoint is
`outputs/sherpa_b200_max4_posthoc_only_v3/doctor_<digest12>/checkpoints/checkpoint_000120`.
Test622 keeps original row order and case seed0+index, one conversation per case,
ordinary Doctor FINAL parsing, no final retry and unparsed cases in the accuracy
denominator. Discussion uses this run's joint Doctor/need protocol, but no rubric
or reflection is generated during test. The evaluation action refuses an unfinished
train120 or another checkpoint. Re-running it resumes missing evaluation cases;
completed cases are not evaluated again. `final_evaluation_selection.json` fixes
identity; `final_accuracy.json` records the final metric. No checkpoint search or
test-based tuning is performed.

Optional CPU regression command (fake model/API calls, tiny CPU model only):

```bash
bash scripts/online_rl/test_b200.sh
```

## Diagnostics and additional costs

Case and batch JSONs record counts/ratios with explicit denominators: created and
reused partner rubrics, gate outcomes, unsupported-need exclusions, KEEP/DROP/
INCOMPLETE, actual trials and early decisions, applicability, sibling score
variation, equal-outcome/different-rubric groups, N/A/unscorable/excluded groups,
accuracy-only eligible/trained groups, accuracy/partner score histograms and
optimizer steps/skip reasons. Full verification records include frozen source,
anchor, planned seeds, actual continuation lengths and outcomes. Saved needs
include generation attempts, repairs and timing. Missing/error results are not
made into synthetic scores.

`api_usage.jsonl` records logical role calls and provider usage;
`http_attempts.jsonl` records physical HTTP attempts, failures, reported usage and
latency without request headers, keys or prompt/error bodies. `retry_audit.jsonl`
records retry counts/waits. Case cache error logs and `verification_attempts.jsonl`
retain failed operation/trial evidence. Batch diagnostics distinguish these
counts; nested cache failures are not counted as extra physical API requests.
Reported API cost and unknown/unpriced usage are separate; missing price is not
claimed as zero. Assistant private/visible tokens are recorded separately.

Joint Doctor calls expose exact **combined public+need** tokens and elapsed time.
Need-only incremental generation tokens/time are null because Ollama does not
provide a per-field breakdown and no public-only counterfactual request is made.
Need serialized character count is also retained. The joint cap is640 tokens
(public reply instructed to stay within256); this is an intentional change from
the old256-token response, and the public portion is prompt-limited rather than
independently hard-truncated. Reflection cap grows512→1024 for need links. These
changes can affect generation and costs and are recorded in the configuration.
Parallel case phase times are sums, not additive wall-clock GPU time.

CPU tests cover pre-generation/sharing/private isolation, exact public evidence,
immutable need/reflection links, shared applicability, fixed criterion uniqueness,
verification exclusion and actual reward connection, guided-only anchor behavior,
full-turn/early-stop continuations, all binary majority outcomes, partial trial
resume, no outcome fallback, five-case barriers and optimizer/policy/RNG recovery.
Real B200 memory, remote paths and live model availability remain execution-time
checks, not results of these CPU tests.

Validation on the editing server: **719 passed, 12 skipped, 9 deselected** in the
active regression suite, including **51 new B200 tests**. The existing active
test script lists the nine excluded historical contracts; data-dependent tests
for the undistributed older training pool remain skipped. CPU-only preflight has
no blockers with the three verified local data files. Shell syntax, Python AST
parsing and `git diff --check` also pass. Local-socket tests required running the
CPU suite outside the filesystem/network sandbox; their calls still used fakes.
No live training, model inference, paid API call or service management was run.

Doctor download addition: **59 targeted CPU tests passed**, including8 new setup tests. Downloads and HTTP responses are mocked; no real installation, model pull or service launch was performed during implementation. The full-suite719 result above predates this setup-only addition.

## Parallel-bank diagnostics

`rubric_bank.candidate_bank` and case diagnostics include scheduling chunk size,
candidate tasks/logical generation requests, generated candidates, canonical
rubrics, duplicates removed and exclusions with stage/reason. The canonical plan
retains every original candidate and source/anchor mapping. Batch diagnostics
aggregate candidate counts. Physical generation attempts, including correction
and retry requests, are separately counted under `rubric_generator`.

`http_attempts.jsonl` records actual in-flight at entry, the process peak and cap,
limiter wait, role, usage, reported cost, status/error and elapsed time. Role
summaries include HTTP429 counts, retry events, summed time and unpriced attempts.
The maximum is the maximum across roles, never the sum of per-role maxima.
`retry_audit.jsonl` records both body-signaled rate limits and transport retries.
Total role request seconds are not wall-clock speed; parallel work overlaps.
Unknown cost remains explicitly unpriced. Compare both wall time and actual
attempt/token/cost totals when assessing this change.

## Previous CPU validation: parallel bank

**743 passed, 12 skipped, 9 deselected** in56.53s with:

```bash
SHERPA_PYTHON=.venv/bin/python PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 bash scripts/online_rl/test_b200.sh
```

This includes16 new fake-API/bank tests: actual OpenRouter payload serialization,
unsupported-option rejection, model mismatch, model/reasoning cache separation,
sorted input and source under reversed completion order, single bank owner,
original member mapping, canonical-only gate/verification, partial candidate and
post-consolidation resume, global transport limiting and retry-slot release, and
real TreeTrainer wiring through the fixed-accuracy rubric-only reward path.
Existing tests cover max4 normal continuation, anchored-only guidance, private
need sharing/isolation, all early-majority cases, verification resume, five-case
barriers and policy/optimizer/checkpoint restoration without double updates.
The12 skips and9 exclusions belong to the existing inactive/legacy fixture set;
no new exclusion was introduced. Local Unix-socket fake-worker tests required
running outside the sandbox. No actual GPU inference, API request, installation,
download, service management, training or test evaluation was performed.
CPU preflight returned an empty blocker list. Remote resources/model support
must still pass the user-run checks described above.

Learning-rate override addition: **74 targeted CPU tests passed**, including13 new LR tests (CLI/environment/default precedence, invalid values, separate output paths, resume identity, real CPU optimizer and progress reporting). CPU preflight with `SHERPA_LEARNING_RATE=5e-6` returned no blockers. The743 full-suite result above precedes this addition. No installation or live execution was performed.

## Direct OpenAI / 50 requests update CPU validation

802 distinct active tests passed:794 in the sandbox plus8 local-socket tests
confirmed outside it (the socket rerun passed10 tests, including2 already passed).
12 existing skips and9 deselections remain. Six new fake-HTTP tests cover direct
OpenAI credentials, all four role URLs/payloads, provider-specific cache identity,
model substitution refusal and bounded global concurrency with retries. After
adding API provider labels to the manifest,22 API/bank tests passed again.
CPU-only preflight reported no blockers; actual B200 model paths/account access
must still pass normal preflight in the user terminal. No live LLM API call,
GPU inference, training or held-out evaluation was executed.

## Posthoc-only change record

The user removed prospective Doctor needs after real Gemma output omitted a
required quotation and copied the schema's pipe-separated status placeholder.
Validation correctly rejected those outputs; this update does not relax it.
The legacy need implementation/tests remain available, but this run disables it
in both normal rollout and continuation verification. Reflection returns to its
original two fields and512-token budget. The fixed accuracy criterion, GRPO,
public-only scoring, source/seed plan and early-majority verification remain.
Generator/gate/consolidate prompts and opening-cache identities change explicitly.
A fresh output/run name is required; previous files remain untouched. No automatic
fallback from enabled needs to disabled needs occurs on an error. Diagnostics
mark needs disabled and do not count intentional absence as missing records.

Posthoc-only CPU validation:806 distinct active tests passed (798 in sandbox,
plus8 local-socket tests confirmed outside it; the socket rerun passed10 including
2 already passed).12 existing skips and9 deselections remain.74 targeted tests
passed, including ordinary HF Doctor public/STATUS handling without need JSON,
terminal-only reflection and no gold/need leakage, disabled-need diagnostics,
unchanged training contracts and refusal to resume an old need-enabled run.
CPU-only preflight reported no blockers. No real API/GPU/training/evaluation ran.
