# SHERPA

SHERPA trains a clinical collaborator Agent with online GRPO in multi-turn Doctor–Agent discussions of
multiple-choice medical questions. For every training case the current policy plays a branching interaction tree. Each
completed trajectory gets one behavioral rubric, written after the outcome is known and verified by intervention
reruns. The Agent is rewarded for downstream success and for satisfying verified rubrics. In parallel, the
Rubric-Generator prompt evolves toward prompts whose rubrics actually change outcomes when acted on.

## Pipeline

```
case (one frozen Agent policy for the whole case)
 └─ interaction tree: Doctor opens (frozen shared D1 per case); at EVERY live Agent turn each branch samples G = 4 Agent actions; up to T_max = 5
    Agent turns (≤ 4^5 leaves); a branch also ends when the Doctor's hidden status says the discussion is settled
    (adaptive halting, after ≥ 3 Agent turns in the current v2.1 config)
     └─ completed trajectories → Doctor FINAL → outcome (success / failure)
         └─ Doctor feedback on the Agent's behavior (no gold answer)
             └─ per case and outcome stratum: a NEW rubric for a new behavioral reason, or REUSE of the existing rubric
                for the same reason (every trajectory maps to exactly one rubric) → validity gate on new rubrics
                 └─ verification on EVERY member trajectory: rewrite its matched Agent reply at Score-2 (failure) /
                    Score-0 (success), K continuations → s_j = share of reruns that flip the outcome
                     └─ KEEP iff max_j s_j ≥ 2/3; R_O = max_j s_j → kept rubrics score every Agent reply once → R_A
                         └─ reward per reply = w_outcome · mean(descendant outcomes) + w_rubric · R_A
                             └─ GRPO over same-state sibling actions → updated Agent (after the whole case)

prompt evolution (Evolving arm; every N cases, same stored trajectories, same frozen policy):
  per-case REUSE/NEW rubrics under P → K reruns per member → S_success(P), S_failure(P) → J(P) = S_success · S_failure
   → reflection / refinement → candidate prompts ranked by J → stratified paired bootstrap on ΔJ → P_t+1
```

- **Agent action.** Each Agent action is two-stage (shared infrastructure, all arms): a private assessment of the case
  formed without the Doctor's dialogue, then the visible reply conditioned on the dialogue and that assessment, which it
  may keep or revise on the evidence. It reduces physician anchoring; it does not force disagreement. The assessment never
  appears in the transcript, and GRPO credits the sibling advantage to both stages as one joint action.
- **G vs K.** G is the branching factor of the tree; K = 3 is the number of continuations per intervention target.
- **s_j / R_O** only keep or remove a rubric, weight it inside R_A and score Rubric-Generator prompts (J uses every
  trajectory's own s_j). They are never added to the Agent reward on their own.
- **R_A** of a reply is the R_O-weighted mean of the levels (0 / 0.5 / 1) of the kept rubrics that apply to it
  (two-stage Luna judge: applicability to the situation the reply must respond to, then the 0/1/2 level); None if none
  applies.

For `rubric_only`, a reply with no applicable kept rubric has reward `None` and is excluded from GRPO advantage
calculation and loss. An applicable rubric's genuine level-0 score remains reward 0 and participates. A sibling group
needs at least two scored replies with nonzero reward variation to update. New runs record `exclude_unassessed_v1` in
the manifest and refuse to resume older rubric-only runs that treated missing scores as zero. Arms with an active
outcome reward continue to use that outcome component when R_A is missing.

- **Verification protocols.** `tree_v1.json`: exhaustive member verification (every member, every Applicable reply,
  KEEP iff max_j s_j ≥ 2/3, R_O-weighted R_A). `tree_v2_witness.json` (bounded_witness_v1): each rubric is checked for a causal
  witness in up to 4 topology-diverse members at each member's feedback locus (K = 3, 2-of-3 short-circuit); KEEP at the first
  witness, else NOT_VALIDATED; equal-weight R_A over KEEP rubrics. Prompt-evolution J is the same in both.

## Reward arms (`--condition`)

| Arm | w_outcome | w_rubric | Rubric pipeline | Prompt evolution |
|---|---|---|---|---|
| `outcome_only` | 1.0 | 0.0 | off (no feedback / rubric / gate / rerun / judge calls) | off |
| `rubric_only` | 0.0 | 1.0 | on (outcome logged, not rewarded; no fallback) | off |
| `fixed` | 0.5 | 0.5 | on | off |
| `evolving` | 0.5 | 0.5 | on | on |

## Model roles

| Role | Model |
|---|---|
| Doctor (simulated physician, FINAL decision) | `google/gemma-3-12b-it`, served locally by Ollama (`gemma3:12b-it-q8_0`, Q8_0, pinned digest) |
| Agent (trained policy; also writes intervention replies) | `Qwen/Qwen3-8B` + LoRA (local, seeded initial adapter) |
| Feedback Generator (post-hoc feedback on the Agent's replies) | `anthropic/claude-4.5-haiku-20251001` |
| Rubric Generator (rubrics and the REUSE/NEW decision) | `google/gemini-3.8-flash-20260902` (reasoning effort low, strict JSON schema) |
| Rubric gate (is a newly generated rubric valid?) | `openai/gpt-5.6-luna` |
| Rubric scorer (does a rubric apply to an Agent reply, and at which level 0/1/2?) | `openai/gpt-5.6-luna` |
| Prompt reflection and refinement | `anthropic/claude-4.5-haiku-20251001` |

API models are called through OpenRouter; the Doctor runs on one or more identical local Ollama workers (one per GPU;
the endpoint is runtime provenance). Each role's model id is set in the config and recorded in the run manifest.

## Installation

```bash
pip install -e ".[train,test]"
export OPENROUTER_API_KEY=...        # or a .env file at the repository root, or SHERPA_ENV_FILE=<path to it>
```

Training cases are a JSONL file of MedMCQA-style items (`case_id`, `question`, `options`, `correct_option`) set in
`data.train`. Question text is not redistributed; evaluation-only splits are protected through
`data/splits/protected_eval_v1.json` (ids and question hashes only).

## Running

```bash
python scripts/online_rl/run_tree.py --config configs/online_rl/tree_v1.json --condition outcome_only            # prints blockers only
SHERPA_DOCTOR_ENDPOINTS=http://127.0.0.1:11435 \
  python scripts/online_rl/run_tree.py --config configs/online_rl/tree_v1.json --condition outcome_only --launch --max-cases 100
```

`configs/online_rl/tree_v1.json` is the research configuration: G = 4, T_max = 5, one pass over a 500-case MedMCQA
training pool, learning rate 1e-5, mean-centred advantages, K = 3, min_R_O = 2/3. Each launch trains one arm. The active first main runs (Outcome-only, Rubric-only) train on cases
1–100 only (checkpoints 0 / 50 / 100); there is no automatic continuation to 500. The launcher
refuses to start while any setting the arm uses is unresolved (the Evolving arm's prompt-event and acceptance values are
still open), a pinned prompt differs from its sha256, or a Doctor worker does not serve the pinned model.

Every 50 cases the checkpoint is kept and immediately evaluated for held-out task accuracy (full Doctor–Agent episodes on
VAL50 with the same two-stage Agent action, no training signal) before training continues; between milestones only the latest checkpoint
is kept. `scripts/online_rl/eval_checkpoint.py --initial` evaluates the initial adapter (checkpoint 0);
`scripts/online_rl/arm_report.py <run_dir>` reports sibling-group reward variance by depth, update counts, rubric survival
and Agent anchoring diagnostics. The training pool and the evaluation split are local files checked by sha256 and against
the protected-split manifest; they are not redistributed.

## Running SHERPA Tree v2 (frozen)

Frozen protocol: Tree v2 = bounded causal-witness verification (`verification.mode = bounded_witness_v1`) + feedback v4
(`behavior_observer_v4`: the feedback writer sees only the labelled dialogue, an exact lexical answer guard with one repair,
invalid feedback excluded) + equal-weight R_A. Prompt-evolution J is unchanged.

- Config: `configs/online_rl/tree_v2_witness.json`; rollout protocol hash `5b77544a79d30533710abaf3cc215911420654c8b22f23ca4866a0ed0efbe444`
  (printed in each run's `run_manifest.json`).
- Launch (8 × A6000; Outcome-only on GPUs 0–2, Rubric-only on GPUs 3–7; checkpoint-0 VAL50 first, then 100 cases per arm):
  `bash scripts/online_rl/launch_tree_v2_100.sh` (set `PYTHON` to the environment's interpreter; `.env` holds `OPENROUTER_API_KEY`).
- Doctor: one local Ollama `gemma3:12b-it-q8_0` worker (digest `997a7c2c0975…`) per GPU, on ports
  11450–11457 (the endpoints in the launch script).
- External data (not redistributed; checked by sha256 at launch):
  - `data/interim/medmcqa/train_random500_seed20261006.jsonl` (sha256 `5307963d7d05fc1a…`) and its `.manifest.json`
  - `data/interim/doctor_openings/train500_d1_frozen_shared_v1.json` (sha256 `ec955974cd73d90d…`)
  - `data/interim/doctor_openings/val50_d1_frozen_shared_v1.json` (sha256 `aa2483572e056b0e…`)
  - `data/interim/eval/val50.jsonl` (VAL50)
- Checkpoint 0 (initial adapter = base model) VAL50 accuracy under this protocol: 0.56.
- Write to a fresh `outputs/sherpa_tree_v2_witness/`; never resume or mix with v1 outputs or caches (`tree_v1.json`, protocol
  `af8ee6bb…`). Main-pool positions 401–410 (`data/splits/calibration_only_v1.json`) were used to calibrate this protocol and are
  refused by training runs.

## Tests

```bash
python -m pytest -q tests
```

The tests use mocks and a tiny CPU LoRA model; no API access or GPU is needed. Tests that need the local training split
are skipped without it.

## Repository structure

```
src/online_rl/tree/         case tree: rollout, post-hoc feedback and rubrics, verification, scoring, reward, trainer,
                            prompt search, real backends
src/online_rl/integrated/   GRPO update, prompt store, records and statistics shared with the tree (and the earlier
                            single-state pipeline, run by scripts/online_rl/run_integrated.py)
src/online_rl/              policy runtime, environment, case loading, protected-data guards
src/prompt_opt/             Agent/Doctor harness, rubric parsing and gates
src/judges/                 judge clients and two-stage rubric scoring questions
src/rq_eval/                evaluation harnesses
scripts/online_rl/          launchers
configs/online_rl/          configurations
prompts/                    Agent, Doctor, Rubric-Generator, feedback, intervention and prompt-search prompts
tests/                      unit tests and mocked end-to-end runs
```

## Reproducibility

Every external call (Doctor, Agent, judge, rubric generation, reruns) goes through an immutable call cache, so a resumed
run repeats no completed work. Each case commits its record, the adapter checkpoint and its hash atomically. Seeds
derive from `sha256(run_seed | key | role | index)`; every Doctor call is seeded too, but local Doctor output also depends
on the server's prompt-cache state, so cached results are treated as recorded samples. The Doctor's opening message (D1),
which does not depend on the Agent, is generated once per case into immutable tables (training 500, VAL50) and read by
every arm and checkpoint (`doctor_opening_mode = frozen_shared_v1`; table SHA256s are part of the protocol hash). The initial LoRA adapter is seeded
(`policy.lora.init_seed`), so every arm starts from the same adapter. Prompt files are pinned by sha256 and prompt versions are content-addressed.

## Status

The protocol is frozen for the first 100-case runs of the Outcome-only and Rubric-only arms (Doctor v3 with adaptive
halting, frozen shared Doctor opening, two-stage Agent action, same-reason rubric reuse, scorer v2). The Evolving arm's prompt-event cadence and
acceptance parameters are still to be set. Diagnostics so far: terminal-outcome reward varies in few sibling groups
(sparse), while kept rubrics can separate siblings whose outcome is identical; both are measured during the runs.

### Five-GPU shared rollout and GPU4 update

`configs/online_rl/tree_v21_five_gpu_shared5.json` runs the current v2.1
rubric-only arm. GPU0–4 each hold one frozen HF Agent and one pinned Ollama Doctor,
coordinate one case each, and help other cases through a shared ready-work pool.
The five cases trigger one GPU4 GRPO update after all rollout/rerun/scoring work
finishes. Idle evaluation is disabled. This launcher starts a new run in
the configured run output directory.

Each five-case batch uses one common generator prompt P0. Once its five P0
assessments finish, reflection/refinement proposes exactly one common P1. P1 uses
the saved P0 trees and feedback, with the same frozen Agent adapter and node-matched
rerun seeds. Prompt quality is the mean intervention effect over all valid-feedback
trajectories in each source-outcome direction, multiplied together:
`J = mean(success→failure effects) * mean(failure→success effects)`. Invalid,
gate-rejected and No-Match rubrics contribute zero; invalid feedback and unparsed
source outcomes are excluded from both prompts. Full member verification computes
these effects; all-members witness verification (all same-ID members until the first 2-of-3 witness) remains the KEEP
gate for training. No behavior-compliance score enters J.
Reflection receives full-batch and per-case aggregates plus at most 24 trajectory
examples distributed across cases and outcome directions (low/high effects).
This bounds API context; J still uses every eligible trajectory.

Only a strictly higher P1 product replaces P0 for the next batch. Ties, a missing
direction, or no valid distinct candidate preserve P0. There is no bootstrap,
minimum delta, acceptance split, or multi-candidate search in this launcher.
Regardless of the winner, each case is scored using the union of both prompts'
KEEP rubrics, with equal weight after exact-content and conservative same-reason
deduplication. The five resulting case targets produce one GPU4 update. Prompt
choices are saved in `prompt_events/batch_*.json`; worker `P0.json`, `P1.json` and
`union.json` preserve the comparisons. The winner is committed with the adapter
and recovery receipt, so restarting does not double-update or lose inheritance.

```bash
cd /home/kai/Collaborative_agent
# Validate config, prompt hashes, data hashes and train/evaluation overlap; no GPU/API calls.
python scripts/online_rl/run_five_gpu.py --config configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7.json
# Start five owned Doctor servers, then train. Re-running resumes.
bash scripts/online_rl/launch_five_gpu.sh
# Legacy four-case evaluation only (the new shared config disables evaluation).
SHERPA_FIVE_GPU_CONFIG=configs/online_rl/tree_v21_five_gpu.json \
  bash scripts/online_rl/launch_five_gpu.sh --eval-only
```

Set `SHERPA_PYTHON`, `SHERPA_ENV_FILE`, `SHERPA_OLLAMA_BINARY`,
`SHERPA_OLLAMA_MODELS` or `SHERPA_FIVE_GPU_CONFIG` to override the Python environment,
credentials file, Ollama executable/model directory or config. The launcher uses `python`
from the activated environment by default. The Doctor tag, digest and Ollama version remain
pinned; missing or different models fail validation. Existing servers on these ports are
accepted only when their owned PID and GPU assignment match.

## Paper artifact

The frozen reported checkpoint LoRA, exact evaluation set, per-case predictions, pinned model identities, a read-only evaluator and the training reproduction configuration are available on the `collaborative-agent-20261013` branch. See [`docs/reproducibility.md`](docs/reproducibility.md).
CUDA must expose physical GPUs 0–4 in that order. The shared v7 pipeline has been
launched on the five-A100 host; another server must pass the same data/model and
Doctor-identity preflight before it can run.

The launcher defaults to `configs/online_rl/tree_v21_five_gpu_shared5.json`.
GPU0–4 coordinate five cases per batch and serve ready model calls from a shared
local queue: four concurrent Agent rows and one Doctor request per GPU, using the
measured settings. FINAL requests emit only `Final answer: option <label>` with a
32-token cap; discussion/STATUS budgets stay unchanged. Completed GPUs continue serving other cases' ready Agent,
Doctor, FINAL, intervention/rerun and teacher-logprob calls. Case caches, branch
IDs, per-request seeds and sibling GRPO groups remain attached to the original
case. All actors verify the same adapter before each phase; no case starts until
the five actor preparations finish. One P0/P1 comparison covers the five cases.
Only after the complete batch and all pool leases finish does GPU4 unload its
Doctor and update the same Agent model. The other GPUs wait for the next adapter.

**Idle BASE and checkpoint evaluation are disabled in this configuration.**
No evaluation jobs are registered or drained. The old four-case configurations
still support their separate resumable evaluation queue when explicitly selected.

Permanent snapshots are saved at the configured checkpoint interval and at the final completed problem,
plus the initial adapter. Five-case batches give 100 update opportunities; six
batches reach each regular snapshot boundary. A recovery checkpoint with optimizer
and CPU/CUDA RNG states is also committed after each batch. Receipts prevent a
replayed batch from updating twice after a crash. Previous recovery checkpoints
are deleted after the next state commit; permanent checkpoints are kept.

Saved artifacts include full cached calls in each case's `calls.jsonl`, complete
trajectories/trees, generated/reused rubrics, applicability/KEEP evidence and
intervention effectiveness in `P0.json` and `P1.json`, both-prompt deduplicated
rubrics and response scores/rewards in `union.json`, and final training targets
and batch-update diagnostics under `cases/`. `prompt_events/` records directional
effect means, their product, the P0/P1 decision and inherited prompt; `prompts/`
preserves prompt versions. `api_usage.jsonl` records usage/cost, `live/` records
GPU timings and `pool_gpu*.jsonl` audits the actual serving GPU, original case,
policy hash, seed, input hash, task ID and result status. RPC credentials are never
written to these logs. The full shared five-GPU training still needs a hardware
run; CPU socket/process and crash-recovery tests exercise the coordinator.

The default schedule is 500 unique training cases. It keeps the original 487
non-calibration cases in their original execution order, then appends 13 deterministic
replacements from the MedMCQA TRAIN parquet (seed 20261008). Reserved calibration
IDs are excluded explicitly; the new dataset and fixed order are SHA256-pinned.
Rebuild these local data files with the CUDA environment, which includes PyArrow:

```bash
python scripts/online_rl/prepare_five_gpu_train500.py
```

### Untrained BASE reference on a separate server

`tree_v21_base622.json` pins the same Qwen3-8B base, Gemma3-12B Doctor, discussion
budgets and stopping rules as the v7 rollout (`min_agent_turns=2`, repeated-Agent
stop after turn 4). This standalone path disables the LoRA adapter and precommit,
does no rubric generation or training, and evaluates all 622 English CasiMedicos
cases. It makes no OpenRouter calls. It writes resumable results to its own
`base622_original_*` output directory.

Prepare the three source files `en_train_casimedicos.jsonl`,
`en_dev_casimedicos.jsonl` and `en_test_casimedicos.jsonl` on the evaluation server.
Activate an environment with this repository's Python dependencies, install Ollama
0.12.5, and make the pinned Qwen3-8B weights and `gemma3:12b-it-q8_0` Doctor model
available locally. Then run on the separate five-GPU host (physical GPU4 is used for
both the BASE model and its Doctor):

```bash
cd /path/to/Collaborative_agent
python scripts/online_rl/prepare_casimedicos622.py --source /path/to/casimedicos-exp/source/data/en
ollama pull gemma3:12b-it-q8_0
export SHERPA_OLLAMA_BINARY="$(command -v ollama)"
export SHERPA_OLLAMA_MODELS="$HOME/.ollama/models"
CUDA_VISIBLE_DEVICES=4 python scripts/online_rl/start_five_gpu_doctors.py \
  --config configs/online_rl/tree_v21_base622.json --launch --gpu 4
CUDA_VISIBLE_DEVICES=4 python scripts/online_rl/run_base622.py \
  --config configs/online_rl/tree_v21_base622.json --gpu 4
```

The runner checks the prepared dataset hash and requires 622 unique cases. Results,
accuracy, mean Agent turns and per-problem time are saved in the config's independent
output directory. Re-running resumes completed cases; create a new output directory
if you change the model, dataset, prompt or stopping rules.


The opt-in config `configs/online_rl/tree_v21_five_gpu_shared5_halt4_visible_pipeline.json`
uses `observed_context_behavior_v1`: User-side visible context and the Agent's
earlier visible utterances/actions are conjunctive pre-reply conditions. Hidden
precommit assessments are excluded from rubric generation, reuse, applicability
and scoring. Levels 0/1/2 repeat the same contexts and differ only in target
behavior. A source-locus repair may change the contexts, preserving every behavior
anchor and repeating the revised contexts across levels. The existing local
precommit Agent generation remains enabled; it does not make external API calls.

With `prompt_update.candidate_seed = first_completed_P0_v1`, the first case to
finish its P0 rubric/witness/full-effect evaluation supplies feedback for one
common P1. Other cases evaluate that fixed P1 as soon as their own P0 completes.
The winning prompt still uses all five cases' valid directional trajectory
effects (product of the two directional means, with the existing tie and
missing-direction rules). Both prompts' validated rubrics enter union scoring
before the single GPU4 update. `batches/batch_*/prompt_seed.json` and
`prompt_candidate.json` fix the source evidence and P1 across crashes;
`prompt_events/` records seed provenance and the full-batch comparison. This
explicitly permits first-finisher selection and its potential easy-case bias.

The same opt-in config adds the stable-choice/no-objection exit from turn 4,
retaining existing exits from turn 2. A running original-config experiment can
transition only after a committed five-case batch using the owned-process
controller `scripts/diagnostics/after_batch_halt4.py`; `--dry-run` is read-only.
Migration preserves optimizer/RNG, policy, winning prompt and prior evidence in
a fresh run directory with `protocol_transition.json`; incomplete old-protocol
batches are retained only in the source. The schedule stays 500 cases, five per
update, with permanent snapshots at the configured interval and recovery every batch.

Ready API work overlaps GPU work in this shared pipeline. Each completed FINAL
queues its leaf's feedback immediately, before the rest of that case's tree
finishes; the per-case API worker limit still applies and results retain leaf
order. Cached feedback is reused on recovery. A case's P1 completion immediately
queues its deduplicated P0/P1 union response scoring while other cases continue
P0 or P1. Union scoring does not depend on the eventual prompt winner. The final
full-batch prompt decision and Agent update wait for all required results;
canonical-bank construction retains its fixed membership/order dependencies.

The transition config also stops a branch immediately at or after Agent turn 4
when its new visible reply exactly repeats an earlier Agent reply after Unicode
normalization, case folding and punctuation removal. This local comparison adds
no model/API call and skips the next Doctor discussion turn; the required FINAL
call still runs. It is recorded as `agent_repeated_reply`.
