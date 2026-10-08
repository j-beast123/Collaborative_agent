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
    (adaptive halting, after ≥ 2 Agent turns)
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
