# SHERPA online — anchored group learning

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
