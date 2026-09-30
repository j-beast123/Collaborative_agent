# clinicalqa-en

English implementation of the **Self-Evolving Rubrics for Collaborative
Clinical Agents** experiments.

This project studies whether a clinical agent can collaborate appropriately
with a clinician when evidence, claims, and uncertainty change across a
multi-turn interaction. The goal is not only to improve final answer accuracy.
The agent should also learn when to maintain its current judgment, when to
revise it in response to stronger evidence, and when to request clarification
instead of changing its belief prematurely.

The English project is being rebuilt from `clinicalqa-ko` while adopting the
updated evaluation and reward design described below.

## Research Goal

Standalone clinical competence and collaboration capability are different
properties. An agent may answer a question correctly on its own but still
respond poorly during collaboration: it may accept unsupported pressure,
ignore new evidence, or change a well-supported judgment without justification.
Conversely, an initially incorrect agent should be able to revise its judgment
when a collaborator provides better evidence.

The target capability is:

```text
Clinical competence
+ context-appropriate interaction
+ effective state transition
-> robust collaboration
```

The central claim is:

> Collaborative agents should learn not only what answer is correct, but how to
> interact appropriately and effectively as evidence and partner claims evolve.

## Evaluation Signals

The project keeps four related but distinct concepts separate.

### Correctness

Correctness concerns whether the agent's clinical answer matches the task
answer or reference. It is not a definition of interaction quality.

### Rubric Adherence

Rubric adherence measures whether an observed response performed the behavior
described by a particular rubric. A rubric uses an ordinal scale:

```text
0 = clear violation
1 = partial or incomplete adherence
2 = ideal behavior
```

The score is local to the rubric and the observed response. It must not be
treated as final correctness or as effectiveness.

### Behavioral Appropriateness

Appropriateness asks:

> Given the evidence and epistemic situation available immediately before this
> response, was the interaction behavior appropriate?

The current design uses a turn-local situation assessment. The evaluator sees
only the history up to `h_t`, then produces a short evidence-grounded reason
followed by compact Yes/No judgments:

```text
context -> reason -> Yes/No situation assessment
```

The reason and decision are checked for internal consistency. A contradictory
assessment is treated as evaluator uncertainty or error, not automatically as
evidence that the agent behaved inappropriately.

Appropriateness is conceptually based on the consistency between the
context-derived epistemic judgment and the behavior defined by applicable
rubrics. The exact numeric mapping from the judgments and rubric scores to a
reward remains an empirical design choice and is not fixed in this repository
yet.

### Effectiveness

Effectiveness asks whether the response moved the interaction toward downstream
task success. Let `Y` be the final task outcome and define:

```text
V(h) = E[Y | h]
Eff_t = V(h_{t+1}) - V(h_t)
```

When `V(h)` is in `[0, 1]`, `Eff_t` is a dense signal in `[-1, 1]`. It measures
the change in expected downstream success after the response. It is not a
claim of a perfectly identified causal treatment effect.

### Final Outcome

Outcome is the terminal task result, such as final success or failure. A good
outcome does not prove that every intermediate behavior was appropriate, and a
failed outcome does not prove that every earlier response was inappropriate.

The intended separation is:

```text
rubric adherence != behavioral appropriateness
behavioral appropriateness != effectiveness
effectiveness != final outcome
correctness != behavioral appropriateness
```

## Self-Evolving Rubric Pipeline

Rubrics are discovered from actual interaction failures rather than being
limited to a fixed, predefined action taxonomy.

```text
1. Generate collaborative trajectories
2. Collect failed trajectories
3. Identify the interaction behavior closest to the failure
4. Generate a reusable rubric candidate
5. Define ideal behavior and observable 0/1/2 criteria
6. Validate the candidate with ideal-behavior intervention
7. Consolidate semantically redundant rubrics
8. Score applicable rubrics on new interactions
9. Update the agent with the validated signals
10. Collect new failures and evolve the rubric set
```

Rubrics should describe reusable interaction behavior, not memorization of a
specific disease, answer choice, or failed case.

## Situation Assessment

The current English design uses `Reason -> Yes/No` rather than a bare binary
judgment. The reason is short and evidence-grounded; it is used to audit the
assessment and check consistency, not as a free-form reward target.

The assessment must not use:

- future turns;
- final correctness or ground truth;
- the final task outcome;
- a later clinician response.

The assessment is an elicited epistemic judgment. It is not a direct reading of
the model's hidden internal belief.

If the reason and Yes/No decision are inconsistent, the assessment may be
regenerated once. If it remains inconsistent, the appropriateness signal for
that turn should be masked or ignored rather than assigning a negative agent
reward to an unreliable evaluator output.

The project intentionally does not introduce a fixed multi-class action space
such as `Maintain`, `Revise`, `Seek Evidence`, `Accept`, or `Reject` as the
central supervision structure. Context-dependent behavior should emerge from
failure-driven rubrics.

## Effectiveness Estimation

Whenever practical, value estimates should be obtained from future
continuation rollouts rather than asking a language model to guess a probability
without validation.

For a state `h`, generate `M` continuations and measure their terminal outcomes:

```text
V_hat(h) = (1 / M) * sum(Y^(m))
```

These empirical targets may later be used to train a value estimator `V_phi`.
Training can then use:

```text
Eff_t = V_phi(h_{t+1}) - V_phi(h_t)
```

The estimator must be validated on held-out states using additional rollouts.
Useful checks include MAE, Pearson and Spearman correlation, sign agreement,
value calibration, and downstream success by effectiveness bin. The desired
relationship is:

```text
higher estimated effectiveness -> higher empirical downstream success
```

## Rubric Validation

The original `delta_k` approach is not treated as the core reward or rubric
weighting mechanism in this version. Its useful principle is retained:

> Retain rubrics that contribute to actual task success, not merely rubrics that
> sound behaviorally desirable.

For rubric `k` at an applicable failure turn `t`, compare the original response
with an ideal-behavior intervention:

```text
original:     h_t -> original response -> h_{t+1}^{orig}
intervention: h_t -> ideal response   -> h_{t+1}^{ideal,k}

G_{t,k} = V(h_{t+1}^{ideal,k}) - V(h_{t+1}^{orig})
G_k     = average_t(G_{t,k})
```

`G_k` is a rubric validation statistic. It supports retain, review, or remove
decisions; it is not multiplied into the training reward.

This is distinct from the actual-response effectiveness reward:

```text
Eff_t -> quality of the response that actually occurred
G_k   -> whether the rubric's ideal intervention beats the original behavior
```

Validated rubrics are consolidated when they describe the same failure pattern.
A rubric may be removed because intervention validation fails, because it is
redundant, or because it becomes mastered after sufficient support and no
longer provides useful learning signal. Low support alone is not evidence of
mastery.

## Intended Reward Design

The conceptual training objective combines three signals:

```text
R = lambda_A * R_Appropriateness
  + lambda_E * R_Effectiveness
  + lambda_O * R_Outcome
```

The exact weights, thresholds, and mapping from situation judgments plus rubric
scores to appropriateness reward are intentionally left for ablation and
sensitivity experiments. In particular, the effectiveness estimator may be
noisy and should not be given an unjustified dominant weight.

Recommended comparisons include:

1. Outcome only
2. Appropriateness plus outcome
3. Effectiveness plus outcome
4. Appropriateness plus effectiveness plus outcome
5. The prior `delta_k`-based design
6. Rubrics without intervention validation
7. Rubrics without consolidation

## Repository Layout

```text
clinicalqa-en/
├── data/
│   ├── raw/              # immutable source data and snapshots
│   ├── interim/          # normalized and review-stage data
│   └── processed/        # experiment-ready English splits
├── metadata/             # schemas, versioning, labels, statistics
├── annotations/          # translation and clinical terminology policy
├── prompts/              # English generation and evaluation prompts
├── src/
│   ├── data/             # loading, normalization, and splitting
│   ├── generation/       # interaction and counterfactual generation
│   ├── evaluation/       # correctness, consistency, and signal calculation
│   ├── rubric/           # discovery, validation, and consolidation
│   └── models/           # local and API-backed model adapters
├── configs/              # dataset, model, path, and training settings
├── train/                # future policy/value training code
├── outputs/              # trajectories, failures, rubrics, and scores
└── tests/                # schema and pipeline safety tests
```

At the current stage, `prompts/` contains the initial English prompt set. The
generation code, data pipeline, evaluation modules, and training configuration
will be added incrementally rather than assuming that the Korean reward
implementation already matches the updated methodology.

## Current Prompt Set

- `prompts/agent.txt`: collaborative clinical agent behavior
- `prompts/user_simulator.txt`: clinician-side interaction generation
- `prompts/user_sim.txt`: compatibility alias for the user simulator
- `prompts/interaction_state.txt`: local reason-first situation assessment
- `prompts/reason_yn_consistency.txt`: consistency check for the elicited judgment
- `prompts/rubric_evaluator.txt`: failure-trajectory analysis and rubric candidate generation
- `prompts/rubric_manager_construct.txt`: candidate consolidation schema
- `prompts/rubric_manager.txt`: retained-rubric management policy
- `prompts/rubric_scorer.txt`: local, future-blind rubric adherence scoring

All structured evaluators are instructed to return machine-readable JSON. The
English prompts should remain explicit about evidence boundaries so that future
turns and final outcomes do not leak into local appropriateness judgments.

## Development Roadmap

1. Add the English dataset schema and source-data loader.
2. Add normalization, split generation, and dataset statistics.
3. Connect the English prompts to interaction generation.
4. Implement reason/Yes-No consistency handling in the evaluator pipeline.
5. Add rubric candidate construction and semantic consolidation.
6. Implement rollout-based value targets and held-out effectiveness validation.
7. Implement ideal-behavior intervention and `G_k` rubric validation.
8. Add separate appropriateness, effectiveness, and outcome reward paths.
9. Add ablations and reproducible experiment manifests.

## Experiment 1: Belief Readout Fidelity

The first experiment evaluates whether an elicitation method faithfully
recovers the agent's stable standalone judgment. It does not treat clinical
correctness as the primary measure of readout quality.

For each case, run the same agent independently three times from the original
question with no previous conversation. Define:

```text
B_ref = the option selected identically in all three standalone runs
```

Cases are classified as:

```text
stable_correct   = same option 3/3 and equal to ground truth
stable_incorrect = same option 3/3 and different from ground truth
unstable         = the three selected options are not identical
```

Zero correct runs alone do not define `stable_incorrect`; the three outputs
must be identical. The ground truth is used only to label these secondary
correctness groups.

For stable cases, run three semantically equivalent, independently executed
belief probes (`belief_probe_A/B/C.txt`). Each probe sees only the original case
and options. It must output a reason and an option-level `current_judgment`.
The primary metric is:

```text
belief readout fidelity = mean(B_probe == B_ref)
```

Report it separately for stable-correct, stable-incorrect, and all stable cases.
Also report paraphrase agreement, repeated-probe agreement, reason-decision
consistency, the `B_ref`/`B_probe` confusion matrix, and ground-truth accuracy
as a secondary statistic. Do not claim to recover a true hidden belief. The
intended claim is that the elicited readout recovers the model's stable
standalone judgment.

## Scope Boundary

This repository is an English experimental implementation. It should not claim
that a rubric is useful because an evaluator prefers its wording, or claim that
an interaction was appropriate because the final answer happened to be correct.
All such claims require local behavioral evidence, downstream effectiveness
evidence, or both, depending on the claim being made.

## Experiment 2: Effectiveness / Expected-Success Prediction

Effectiveness is implemented independently from behavioral appropriateness. For
an interaction state `h`, define:

```text
V(h) = E[Y | h]
Eff_t = V(h_{t+1}) - V(h_t)
```

The first milestone is empirical and rollout-based. For every state, generate
`M` controlled future continuations from exactly the same state and estimate:

```text
V_hat(h) = successful continuations / M
```

Rollout records preserve the case, state, seed, model, simulator, prompt
version, generation settings, final answer, and correctness. Comparisons must
hold the continuation policy, simulator, turn budget, decoder, and final-answer
rule constant. Paired seeds should be used when practical.

The pilot configuration in `configs/effectiveness_pilot.yaml` starts with 100
English cases, three state types per case, and a configurable primary budget of
16 rollouts per state. The intended paired construction is:

```text
h_t -> ideal response -> h_good
h_t -> poor response  -> h_bad
```

followed by:

```text
Eff_good = V_hat(h_good) - V_hat(h_t)
Eff_bad  = V_hat(h_bad) - V_hat(h_t)
```

All derived states from one clinical case must remain in the same train/dev/test
split. This prevents the value predictor from learning only question difficulty
and tests whether it can rank interaction states within the same case.

The first analysis should inspect target distributions, repeated-batch
stability, positive/neutral/negative transitions, and rollout budgets 4, 8, 16,
and 32. A learned value predictor and RL integration are deliberately disabled
until the empirical targets are shown to be informative. `Eff_t` is the
actual-response training signal; the later rubric intervention statistic `G_k`
is a retain/remove validation statistic and must not be multiplied into the
training reward.

The implementation lives in `src/effectiveness/`:

- `empirical_value.py`: rollout success counts, value, and uncertainty
- `rollout_runner.py`: controlled continuation interface
- `state_builder.py`: leakage-safe interaction-state records
- `paired_states.py`: paired state and Delta-V records
- `effectiveness.py`: `Eff_t` and `G_k` calculations
- `metrics.py`: value, sign, and within-case ranking metrics
- `value_dataset.py`: leakage-safe state text, targets, and case-level splits
- `value_predictor.py`: `V_phi(h)` encoder and binomial/MSE objectives

The predictor is a model scaffold at this stage. Enabling its training does not
replace empirical target validation: rollout-derived `V_hat(h)` targets must be
inspected for variation and stability before interpreting `V_phi` or using its
differences as a training reward.
