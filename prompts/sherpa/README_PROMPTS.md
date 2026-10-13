# Final reported checkpoint prompt map

| Role | Frozen reported checkpoint prompt |
|---|---|
| Agent | `prompts/agent.txt` |
| Doctor | `prompts/user_simulator_v4_final_label.txt` |
| Partner-side reflection | `prompts/sherpa/behavior_feedback_v5.txt` |
| Rubric generator | `prompts/sherpa/trajectory_rubric_generator_P0_aoc_v3.txt` |
| Observed-context addendum | `prompts/sherpa/observed_context_behavior_v2.txt` |
| Gate | `prompts/sherpa/luna_gate_v1.txt` |
| Behavior scorer | `prompts/sherpa/luna_scorer_v2.txt` |
| Success-to-score-0 rerun | `prompts/sherpa/guided_response_template_score0_v1.txt` |
| Failure-to-score-2 rerun | `prompts/sherpa/guided_response_template_score2_v2.txt` |
| Prompt reflection | `prompts/sherpa/tree_prompt_feedback_v3.txt` |
| Prompt refiner | `prompts/sherpa/tree_prompt_refiner_v3.txt` |

`trajectory_rubric_generator_readable.txt` presents the rubric-generator
interface with explicit Input and Output sections. The exact reported-checkpoint configuration
continues to reference the frozen `P0_aoc_v3` file so that the reported result's
protocol hash remains reproducible.
