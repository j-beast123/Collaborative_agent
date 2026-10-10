#!/usr/bin/env bash
# CPU regressions for the fixed-P0 max2 trainer and its active dependencies.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${SHERPA_PYTHON:-python}"
TEST_FILES=(
  tests/test_active_training_config.py
  tests/test_prepare_train120.py
  tests/test_agent_batch.py
  tests/test_agent_precommit.py
  tests/test_batched_generation.py
  tests/test_batched_runtime.py
  tests/test_cache_single_flight.py
  tests/test_call_cache_error_evidence.py
  tests/test_doctor_openings.py
  tests/test_doctor_post_final_reflection.py
  tests/test_doctor_reflection_training.py
  tests/test_doctor_v2.py
  tests/test_feedback_claim_grounding.py
  tests/test_five_gpu.py
  tests/test_gate_quote_catalog.py
  tests/test_gpu_pool.py
  tests/test_grpo_reward_audit.py
  tests/test_integrated_v1.py
  tests/test_live_gpu_monitor.py
  tests/test_member_intervention_bank.py
  tests/test_member_intervention_execution.py
  tests/test_member_intervention_matcher.py
  tests/test_member_intervention_resume.py
  tests/test_observed_context_rubric.py
  tests/test_parallel_rubric_strata.py
  tests/test_pilot_evidence_scorer.py
  tests/test_quality_gate_v2.py
  tests/test_rerun_turn_cache.py
  tests/test_reward_prompt_matrix.py::test_reward_is_the_configured_weighted_sum
  tests/test_reward_prompt_matrix.py::test_mixed_reward_example
  tests/test_reward_prompt_matrix.py::test_na_semantics_equal_the_reference_formula
  tests/test_reward_prompt_matrix.py::test_invalid_experiment_fails_fast
  tests/test_reward_prompt_matrix.py::test_fingerprint_separates_every_research_axis
  tests/test_reward_prompt_matrix.py::test_checkpoint_from_another_run_or_condition_is_refused
  tests/test_reward_prompt_matrix.py::test_fixed_prompt_runs_keep_p0_hash_across_batches
  tests/test_reward_prompt_matrix.py::test_physical_gpu_mapping_is_runtime_only
  tests/test_reward_prompt_matrix.py::test_worker_pins_its_logical_gpu_to_the_mapped_physical_device
  tests/test_reward_prompt_matrix.py::test_batch_guard_stops_before_the_update
  tests/test_reward_prompt_matrix.py::test_resume_with_a_different_theta0_is_refused
  tests/test_rubric_reuse.py
  tests/test_run_tree_wiring.py
  tests/test_scoring_batch.py
  tests/test_source_consistency.py
  tests/test_state_shared_scoring.py
  tests/test_shared_applicability.py
  tests/test_intervention_instruction_audit.py
  tests/test_instruction_alignment_generation.py
  tests/test_v23_launcher_wiring.py
  tests/test_supervision_handoff.py
  tests/test_cumulative_bank.py
  tests/test_cumulative_run.py
  tests/test_cumulative_trainer.py
  tests/test_cumulative_monitor.py
  tests/test_member_matcher_repair.py
  tests/test_idle_eval_priority.py
  tests/test_idle_eval_snapshot.py
  tests/test_idle_eval_worker.py
  tests/test_idle_eval_hooks.py
  tests/test_idle_eval_doctor_restart.py
  tests/test_idle_eval_supervisor.py
  tests/test_trace_logprobs.py
  tests/test_trajectory_schema.py
  tests/test_tree_na_reward.py
  tests/test_tree_v1.py
  tests/test_witness_protocol.py
  tests/test_witness_trainer.py
)
# Exclude historical contracts outside the active fixed-P0, shared-five-GPU run.
# Matrix tests above select fixed-mode behavior; P1/Pfinal fixtures require an old phase protocol.
LEGACY_TESTS=(
  # Four-GPU idle evaluation expects a checkpoint-0 report; active training has no evaluation.
  'tests/test_five_gpu.py::test_runner_commits_30_resumes_and_drains_eval_queue[False-False-False]'
  'tests/test_five_gpu.py::test_runner_commits_30_resumes_and_drains_eval_queue[False-True-False]'
  # These assertions require a historical 500-case dataset or compare all historical pilot configs.
  tests/test_gpu_pool.py::test_shared_config_allows_checkpoint_barrier_evaluation
  tests/test_run_tree_wiring.py::test_research_and_pilot_configs_pin_the_same_init_seed
  # Old guided prompt text and P1 reflection/fitness views are not used by this configuration.
  tests/test_tree_v1.py::test_intervention_backends_reuse_the_guided_precedent
  tests/test_tree_v1.py::test_configured_score2_template_uses_the_env_guided_call_shape
  tests/test_tree_v1.py::test_gate_failure_reasons_are_exposed_without_gold
  tests/test_witness_protocol.py::test_prompt_comparison_omits_incomplete_members_instead_of_using_zero
)
PYTEST_ARGS=()
for test in "${LEGACY_TESTS[@]}"; do PYTEST_ARGS+=("--deselect=$test"); done
exec "$PYTHON_BIN" -m pytest -q -p no:cacheprovider "${TEST_FILES[@]}" "${PYTEST_ARGS[@]}" "$@"
