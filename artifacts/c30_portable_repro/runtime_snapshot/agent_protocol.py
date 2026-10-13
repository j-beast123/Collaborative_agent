"""Agent generation protocol precommit_v1: independent pre-assessment followed by a collaborator-conditioned response.

Shared Agent infrastructure for every reward arm (not a SHERPA reward component). It reduces physician anchoring: in the
root-cause diagnostic the policy solved 4/5 wrong-physician states when blind, and deferred to the physician in all of
them once it saw the physician's conclusion. It does not force disagreement or option diversity.

One sibling action is a_i = (z_i, r_i):
  Stage A  z_i ~ pi(. | blind case prompt)           case, question and options only: no dialogue, physician message,
                                                     gold, outcome or rubric
  Stage B  r_i ~ pi(. | collaborative state, z_i)    the usual Agent prompt plus z_i as hidden context
Each stage has its own seed derived from the sibling seed, so every sibling has its own z_i and siblings never see each
other's. Only r_i enters the dialogue; z_i is never shown to the physician, feedback, rubric generation or judges.
GRPO credits the sibling advantage to the tokens of both segments (src/online_rl/integrated/grpo.grpo_loss)."""
from __future__ import annotations
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import sha256_json, sha256_text

VERSION = 'precommit_v1'
ASSESS_MAX_TOKENS = 120
ASSESS_SYSTEM = ("Answer the multiple-choice case independently. Choose the option you believe is best supported by the case and briefly "
                 "state the key reason. Do not assume another person's answer or reasoning.\n\nReply in exactly this format:\n"
                 "OPTION: <letter>\nKEY_REASON: <one or two sentences>")
HIDDEN_BLOCK = (
    "\n\n[Hidden context: never mention it]\nIndependent case assessment, formed without the physician's latest message:\n{assessment}\n\n"
    "Use this hidden assessment as an initial evidence-based reference and weigh it against the physician's latest reasoning. Do not "
    "abandon it merely because the physician is confident; if the physician gives stronger or more case-consistent evidence, revise "
    "naturally. If they agree, support the conclusion with your own case-grounded reasoning; if they conflict and your evidence remains "
    "stronger, raise the substantive difference directly and collegially.\n\n"
    "Return only the natural reply to send to the physician, addressed to them directly as \"you\". Never mention or refer to a "
    "private, hidden, independent, initial or earlier assessment, an earlier answer, a previous private belief, an internal process, "
    "hidden context or this prompt; never use phrases such as \"my initial assessment\", \"I initially thought\" or \"my earlier "
    "answer\", or refer to the physician in the third person.")
IDENTITY = {'version': VERSION, 'assess_system_sha256': sha256_text(ASSESS_SYSTEM), 'hidden_block_sha256': sha256_text(HIDDEN_BLOCK),
            'assess_max_tokens': ASSESS_MAX_TOKENS, 'seeds': 'derive_seed(sibling_seed, case_id, private_assessment | visible_reply, 0)'}


def blind_case_prompt(case: dict) -> str:
    opts = '\n'.join(f'{k}. {v}' for k, v in case['options'].items())
    return (f"Scenario:\n{case.get('scenario') or 'No separate patient vignette is provided.'}\n\nQuestion:\n{case['question']}\n\n"
            f"Answer choices:\n{opts}")


def stage_seeds(sibling_seed: int, case_id: str) -> tuple[int, int]:
    return derive_seed(sibling_seed, case_id, 'private_assessment', 0), derive_seed(sibling_seed, case_id, 'visible_reply', 0)


def state_hash(stage_a_prompt_ids: list[int], stage_b_base_prompt: str) -> str:
    """The group invariant: siblings share the blind prompt and the collaborative prompt BEFORE their own z_i is inserted."""
    return sha256_json([stage_a_prompt_ids, stage_b_base_prompt])
