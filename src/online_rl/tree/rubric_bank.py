"""Same-reason REUSE vs NEW rubric generation (per case, per outcome stratum).

Within ONE case, decided trajectories are processed in a fixed order (leaf id) separately for the FAILURE and the SUCCESS
stratum, each with its own bank of canonical rubrics; a trajectory is never matched to a rubric of the other stratum and
no bank carries over to another case.

- The first trajectory of a stratum (and any trajectory meeting an empty bank) gets a NEW rubric from the normal
  rubric-generation call (generation section of the Rubric-Generator prompt).
- Every later trajectory first gets ONE MATCH-vs-NEW decision from the same Rubric-Generator model (full prompt package:
  generation section + "[Assign or create]" section, plus the current bank), logged and cached separately from
  generation. REUSE_EXISTING is accepted only with an existing rubric id and all six same-reason / membership checks
  true; the trajectory is then attached to that rubric and NO rubric is generated for it. Anything else (NEW, an unknown
  id, a failed or uncertain check) -> the normal rubric-generation call for this trajectory.
- Only a newly generated rubric gets the deterministic checks and the full Luna gate; a failed one never enters the bank
  (assignment CREATE_REJECTED). A REUSED_EXISTING trajectory triggers no gate call: its decision carries the checks.
- Verification is per member: the rubric's Stage-1 matching on that member's own path gives its targets; the normal
  Score-2 (failure) / Score-0 (success) intervention and K reruns give s_j (InterventionRun records as before).
- KEEP(R) iff max_j s_j >= min_R_O; the rubric's single effectiveness value is R_O(R) = max_j s_j (team decision,
  2026-10-07). A kept rubric enters response scoring and R_A ONCE, whatever its member count. Prompt-evolution J keeps
  using every member's own s_j.
Operational failures (truncated / non-parsing model output, transport errors) raise and are retried on resume; they are
never recorded as a rubric or as s_j = 0."""
from __future__ import annotations
import dataclasses, json
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.effectiveness import evaluate_rubric
from src.online_rl.tree.posthoc import TRANSCRIPT_FORMAT, generate_rubric, trajectory_message
from src.online_rl.tree.records import CanonicalSupport, Leaf, RubricAssignment, RubricEffectiveness, TrajectoryRubric, TrajectoryRubricInput

ASSIGNMENT_VERSION = 'assign_or_create_v1'
BATCH_VERSION = 'snapshot_match8_new_recheck_v1'
ASSIGNMENT_CACHE_VERSION = 'member_intervention_specs_v1'
MARKER = '[Assign or create]'
CHECKS = ('same_construct', 'applicability_visible', 'same_required_behavior', 'same_direction', 'score_semantics_apply', 'no_new_premise')
STRATA = (('failure', 0.0), ('success', 1.0))
VISIBLE = ('name', 'applicability_condition', 'required_behavior', 'scores')


class PromptPackageError(ValueError):
    """A Rubric-Generator prompt without its '[Assign or create]' section."""


def prompt_sections(text: str) -> tuple[str, str]:
    """(generation section, full package). CREATE uses the generation section; ASSIGN_OR_CREATE the full package."""
    if MARKER not in text:
        # A fixed generation prompt may intentionally contain no matching policy. Keep its
        # bytes untouched for CREATE/hash identity and supply the independent reuse contract
        # only to the matcher package.
        appendix_path = Path(__file__).resolve().parents[3] / 'prompts/sherpa/trajectory_rubric_generator_P0_aoc_v2.txt'
        appendix_text = appendix_path.read_text()
        if MARKER not in appendix_text: raise PromptPackageError(f'the matcher appendix has no {MARKER!r} section')
        appendix = appendix_text.split(MARKER, 1)[1].strip()
        return text.rstrip() + '\n', text.rstrip() + '\n\n' + MARKER + '\n\n' + appendix + '\n'
    return text.split(MARKER, 1)[0].rstrip() + '\n', text


def bank_view(bank: list[TrajectoryRubric]) -> list[dict]:
    from src.online_rl.tree.agent_state import visible_rubric
    return [{'rubric_id': r.rubric_id, **visible_rubric(r.rubric)} for r in bank]


def assign_message(tin: TrajectoryRubricInput, stratum: str, bank: list[TrajectoryRubric], *, member_interventions: bool = False) -> str:
    if member_interventions or any(r.intervention_instruction is not None for r in bank):
        tin = dataclasses.replace(tin, target_reply_id=None)
    return (f"{trajectory_message(tin)}\n\nCURRENT BANK ({stratum} trajectories of this case; {len(bank)} canonical rubrics):\n"
            f"{json.dumps(bank_view(bank), ensure_ascii=False, indent=1)}")


def _member_spec(rub: TrajectoryRubric, tin: TrajectoryRubricInput, assignment: str, decision: Optional[dict]):
    if assignment == 'CREATE_REJECTED' or rub.intervention_instruction is None: return None
    from src.online_rl.tree.member_intervention import make_member_intervention
    if assignment == 'CREATED_NEW':
        turn = (((rub.rubric or {}).get('source_provenance') or {}).get('analysis') or {}).get('selected_agent_turn_id')
        instruction, origin = rub.intervention_instruction, 'generated'
    elif assignment == 'REUSED_EXISTING':
        turn = (decision or {}).get('selected_agent_turn_id')
        instruction, origin = (decision or {}).get('intervention_instruction'), 'matched'
    else:
        raise ValueError(f'Unknown rubric assignment {assignment!r}')
    return make_member_intervention(rub, tin, selected_agent_turn_id=turn,
                                    intervention_instruction=instruction, origin=origin)


def _validate_member_decision(decision, bank, tin):
    if not match_problem(decision, bank):
        canonical = next(r for r in bank if r.rubric_id == decision['rubric_id'])
        _member_spec(canonical, tin, 'REUSED_EXISTING', decision)
    return decision


def match_problem(decision: dict, bank: list[TrajectoryRubric]) -> str:
    """'' when the decision is an acceptable REUSE_EXISTING, else why it is not (then a new rubric is generated)."""
    if decision.get('decision') != 'REUSE_EXISTING': return 'decision CREATE_NEW'
    if decision.get('rubric_id') not in {r.rubric_id for r in bank}: return f"REUSE_EXISTING names an unknown rubric id {decision.get('rubric_id')!r}"
    failed = [c for c in CHECKS if (decision.get('checks') or {}).get(c) is not True]
    return f"REUSE_EXISTING with failed checks {failed}: treated as uncertain -> CREATE_NEW" if failed else ''


def build_banks(case: dict, leaves: list[Leaf], tins: dict[str, TrajectoryRubricInput], prompt_text: str, prompt_sha256: str,
                generate_fn: Callable, assign_fn: Callable, gate_fn: Optional[Callable], cache: CallCache, parallel_strata: bool = False,
                source_stage1_for: Optional[Callable] = None, match_batch_size: Optional[int] = None) -> tuple[dict[str, TrajectoryRubric], list[RubricAssignment]]:
    """Rubrics generated for this case (id -> rubric, gate-passed or rejected) and one assignment per decided trajectory.
    The failure and success banks are independent (each stratum reads and writes only its own bank); parallel_strata
    (execution only) builds them concurrently and merges them in the fixed stratum order, so the result is identical."""
    create_text, package = prompt_sections(prompt_text)
    if match_batch_size is None:match_batch_size=getattr(assign_fn,'batch_size',1)
    if match_batch_size not in (1,8):raise ValueError('rubric match_batch_size must be 1 or 8')
    if match_batch_size==8 and not callable(getattr(assign_fn,'batch',None)):raise ValueError('Batch matching requires assign_fn.batch')

    def batched_stratum(stratum,o):
        ordered=sorted((l for l in leaves if l.outcome==o),key=lambda l:l.leaf_id)
        canon={};bank=[];assigned={};orders={l.leaf_id:k for k,l in enumerate(ordered)}
        def save(leaf,rid,kind,decision,note=''):
            assigned[leaf.leaf_id]=RubricAssignment(leaf.leaf_id,case['case_id'],stratum,orders[leaf.leaf_id],rid,kind,decision,note,sha256_json(tins[leaf.leaf_id].feedback),
                _member_spec(canon[rid],tins[leaf.leaf_id],kind,decision))
        def create(leaf,decision=None,note=''):
            rub=generate_rubric(create_text,prompt_sha256,tins[leaf.leaf_id],generate_fn,cache,gate_fn,
                               source_stage1=source_stage1_for(leaf) if source_stage1_for else None)
            canon[rub.rubric_id]=rub
            if rub.valid:bank.append(rub)
            else:note += ('; ' if note else '')+'generated rubric rejected: '+', '.join(rub.problems or _gate_reasons(rub.gate))
            save(leaf,rub.rubric_id,'CREATED_NEW' if rub.valid else 'CREATE_REJECTED',decision,note)
            return rub.valid
        def match(group):
            snapshot=list(bank)
            identity={'version':BATCH_VERSION,'case':case['case_id'],'stratum':stratum,'prompt':prompt_sha256,
                      'assignment_cache_version':ASSIGNMENT_CACHE_VERSION,
                      'tins':[[l.leaf_id,tins[l.leaf_id].sha256] for l in group],
                      'bank':sha256_json(bank_view(snapshot)),'transcript':TRANSCRIPT_FORMAT,
                      'model':getattr(assign_fn,'model',None),'settings':getattr(assign_fn,'settings',None)}
            def validate_result(result):
                if not isinstance(result,dict) or not isinstance(result.get('decisions'),dict) or set(result['decisions'])!={l.leaf_id for l in group}:
                    raise RuntimeError('Batch match returned incomplete trajectory IDs')
                for leaf in group:
                    _validate_member_decision(result['decisions'][leaf.leaf_id],snapshot,tins[leaf.leaf_id])
                return result
            result=validate_result(cache.call('rubric_match_batch',identity,
                lambda:validate_result(assign_fn.batch(package,[tins[l.leaf_id] for l in group],snapshot))))
            decisions=result['decisions']
            if set(decisions)!={l.leaf_id for l in group}:raise RuntimeError('Batch match returned incomplete trajectory IDs')
            remaining=[]
            for leaf in group:
                decision=decisions[leaf.leaf_id];note=match_problem(decision,snapshot)
                if not note:save(leaf,decision['rubric_id'],'REUSED_EXISTING',decision)
                else:remaining.append((leaf,decision,note))
            return remaining
        # Both strata create their first rubric unconditionally. Later calls always use actual, gate-passed anchors.
        if not ordered:return canon,[]
        create(ordered[0]);cursor=1
        while cursor<len(ordered):
            group=ordered[cursor:cursor+match_batch_size];cursor+=len(group)
            if bank:pending=match(group)
            else:pending=[(l,None,'') for l in group]
            while pending:
                leaf,decision,note=pending.pop(0)
                added=create(leaf,decision,note)
                # NEW members are compared together against the expanded frozen bank, not assigned blindly to their leader.
                if added and pending:pending=match([l for l,_,_ in pending])
        return canon,[assigned[l.leaf_id] for l in ordered]

    def stratum_bank(stratum, o):
        if match_batch_size==8:return batched_stratum(stratum,o)
        canon: dict[str, TrajectoryRubric] = {}; out: list[RubricAssignment] = []
        bank: list[TrajectoryRubric] = []
        for k, leaf in enumerate(sorted((l for l in leaves if l.outcome == o), key=lambda l: l.leaf_id)):
            tin = tins[leaf.leaf_id]; fb = sha256_json(tin.feedback); decision, note = None, ''
            if bank:
                ident = {'tin': tin.sha256, 'prompt': prompt_sha256, 'version': ASSIGNMENT_VERSION, 'stratum': stratum, 'transcript': TRANSCRIPT_FORMAT,
                         'assignment_cache_version': ASSIGNMENT_CACHE_VERSION,
                         'bank': sha256_json(bank_view(bank)), 'model': getattr(assign_fn, 'model', None), 'settings': getattr(assign_fn, 'settings', None)}
                decision = _validate_member_decision(cache.call('rubric_match', ident,
                    lambda tin=tin, bank=list(bank): _validate_member_decision(assign_fn(package, assign_message(tin, stratum, bank,
                        member_interventions=bool(getattr(assign_fn,'member_interventions',False)))),bank,tin)),bank,tin)
                note = match_problem(decision, bank)
                if not note:
                    out.append(RubricAssignment(leaf.leaf_id, case['case_id'], stratum, k, decision['rubric_id'], 'REUSED_EXISTING', decision, '', fb,
                        _member_spec(canon[decision['rubric_id']],tin,'REUSED_EXISTING',decision))); continue
            rub = generate_rubric(create_text, prompt_sha256, tin, generate_fn, cache, gate_fn,          # the normal generation call
                                  source_stage1=source_stage1_for(leaf) if source_stage1_for else None)  # v2.1: source-self-applicability
            canon[rub.rubric_id] = rub
            if rub.valid: bank.append(rub)
            else: note = (note + '; ' if note else '') + 'generated rubric rejected: ' + ', '.join(rub.problems or _gate_reasons(rub.gate))
            out.append(RubricAssignment(leaf.leaf_id, case['case_id'], stratum, k, rub.rubric_id, 'CREATED_NEW' if rub.valid else 'CREATE_REJECTED',
                                        decision, note, fb, _member_spec(rub,tin,'CREATED_NEW' if rub.valid else 'CREATE_REJECTED',decision)))
        return canon, out
    if parallel_strata:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(STRATA)) as pool: parts = list(pool.map(lambda so: stratum_bank(*so), STRATA))
    else: parts = [stratum_bank(*so) for so in STRATA]
    canon = {k: v for c, _ in parts for k, v in c.items()}; out = [a for _, o in parts for a in o]
    if sorted(a.leaf_id for a in out) != sorted(l.leaf_id for l in leaves if l.outcome in (0.0, 1.0)):
        raise AssertionError('every decided trajectory must map to exactly one rubric')
    return canon, out


def _gate_reasons(gate: Optional[dict]) -> tuple[str, ...]:
    return tuple((gate or {}).get('reasons') or ['gate failed'])


def verify_members(case: dict, nodes: dict, leaves: dict[str, Leaf], canon: dict[str, TrajectoryRubric], assignments: list[RubricAssignment], *,
                   scorer: Callable, writer: Callable, continuation: Callable, reruns: int, run_seed: int, policy_sha256: str,
                   strategies: Strategies, cache: CallCache, workers: int = 1, member_workers: int = 1) -> tuple[list, dict[str, Optional[RubricEffectiveness]]]:
    """Every member trajectory of every active canonical rubric is verified on its own path (targets = Stage-1 matches of
    the canonical rubric there). Returns the source-path scores and leaf_id -> its verification (None for CREATE_REJECTED).
    The bank is final here, so members are independent; member_workers > 1 (execution only) verifies them concurrently and
    collects the results in assignment order."""
    def one(a):
        if a.assignment == 'CREATE_REJECTED': return [], None
        canonical = canon[a.canonical_rubric_id]
        if canonical.intervention_instruction is not None:
            from src.online_rl.tree.member_intervention import resolve_member_intervention
            resolve_member_intervention(canonical,leaves[a.leaf_id],member_spec=a.intervention_spec)
            member = canonical
            options = {'member_spec': a.intervention_spec}
        else:
            member = dataclasses.replace(canonical, leaf_id=a.leaf_id)
            options = {}
        return evaluate_rubric(case, nodes, leaves[a.leaf_id], member, scorer=scorer, writer=writer, continuation=continuation, reruns=reruns,
                               run_seed=run_seed, policy_sha256=policy_sha256, strategies=strategies, cache=cache, workers=workers, **options)
    if member_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(member_workers) as pool: results = list(pool.map(one, assignments))
    else: results = [one(a) for a in assignments]
    scores, effs = [], {}
    for a, (ss, eff) in zip(assignments, results): scores += ss; effs[a.leaf_id] = eff
    return scores, effs


def canonical_support(canon: dict[str, TrajectoryRubric], assignments: list[RubricAssignment], effs: dict[str, Optional[RubricEffectiveness]],
                      min_R_O: float) -> dict[str, CanonicalSupport]:
    """Per gate-passed rubric: member s_j (None when the rubric matches no reply on that member's path, exactly as a No-Match
    trajectory rubric before), KEEP iff max_j s_j >= min_R_O, R_O(R) = max_j s_j. Other statistics are recorded for audit only."""
    if not isinstance(min_R_O, (int, float)): raise ValueError(f'selection.min_R_O must be a number, got {min_R_O!r}')
    out = {}
    for cid, rub in canon.items():
        if not rub.valid: continue
        ms = [a for a in assignments if a.canonical_rubric_id == cid]
        s = {a.leaf_id: (effs[a.leaf_id].R_O if effs.get(a.leaf_id) is not None else None) for a in ms}
        vals = [v for v in s.values() if v is not None]; sup = [v for v in vals if v >= min_R_O]
        mx = max(vals) if vals else None
        out[cid] = CanonicalSupport(cid, ms[0].stratum, len(ms), s, len(sup), len(sup) / len(ms), sum(vals) / len(vals) if vals else None,
                                    sum(sup) / len(sup) if sup else None, min(vals) if vals else None, mx, bool(sup), mx if sup else None)
    return out


def assignment_telemetry(canon: dict, assignments: list[RubricAssignment]) -> dict:
    """Consolidation counts only (no verification quantities)."""
    by = {s: [a for a in assignments if a.stratum == s] for s, _ in STRATA}
    return {'trajectories': len(assignments), **{f'{s}_trajectories': len(v) for s, v in by.items()},
            'created_new': sum(a.assignment == 'CREATED_NEW' for a in assignments), 'reused_existing': sum(a.assignment == 'REUSED_EXISTING' for a in assignments),
            'create_rejected': sum(a.assignment == 'CREATE_REJECTED' for a in assignments),
            'generation_calls': sum(a.assignment != 'REUSED_EXISTING' for a in assignments), 'match_decisions': sum(a.decision is not None for a in assignments),
            'gate_pass': sum(1 for r in canon.values() if r.valid), 'gate_reject': sum(1 for r in canon.values() if not r.valid)}


def bank_telemetry(canon: dict, assignments: list[RubricAssignment], support: dict[str, CanonicalSupport]) -> dict:
    by = {s: [a for a in assignments if a.stratum == s] for s, _ in STRATA}
    sizes = [v.member_count for v in support.values()]
    return {'trajectories': len(assignments), **{f'{s}_trajectories': len(v) for s, v in by.items()},
            'created_new': sum(a.assignment == 'CREATED_NEW' for a in assignments), 'reused_existing': sum(a.assignment == 'REUSED_EXISTING' for a in assignments),
            'create_rejected': sum(a.assignment == 'CREATE_REJECTED' for a in assignments),
            'generation_calls': sum(a.assignment != 'REUSED_EXISTING' for a in assignments), 'generation_calls_saved': sum(a.assignment == 'REUSED_EXISTING' for a in assignments),
            'match_decisions': sum(a.decision is not None for a in assignments),
            'reuse_attempts_not_accepted': sum(bool(a.decision) and a.decision.get('decision') == 'REUSE_EXISTING' and a.assignment != 'REUSED_EXISTING' for a in assignments),
            'active_canonical': len(support), 'kept_canonical': sum(v.retained for v in support.values()),
            'gate_pass': sum(1 for r in canon.values() if r.valid), 'gate_reject': sum(1 for r in canon.values() if not r.valid),
            'kept_success': sum(v.retained and v.stratum == 'success' for v in support.values()),
            'kept_failure': sum(v.retained and v.stratum == 'failure' for v in support.values()),
            'member_s': sorted(x for v in support.values() for x in v.member_s.values() if x is not None),
            'members_without_target': sum(x is None for v in support.values() for x in v.member_s.values()),
            'compression_ratio': (len(assignments) / len(support)) if support else None,
            'max_members': max(sizes) if sizes else 0, 'singletons': sum(n == 1 for n in sizes)}
