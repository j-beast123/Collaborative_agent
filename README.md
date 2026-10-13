# SHERPA: Partner-Guided Rubric Learning

Code, prompts, configuration, and saved artifacts for clinical collaboration with partner-guided rubric rewards.

## Start here

- [Package contents and model requirements](PACKAGE.md)
- [Installation, training, and evaluation](docs/reproducibility.md)
- [Framework components and source files](docs/paper_implementation_map.md)
- [Stored dialogue, reflection, and rubric example](docs/partner_reflection_example.md)

## File structure

```text
configs/online_rl/           Training and evaluation configuration
prompts/                    Runtime prompt files and readable prompt documentation
src/online_rl/tree/         Rollout, reflection, rubric bank, verification, scoring
src/online_rl/integrated/   GRPO and continuation utilities
scripts/online_rl/          Training, evaluation, and monitoring entry points
artifacts/reported_checkpoint/
                            Adapter, provenance, data identities, saved predictions
tests/                     Contract and artifact tests
docs/                      Reproduction guide and implementation map
```

Other retained source files support older experiments. The linked configuration and implementation map identify this package's execution path; files have not been relocated or removed to avoid breaking imports.

## Local checks

```bash
bash scripts/verify_package.sh
bash run_paper.sh
bash run_paper.sh --full
```

These commands perform checks or preflight only. They do not start training or paid inference.

## Training and evaluation

Configure model services, dependencies, and private credentials as described in the reproduction guide before launching.

```bash
# Use the packaged reported-checkpoint schedule
bash run_paper.sh --launch

# Use the separate 120-case schedule
bash run_paper.sh --full --launch

# Check the saved evaluation artifact without inference
python scripts/online_rl/eval_checkpoint.py
```

The two training configurations use separate output directories. They do not resume each other. Preserve configuration and prompt hashes when reproducing a run.

Saved physician final-decision accuracy is **442/622 (71.06%)**, with no unparsed FINAL decisions. This is a stored result, not a result from a new execution of these commands.

## Protocol and documentation

The configuration and frozen provenance specify model roles, prompts, stopping rules, reruns, rewards, and optimization. Readable prompt presentations do not replace the hashed runtime prompts. Historical development notes are excluded from this anonymous package.

Local artifact and contract checks do not establish successful end-to-end GPU/API execution. Never commit credentials or private environment files.
