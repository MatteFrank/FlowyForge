# Agent Development Notes

- Put reusable package code in `src/flowyforge/`.
- Put prototypes, scratch code, and external examples under `prototypes/`.
- Do not commit datasets, checkpoints, or heavy generated outputs.
- Keep APIs small, explicit, and easy to test.
- Use YAML configs for environments, tasks, and models.
- Every plugin should have at least a minimal smoke test.
- Run `pytest tests/` before committing.
- Do not add heavyweight orchestration dependencies until the project needs them.

