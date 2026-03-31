# Agent Instructions

- Repo-local agent metadata lives in `REPO_META.yaml`.
- Use `REPO_META.yaml` as the local source of truth for build/test entrypoints, owned paths, and allowed automated change classes.
- `develop` is the integration branch for normal work.
- `master` is release-only.
- Prioritize correctness, backward compatibility, task-contract stability, shape and dtype assumptions, and performance regressions.
