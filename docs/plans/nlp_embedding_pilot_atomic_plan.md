# NLP Embedding Pilot Atomic Plan

Purpose: implement the first non-CV predictive task pilot from the platform
roadmap while keeping `neuriplo-tasks` contracts stable and avoiding broad
abstractions before a second NLP task exists.

## Phase 0: Branch And Scope

1. Work in `../neuriplo-platform` first.
2. Create or confirm a feature branch for the domain-contract work.
3. Do not edit untracked platform artifacts such as `data/` or `logs/`.
4. Scope the first non-CV pilot to NLP text embeddings only.

## Phase 1: Platform Contract

1. Add `contracts/task-domain-contract.md` or
   `contracts/nlp-embedding-contract.md` in `../neuriplo-platform`.
2. Define reserved task domains: `cv`, `nlp`, `audio`, `tabular`,
   `multimodal`, and `rl`.
3. Mark only `cv` as shipped.
4. Mark `nlp.embeddings` as the first planned pilot.
5. Define the NLP embeddings contract:
   - task strings
   - input contract
   - output/result contract
   - batching semantics
   - serving protocol: KServe V2
   - streaming support: no
   - backend expectations
   - example model configuration
6. Update `docs/architecture/production-roadmap.md` to reference the new
   contract.
7. Run `scripts/check_platform.py`.
8. Commit the platform contract work.

## Phase 2: Result Contract Skeleton

1. Switch to `/home/oli/repos/neuriplo-tasks`.
2. Create a feature branch from `develop`.
3. Add an `Embedding` result type to `include/neuriplo/tasks/core/result_types.hpp`.
4. Add `TaskType::Embedding`.
5. Add tests proving `Result` can hold and visit `Embedding`.
6. Build and run the focused result-type tests.
7. Commit the result-contract change.

## Phase 3: Embedding Task Implementation

1. Add `include/neuriplo/tasks/nlp/embedding_task.hpp`.
2. Add `src/nlp/embedding_task.cpp`.
3. Implement minimal behavior:
   - preprocess follows the platform text-input contract chosen in Phase 1
   - postprocess converts `[D]` or `[N,D]` float tensors into one `Embedding`
     result per batch item
4. Do not add generic NLP abstractions until a second NLP task exists.
5. Add `tests/test_embedding_task.cpp`.
6. Run the focused embedding test.
7. Commit the task implementation.

## Phase 4: Factory Routing

1. Add `Embedding` to the internal factory family enum.
2. Register aliases:
   - `embedding`
   - `textembedding`
   - `nlpembedding`
3. Add factory tests for:
   - direct alias
   - hyphen, underscore, space, and case normalization
   - expected `TaskType::Embedding`
4. Update README model list and feature bullets.
5. Add `export/nlp_embeddings/README.md`.
6. Update `export/README.md`.
7. Run factory and README tests.
8. Commit factory and docs.

## Phase 5: Batch Contract

1. Add tests for `[N,D]` embedding output.
2. Confirm `batchPostprocess` returns `N` results.
3. Document embedding batching in `docs/batch_support_matrix.md`.
4. Update `docs/ROADMAP.md` with the non-CV pilot status.
5. Run focused batch tests.
6. Commit batch documentation and tests.

## Phase 6: Full Local Gate

1. Run the clang-format check.
2. Run `cppcheck`.
3. Configure with tests and `-DWERROR=ON`.
4. Build.
5. Run `ctest --output-on-failure`.
6. Fix only failures caused by this branch.
7. Commit any fixes.

## Phase 7: Release Or Integration Decision

1. If this is contract groundwork only, push the branch and open a PR.
2. If it should become a published capability, update `CHANGELOG.md`.
3. Tag and release only when ready. Every tag must have a matching GitHub
   Release with notes from `CHANGELOG.md`.

## Phase 8: Platform Pin Follow-Up

1. Return to `../neuriplo-platform`.
2. Update `versions.yaml` to the exact `neuriplo-tasks` ref or tag.
3. Add compatibility-set notes for the NLP embeddings pilot.
4. Keep wording as pilot, planned, or experimental unless fully released.
5. Run `scripts/check_platform.py`.
6. Commit and push the platform follow-up.

## Stop Criteria

The work is complete when the platform defines the domain contract,
`neuriplo-tasks` has a tested NLP embedding pilot, docs match the implementation,
and platform pins reference the exact tasks ref.
