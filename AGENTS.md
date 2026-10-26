# Repository rules

- Treat raw datasets and user-owned files as read-only. Never edit, move, or delete `sources/`; never overwrite `D:\DrowningDemo` or its artifacts.
- Keep this implementation separate from `D:\AI\codex-cli-handoff`; that directory is the specification handoff.
- Route real training through Kaggle. Do not add an implicit local CPU fallback. Local tests may use small synthetic fixtures only and must not be described as trained model results.
- Put database, Drive, Kaggle, and dataset-root settings in ignored `.env` or provider-local auth files. Keep only placeholders in `.env.example`; never log secret values.
- An unmapped or ambiguous source label defaults to review. Do not infer label meaning from substrings, IDs, or examples.
- Update `IMPLEMENTATION_STATUS.md` at each milestone and distinguish implementation, tests run, real-data/cloud checks, and user acceptance.
- Review is advisory by the user's decision (2026-10-26, `configs/review_policy.yaml` `gating.mode`): an unconfirmed mapping or group never blocks smoke/train, but every report and model must say the source is unconfirmed. Unmapped/ambiguous labels still default to review and stay out of training. Never approve on the user's behalf without an AI/human reviewer record.
