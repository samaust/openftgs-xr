# Implementation questions and provisional decisions

Gaps found in the spec during implementation. Every row is provisional until the owner confirms it; confirmed rows may return to the spec as new D-n decisions, through a spec edit the owner approves ("Editing the spec" in `docs/agents/guidelines.md`). Code that relies on a row carries a `# Q-n` comment.

Status is one of: `provisional` (a new row), `confirmed` (the owner accepted the provisional decision), `changed (#n)` (the owner chose otherwise; the row's decision is updated and the code follows in issue #n: the row's own ticket when the owner answered during its build, otherwise an issue implemented on its own), or `in spec (D-n)` (returned to the spec).

| ID | Spec location | Gap | Provisional decision | Rationale | Alternatives | Detecting test | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
