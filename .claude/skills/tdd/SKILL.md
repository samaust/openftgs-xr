---
name: tdd
description: "Test-driven development: the red → green loop, one test at a time, writing each test from its spec §3 row at the public interface (§2, or a signature recorded as a Q-n row). Use when building or fixing code test-first, or when the owner mentions TDD or red-green."
---

# Test-Driven Development

TDD is the red → green loop. This skill is the reference that makes that loop produce tests worth keeping: where tests go, where their expected values come from, the house rules, the anti-patterns, and the rules of the loop. Every section applies on every cycle: consult them before and during the loop, not after.

When exploring the codebase, read `docs/GLOSSARY.md` so test names and interface vocabulary match the project's domain language.

## Guidelines

Follow [docs/agents/guidelines.md](../../../docs/agents/guidelines.md). A test whose expected value depends on a choice the spec doesn't make is written against the provisional decision of its Q-n row, as CLAUDE.md says (record the row first if the gap is new), and cites the row; log every source you read to settle an expected value.

## What a good test is

Spec tests verify behavior through public interfaces, not implementation details. Code can change entirely; spec tests shouldn't. A good test reads like the spec item it pins: its name or docstring cites the §3 ID and states the property (in openftgs, "U-13: with n = 2 the new opacity is 1 − √(1 − σ)"), and it survives refactors because it doesn't care about internal structure.

See [tests.md](tests.md) for examples and [mocking.md](mocking.md) for what to stand in for at the system boundaries.

## Seams: where tests go

A **seam** is the public boundary you test at: the interface where you observe behavior without reaching inside. Tests live at seams, never against internals.

In this repository the public seams are the public API: the signatures of spec §2 plus the additions recorded as Q-n rows (CLAUDE.md, "Working from the openftgs-xr spec"). The tests that pin them are the §3 items (U-n and the like) that the PRD or ticket lists, each with its pass criterion. Use those and don't ask again. A test the spec doesn't list (a regression test, an edge case the PRD agreed) goes at a public interface too, with its own independent source of truth.

Where a §3 row targets something the spec names but gives no signature for (a function of a §2 crate, say), propose the signature as a provisional Q-n row, one row per module, and test at it; carry on with the provisional signature instead of waiting for the owner. Changing a signature §2 lists is never a test-driven decision: stop and ask, or, as a subagent, stop and report it. When the shape of an interface is in question, call the Skill tool with "codebase-design" for the vocabulary (it is a reference to consult, not a session to run).

## Private tests

A private helper (not re-exported by its crate's `lib.rs`: `pub(crate)` or private) may have tests of its own when its numerics or invariants are worth pinning directly (in openftgs: the float64 path of its E11, the tie order of a k-NN search, a clamp in logit space). Such a test lives in the module's own `#[cfg(test)]` block, is named `<helper>_…` without a §3 ID, and follows every rule below except the one about surviving refactors: it changes or goes when its helper does. It never stands in for a §3 row, which is always pinned at the public interface when its target is public.

## Where expected values come from

Write each test from its row in spec §3, not from the code: the cases, the expected values and the tolerances all come from the row. Independent sources of truth, in the order to prefer them (the examples come from openftgs):

- **A literal worked by hand** from the spec: σ = 0.75 with n = 2 gives 1 − √0.25 = 0.5.
- **A property the row states**: an inverse, a symmetry, an invariant (openftgs U-13: 1 − (1 − σ_new)^n = σ; openftgs U-2: symmetric about μt).
- **A closed form against autograd** (openftgs U-1: the Jacobians I, (t − μt)I and −v; `torch.autograd.gradcheck`).
- **An independent oracle the spec names**: the reference backend against gsplat, an independent float64 evaluation (openftgs U-13, U-34), scikit-image or OpenCV as reference oracles (openftgs U-9, U-34 to U-36). These are the intended pattern, not tautologies, as long as the oracle shares no code with what it checks.

## House rules

- Every test file starts with `// SPDX-License-Identifier: Apache-2.0`.
- Put every test in its lane: `cpu` tests run under plain `cargo test`; a test of another lane (`gpu`, `xr`, `device`, `network`, `dataset`) is `#[ignore = "<lane>"]` and its name starts with the lane's prefix (`gpu_`, `xr_`, `dv_`, `net_`, `ds_`), so `cargo test -- --ignored <prefix>` runs that lane. Spec §3 fixes the lanes; until it does, these are the house rule.
- Compare values with the tolerances of the test's §3 row, passed explicitly.
- Draw random inputs within the row's stated domain, and pass any random generator through the public signatures rather than seeding or patching a global one.

Until spec §3 fixes openftgs-xr's own, the rest of the openftgs house rules are a guide, not a rule, read in Rust terms: `cpu` tests on the engine-free crates against hand-worked literals and f64 oracles, comparisons with the row's tolerance passed explicitly (`approx`), and property-based variants (`proptest`) where the spec asks for them.

## Anti-patterns

- **Implementation-coupled**: a spec test that mocks internal collaborators, reaches a private helper, or verifies through a side channel (reading a stored tensor the public interface doesn't promise). The tell: the test breaks when you refactor but behavior hasn't changed. Private tests (above) are coupled by design and are kept apart from spec tests by their names.
- **Tautological**: the expected value is recomputed the way the code computes it (openftgs E11's formula rewritten in the test, a reference value produced by calling the code under test), so the test passes by construction and can never disagree with the code. Expected values come from the sources above.
- **All tests first**: writing all tests first, then all implementation. Bulk tests verify _imagined_ behavior: you test the _shape_ of things rather than behavior at the public interface, the tests go insensitive to real changes, and you commit to test structure before understanding the implementation. Work **one test at a time** instead: one test → one implementation → repeat, each test responding to what the last cycle taught you.

## Rules of the loop

- **Red before green.** Write the failing test first, then only enough code to pass it. Don't anticipate future tests or add speculative features.
- **One test at a time.** One seam, one test, one minimal implementation per cycle.
- **Refactoring is not part of the loop.** It belongs to the review stage (see the `two-axis-review` skill), not the red → green implementation cycle.
