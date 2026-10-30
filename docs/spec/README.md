# Specification

- `openftgs-xr-spec.md`: the openftgs-xr spec. It is edited only with the owner's approval, and each edit adds a row below ("Editing the spec" in `docs/agents/guidelines.md`).
- `openftgs-spec.md`: a copy of the openftgs spec, the reference for the files openftgs writes. It is never edited, only replaced by a newer copy from openftgs.

## Revisions of openftgs-xr-spec.md

| Revision | Date | What changed |
| --- | --- | --- |
| 0 | 2026-10-29 | Skeleton: §0 with openftgs's notation and the document map; §1, §2, §3, Appendix A and Appendix B empty. |
| 1 | 2026-10-29 | §0: the package is a Rust workspace on Bevy and OpenXR with two builds, the Linux desktop player and the Meta Quest 3 app, and the language rule (Rust, WGSL, a minimal Android shim, Python only in dev tooling) is recorded; the rest unchanged. |
| 2 | 2026-10-29 | The full specification: §0 (source rules, tags, notation, map), §1.1–§1.15 with E1–E16, §2.1–§2.7 (workspace, API, dependencies, performance, platform, standards, tests), §3.1–§3.6 (U-1 to U-35, R-1 to R-17, I-1 to I-17, Dv-1 to Dv-13, B-1 to B-9, A-1 to A-10), Appendices A (36 source rows), B (D-1 to D-49, with the owner's decisions of the 2026-10-29 grilling round), C (F-1 to F-34), D (bindings) and E (platform facts). Written from `docs/prd/spec-revision-2-plan.md`. |
| 3 | 2026-10-29 | Owner decisions after milestone M-1 (PRD #1): §0 allows reading permissive dependency source from the local cargo registry; §1.3 E9: single-timestamp instances do not count toward T\_len; §1.5: scenes in the config directory (§1.11), checkpoints loaded by path, no memory mapping; §2.1: the `Depends on` cells of `ftgsxr-core` (memmap2 removed), `ftgsxr-io` and `ftgsxr-testkit`, `alpha_at` gains `w_occ`, `open_model` reads into memory, `ModelBytes { Owned, Checkpoint }`, `load_splat_asset`'s checkpoint route; §2.3: glam 0.32, memmap2 removed; §2.7: `ftgs-adapter info`; §3.1 U-12's weight clause; Appendix B D-50, D-51. |
| 4 | 2026-10-29 | Owner decision on Q-5 (#26): §2.1 `TlsRoots` loses its `Platform` variant, leaving `WebPki`; Appendix B D-20 drops the platform verifier as a desktop-only option and records it as a deferred alternative. |
