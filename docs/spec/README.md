# Specification

- `openftgs-xr-spec.md`: the openftgs-xr spec. It is edited only with the owner's approval, and each edit adds a row below ("Editing the spec" in `docs/agents/guidelines.md`).
- `openftgs-spec.md`: a copy of the openftgs spec, the reference for the files openftgs writes. It is never edited, only replaced by a newer copy from openftgs.

## Revisions of openftgs-xr-spec.md

| Revision | Date | What changed |
| --- | --- | --- |
| 0 | 2026-10-29 | Skeleton: §0 with openftgs's notation and the document map; §1, §2, §3, Appendix A and Appendix B empty. |
| 1 | 2026-10-29 | §0: the package is a Rust workspace on Bevy and OpenXR with two builds, the Linux desktop player and the Meta Quest 3 app, and the language rule (Rust, WGSL, a minimal Android shim, Python only in dev tooling) is recorded; the rest unchanged. |
