# docs/GLOSSARY.md Format

Where the glossary lives, and that there is only one, is in [SKILL.md](SKILL.md). This file describes its shape, taken from the glossary as it stands.

## Structure

```md
# openftgs-xr

{Intro paragraphs: what the project is (and openftgs, whose vocabulary it reuses), and what this glossary fixes (the words used in the spec, the milestones, the questions file and commit messages). It says what each thing is; shapes, units, equations and file layouts belong to the spec.}

Conventions: {which names are canonical: the spec's symbols and its normative tensor names; identifiers fixed by the §2 public API, which keep their spelling in code; and the rule that where the spec and a research paper name one thing differently, the spec's name is canonical and the paper's goes under _Avoid_.}

## Language

### Representation

**Primitive**:
One 4D Gaussian of the model, identified by its row index i, the same across all nine stored tensors; row order is deterministic, kept rows staying in order and new rows going last. openftgs §1.1, §1.3 Growth
_Avoid_: Gaussian, 4D Gaussian, splat, point, particle

**Base opacity σ**:
A primitive's time-independent opacity, the sigmoid of its stored logit; the opacity that the sampling score and the dead test read. openftgs E3
_Avoid_: opacity (unqualified, where the effective opacity could be meant), peak opacity, static opacity

### Time

**Iteration it**:
The index of a training iteration, counting completed optimizer steps from 1 to T_it (30,000 by default). openftgs D-41
_Avoid_: step (reserved for Adam's per-tensor step counter), epoch, tick
```

Never drop or rewrite the intro and the Conventions paragraph when adding terms: they say which names win.

## Entries

- **Term line.** `**Term**:`, with the spec's symbol after the name where it has one (`**Base opacity σ**:`).
- **Definition.** One or two sentences, saying what the thing IS, not what it does or how it is computed. A semicolon-joined qualifying clause is allowed when it pins the boundary between this term and a neighbouring one ("the opacity that the sampling score and the dead test read").
- **Reference.** End the definition with the spec items that define the term (`§1`, `E-n`, `D-n`), separated from the text by a space. An openftgs term cites the openftgs spec, its list led by the `openftgs` prefix (`openftgs §1.1, §1.3 Growth`), and any Q-n row it cites carries the prefix too (`openftgs Q-214`). A process term with no spec item cites where it is defined instead (`CLAUDE.md`, `docs/prd/milestones.md`).
- **_Avoid_ line.** Always present. List the other words for the same concept, including a research paper's name for it where the spec uses another, and say in parentheses why one is reserved when it means something else here ("step (reserved for Adam's per-tensor step counter)").

## Rules

- **Be opinionated.** When multiple words exist for the same concept, pick the best one and list the others under `_Avoid_`. A term the spec defines keeps the spec's name.
- **Names, not implementations.** The spec's normative names (its symbols, its tensor names, the §2 identifiers) may appear in a definition, since the Conventions paragraph makes them canonical. Implementation choices may not: private function names, file layouts, algorithms the spec leaves open. Those belong in module docs.
- **Only include terms specific to this project.** General programming concepts (timeouts, error types, utility patterns) don't belong even if the project uses them extensively. Before adding a term, ask: is this a concept unique to openftgs-xr, its spec or its prd process, or a general programming concept? Only the former belongs.
- **File it under its area.** Put each term under the `###` area it belongs to (Representation, Time, Training, Initialization, Rendering, Data, Files, Evaluation, Spec process, Workflow). Add a new area only when a cluster of terms fits none of them. The glossary's Differential harness area holds openftgs terms only; openftgs-xr has no harness, so file no new term there.
