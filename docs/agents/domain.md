# Domain Docs

How the skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`docs/GLOSSARY.md`**: the project's single glossary: every term used in openftgs-xr and its definition, grouped by area. For now most of it is the vocabulary of openftgs, which openftgs-xr reuses; openftgs-xr's own terms are added as its spec is written and implemented.
- **`docs/modules/<module>/`**: the module docs of the module you are about to work in, if any: how it meets §2, its internal seams and invariants, and the Q-n rows it relies on. They explain the spec; they never override it.

## File structure

An example of the layout; a module has docs only once someone writes them.

```
/
└── docs/
    ├── GLOSSARY.md
    └── modules/
        ├── module1/
        |    ├── subject1.md
        |    └── subject2.md
        └── module2/
            └── subject3.md
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `docs/GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

## Create or update the glossary's vocabulary

If the domain concept you need isn't in the glossary then call the skill `/domain-documenting` to add it. It edits on a branch, never as uncommitted changes on `main` (see its "Where the edits go").
