---
name: setup-skills
description: "Set up GitHub for this repository's skills: create or update the triage, category and skill labels with the bundled script, and check that the skill configuration files are in place. Run once per repository, and again whenever the label set changes."
disable-model-invocation: true
---

# Setup Skills

The skills in `.claude/skills/` work against GitHub Issues at github.com/samaust/openftgs-xr, through the `gh` CLI. Their configuration is already decided and lives in the repository:

| File | What it fixes |
| --- | --- |
| `docs/agents/issue-tracker.md` | GitHub only: the `gh` operations, native sub-issues and blocked-by dependencies, and pull requests not being a request surface |
| `docs/agents/triage-labels.md` | the seven triage roles (five state, two category) and their label strings |
| `docs/agents/domain.md` | the single glossary `docs/GLOSSARY.md` and the module docs under `docs/modules/<module>/` (no ADRs) |
| `docs/agents/guidelines.md` | how skills and their subagents apply the guidelines rules |
| `CLAUDE.md`, "Agent skills" | pointers to the files above |

This skill doesn't re-decide any of it, and doesn't rewrite those files: they are edited directly when a choice changes. It creates the labels the skills apply, on GitHub, and checks that the files are in place.

## Guidelines

Follow [docs/agents/guidelines.md](../../../docs/agents/guidelines.md). This skill reads nothing outside this repository and its own GitHub labels; its `gh` calls target this repository only. It never leaves files changed on `main`: a repair goes on a branch (step 3).

## Labels

| Label | Kind | Applied by |
| --- | --- | --- |
| `needs-triage` | triage role (state) | `triage` |
| `needs-info` | triage role (state) | `triage` |
| `ready-for-agent` | triage role (state) | `conversation-to-prd`, `prd-to-tickets`, `triage` |
| `ready-for-human` | triage role (state) | `triage` |
| `wontfix` | triage role (state) | `triage` |
| `bug`, `enhancement` | triage role (category) | `triage` |
| `q-n` | Q-n issue (no category) | the implement skills, for Q-n decision issues; any skill that records Q-n candidates outside a PRD or agent brief (`grilling`, `codebase-design`, `two-axis-review` run on `main`), for Q-n candidates issues |
| `out-of-scope` | out-of-scope record | `triage`, on the canonical issue of a rejected enhancement |
| `skill:conversation-to-prd` | skill label | `conversation-to-prd` |
| `skill:prd-to-tickets` | skill label | `prd-to-tickets` |

The triage-role strings come from `docs/agents/triage-labels.md`, and the script's list must match it: change both together.

## Process

1. **Check the target.** Run `gh auth status`, and check that `git remote -v` points at github.com/samaust/openftgs-xr. If either fails, stop and tell the owner.
2. **Create the labels.** From the repository root, run the bundled script [scripts/create-labels.sh](scripts/create-labels.sh) as `.claude/skills/setup-skills/scripts/create-labels.sh`. It creates each label in the table, or updates it in place, so it is safe to run again. Run it; don't read it or retype its commands. Then run `gh label list` and confirm that every label in the table exists.
3. **Check the configuration files.** Confirm that every file in the first table exists and that `CLAUDE.md` has its "Agent skills" block pointing at them. If one is missing, restore it from git history on a short branch `setup-skills-repair` from `main`, in its own worktree ("Branches live in worktrees" in `docs/agents/guidelines.md`; start from a clean working tree; ask the owner if it isn't one), and ask the owner before merging it into `main` with `git merge --no-ff`.
4. **Report** the labels created or updated, the files checked, and anything restored or still missing.
