# Out-of-Scope Record

Rejected feature requests are recorded as closed GitHub issues labelled `out-of-scope`, not as files in the repository: triage runs on `main`, and the tracker needs no branch or merge. The record serves two purposes:

1. **Institutional memory**: why a request was rejected, so the reasoning isn't lost when the issue is closed.
2. **Deduplication**: when a new request matches a prior rejection, triage surfaces the earlier decision instead of re-litigating it.

In this repository, scope is the spec. Most rejections read "not in the spec" and cite the item that bounds it. A request that should change the spec is not rejected here: it is the owner's decision, made as a spec edit they approve ("Editing the spec" in `docs/agents/guidelines.md`).

## The canonical issue

One issue per **concept**, not per request: the first request for a concept becomes its canonical issue. It carries the labels `wontfix` and `out-of-scope`, is closed as not planned (`gh issue close <n> --reason "not planned"`), and ends with a decision comment in this shape:

```markdown
## Out of scope: Screenshot feature

**Decision:** openftgs-xr has no screenshot feature.

**Reason:** the features are fixed by spec §2.
A screenshot feature is not planned in the current spec revision.

**Prior requests:**
- #42: "Add a take a screenshot feature"
```

Name the concept in the heading so that someone reading the list of closed `out-of-scope` issues recognises it without opening it.

### Writing the reason

The reason is substantive and durable: not "we don't want this" but why, citing the spec item that bounds the scope (§n, D-n) or the technical constraint. A temporary circumstance ("not now") is a deferral, not a rejection: leave such a request open.

## When to check the record

During triage (step 1, Gather context), list the record with `gh issue list --label out-of-scope --state closed --json number,title,body,comments`, and match the new request against it by concept, not keyword: "save the current view as an image" matches "Screenshot feature". (`gh issue list --search "…"` narrows the list.) If there's a match, surface it to the owner: "This is similar to #42 (Screenshot feature), rejected because [reason]. Do you still feel the same way?"

The owner may:

- **Confirm**: close the new issue with `gh issue close <n> --duplicate-of <canonical>` (GitHub marks it a duplicate and links the two; no comment, so nothing hides the duplicate marker, as `SKILL.md` says), and add it to the canonical issue's Prior requests list by editing the decision comment ("Edit a comment" in `docs/agents/issue-tracker.md`).
- **Reconsider**: reopen the canonical issue, remove its `out-of-scope` label, and let the new request proceed through normal triage, or become a spec change.
- **Disagree**: the requests are related but distinct; proceed with normal triage.

## When to record a rejection

Only when an **enhancement** (not a bug) is rejected as `wontfix`. Don't label `out-of-scope` an issue closed as `wontfix` because it is **already implemented**: that is a built feature, not a rejected one, and it would poison the deduplication. Its closing comment points to where the feature lives instead.

The flow:

1. The owner decides a request is out of scope.
2. Check the record for a matching canonical issue.
3. If there is one: close the request as a duplicate of it and add it to its Prior requests, as under Confirm.
4. If there isn't: make this request the canonical issue. Post the decision comment, add the labels `wontfix` and `out-of-scope`, and close it as not planned.
