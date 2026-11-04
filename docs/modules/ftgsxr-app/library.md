# The library in `ftgsxr-app`

How `ftgsxr_app::library` meets §1.5's library and §2.1's `Library`, `LibraryEntry`, `EntryState`, `LibraryCommand`, `LoadedModel` and `LibraryFailure`: the model sources and their listings, header-only reads, loads and unloads, the pickers and dropped files (§1.7). The spec items are §1.5, §1.7 (Library tab, file drops), §1.2, §2.1 `ftgsxr_app::library`, §2.4 Threads, D-31, D-40 and D-49; this doc explains them and never overrides them. Built in M-4 (#278, #279).

## Structure

A private `LibraryPlugin`, added by `AppPlugin` first among the feature plugins, holds the sources, the pending tasks and the commands' system set `LibraryCommands`. Its systems run in `Update`: the commands, then the polling of the I/O tasks, the pickers and the drops; `PostUpdate` follows which entries are in use. `persist`'s chain runs before `LibraryCommands`, so a scene's unloads and loads land in the same frame's commands.

- **Threads.** Listings, header reads and HTTP fetches run on the I/O task pool; the CPU part of a load is `SplatLoad` on the compute pool (§2.4 Threads, Q-99). Nothing blocks the frame.
- **Entries.** One `LibraryEntry` per model a source offers, keyed by the source and the model's reference; its `EntryState` carries every state and failure text the Library tab shows. No toast is raised here: a failure with no entry to carry it (a `LoadRef`, a picked file, a drop, a failed listing) is a `LibraryFailure { name, reference, code, text }` message that `ui` turns into a toast (Q-173, Q-193).
- **Directories.** A listed subdirectory stays an entry; loading it lists it and puts its entries right after it; a source's `Refresh` relists the root only (Q-192). On a first run the default models directory may not exist: it lists as empty, with no toast, and is not created (Q-216, Q-14).
- **Refresh.** Relisting keeps an entry's state, and its information, identity and thumbnail while its size, mtime (nginx listings only) and SHA-256 are unchanged (§1.5, Q-231). An HTTP entry's header read is preceded by a `HEAD` whose ETag the library keeps privately, since `Entry` and `LibraryEntry` carry none; a refresh checks it with `HttpClient::revalidate` in the background: a 304 keeps the information, another ETag drops it and reads the header again, and a failed check changes nothing (Q-230, Q-232). A load of an HTTP entry without information sends that `HEAD` before its download, of the entry's location or, for a checkpoint, of its `manifest.json`, and its ETag is kept with the information the load fills; a refresh checks the same URL, and a `ply` or checkpoint entry whose ETag changed gets its information back from its next load (Q-242).
- **HTTP.** A model is fetched into the cache, whose size limit `AppPlugin` passes from the settings' `[cache]`; an index entry's soundtrack is downloaded with the model and handed on in `LoadedModel.audio` (#279), and its `scene` manifest is downloaded the same way and put on the instance for `ui`'s training cameras, in a crate-private component (#374). Both stay in the cache's in-use set while the instance is loaded; a failed download is logged and the model loads without them.

## A load's life

1. `Load` or `LoadRef` spawns the instance with `SplatLoad` and the entry goes to its loading state.
2. When `SplatModel` appears (the CPU load done), `LoadedModel` is inserted with the model's information and `frame_step_s` (Q-174, from `SplatAsset`), and the instance is `Hidden` (Q-148). Instances are parented under `ContentGroup` (`rig`).
3. `SplatEvent::Uploaded` shows the instance, unless the crate-private `ShowOnUpload` that `persist` set false for a scene's hidden instance says otherwise (Q-202), and the entry becomes `Loaded(entity)`. `SplatEvent::Failed` despawns it and the entry becomes `Failed(text)`, or a `LibraryFailure` is sent.

A model over the memory budget is refused with E-MEM-BUDGET and the reduce-to-budget count N in its text (Q-117). N is the budget's remaining bytes, from the public `SplatGpu`'s device (Q-172), divided by the static bytes per primitive under the tier's layout and the governor's SH cap, floored (Q-158).

## Unloads

`Unload` and `UnloadAll` despawn the instances; the model's GPU memory is released by the render plugin, which I-1 checks through `SplatGpu`. An unload asked for from the keyboard goes through `ui`'s confirmation first (Q-199, `ui.md`).

**Replacing a missing model** (§1.5 Library, Q-255). `ReplaceEntry { identity, file }` is a confirmed "Replace with…": `file` loads with the missing entry's identity attached, and the model read is kept whatever its identity. A file of another identity clears only the entry of `identity` (another missing entry of the file's identity stays, Q-258) and its instance carries `Replaces(identity)`, through which `persist` gives it that scene model's attributes; a file of the entry's own identity is the model, as a "Locate…" pick of it. A message naming no missing entry does nothing; a load that fails leaves the entry missing with the scene's attributes kept (Q-259). Any kind §1.2 reads may replace a model (Q-257).

**A live tier switch** (Q-238). When `SplatTierState.tier` changes, every instance with a `SplatModel` is reloaded on its own entity: its `SplatModel` goes, its entry turns `Loading`, its source is re-read with `open_ref` and a `SplatLoad` with its first load's kind goes on the same entity, so the pose, time, mode, visibility, color transform, selection and soundtrack stay; its `LoadedModel` is kept. The upload shows it only if it was shown. A re-read, load or upload that fails, an E-MEM-BUDGET under the new tier's budget say, unloads that instance with §1.15's text on its entry or as a `LibraryFailure`; the others stay. An instance whose load asked to reduce to budget asks again (owner, #391; #395): once its re-read completes and the render plugin's device has been made for the new tier, its row count is Q-158's under the new tier, against the models then on the device, all its rows when they fit; `LoadedModel`'s `info` and `reduced` follow it. One loaded without the option is reloaded without it.

## Q-n rows relied on

Q-148, Q-158, Q-166 (messages), Q-172, Q-173, Q-174, Q-192, Q-193, Q-202, Q-203 (missing entries from a scene, `persist.md`), Q-216, Q-230, Q-231, Q-232, Q-238, Q-242, Q-245 (a scene model's file of another identity, `persist.md`), Q-255, Q-257, Q-258, Q-259 (`ReplaceEntry`, below); with Q-99, Q-117 and Q-14 from earlier milestones.
