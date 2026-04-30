# MR comment detail files

One file per reviewer comment from the GitLab MR. Created when the user pastes a comment in chat; indexed by `../mr_tracker.md`.

Filename format: `MR-NNN_<author-lastname>_<short-slug>.md` — e.g. `MR-001_smith_depth-loss-eps.md`. NNN is monotonic and zero-padded; assigned by reading the highest existing ID in the tracker and incrementing.

See `../mr_tracker.md` for the file template and status vocabulary.
