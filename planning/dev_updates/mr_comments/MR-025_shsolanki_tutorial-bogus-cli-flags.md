---
id: MR-025
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: docs/source/examples/dynamic_surgical.rst:54-56 (also line 30)
status: open
labels: [docs, tutorial, accuracy]
---

# MR-025 — Tutorial quickstart shows CLI flags that don't exist on `Config`

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> The Quickstart at `dynamic_surgical.rst:54-56` shows `--dataset_type endonerf --strategy dynamic`, but neither flag exists on `Config` (`dynamic_surgical_trainer.py:76-131`) — tyro will reject the command.
>
> Also cites SCARED at line 30, but no SCARED parser exists.
>
> Trim to the real flags (`--data_dir`, `--depth_mode`), and either drop the SCARED mention or mark it "not yet implemented".

## Summary
Tutorial documents CLI flags that don't exist (`--dataset_type`, `--strategy`) — running the quickstart as-written will fail with a tyro error. Need to align the doc with the real `Config` dataclass and either delete or qualify the SCARED reference.

## Scope / affected files
- `docs/source/examples/dynamic_surgical.rst` — quickstart at lines 54-56; SCARED mention at line 30.

## Action plan
- [ ] Audit `Config` (`examples/dynamic_surgical_trainer.py:76-131`) and list the actual tyro-recognised flags.
- [ ] Rewrite the quickstart command in `dynamic_surgical.rst:54-56` to use real flags (`--data_dir`, `--depth_mode`, `--coarse_steps`, `--fine_steps`, `--render_gif_after_train`, etc.).
- [ ] Either delete the SCARED line at line 30 or annotate it "not yet implemented (SCARED loader is a `NotImplementedError` stub; see follow-up MR)".
- [ ] Rebuild the RST locally and skim the rendered HTML to confirm.

## Resolution
<filled after fix lands>

## User-side reply draft
> Rewrote the quickstart to use the real flags from `Config` (`--data_dir`, `--depth_mode`, `--coarse_steps`, `--fine_steps`, `--render_gif_after_train`) and added a "(not yet implemented)" tag next to the SCARED mention. Verified the rendered HTML.
