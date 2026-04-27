# Attic: v10 journal-revision documentation and exploratory artefacts

This directory holds top-level documentation, exploratory notebooks,
one-shot data-extension scripts, and reference PDFs that were produced
during the v10 (journal-revision) phase of the project but are not
needed for the arXiv preprint or its companion code.

| Item | Origin | Why archived |
|---|---|---|
| `LITERATURE-REVIEW.md` | v10 reference compilation (April 2026) | Historical bibliography; not loaded by any script |
| `user-comments.md` | v10 reviewer-style feedback log | All items addressed in current arXiv version |
| `Notebooks/01..05.ipynb` | v10 exploratory Jupyter notebooks | Educational, not part of the build pipeline |
| `planning/DECISION-MEMO.md`, `planning/plan-equity-paper.md` | v10 journal-submission planning | Superseded by arXiv-only scope |
| `downloaded-references/*.pdf` | Reference PDFs from journal targeting | Already gitignored; physical copies preserved here for personal reference only |
| `fetch_oos_extended.jl` | One-shot Alpaca API extension script | Output already bundled into `data/CHMM-SP500-OoS-Remainder.jld2`; rerun only if data refresh is planned |
