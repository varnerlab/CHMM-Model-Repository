# r_msgarch/

R-side scaffolding for the reference Markov-switching GARCH baseline. The
Julia code in `src/MSGARCHReference.jl` calls these helpers via
[RCall.jl](https://juliainterop.github.io/RCall.jl/) so the paper's MS-GARCH
row in Table 2 is produced by the canonical Ardia et al. (2019, JSS)
implementation rather than the in-house Nelder-Mead self-fit.

## Why this is here

Peer review (R2 W1) requires either re-running MS-GARCH with the `MSGARCH`
R package or weakening every "multi-state benefit" claim to "in our
re-implementation" wording. We chose the former. RCall lets us drive the R
package from Julia without porting the Bayesian sampler ourselves.

## What gets pinned

After running `setup.R` once, the following files exist and should be
committed:

- `.Rprofile`         renv activator (auto-loads on every R invocation)
- `renv/activate.R`   renv bootstrapper
- `renv.lock`         pinned package versions (R + MSGARCH + transitive deps)

The actual package binaries live under `renv/library/` (gitignored).
`renv.lock` is the reproducibility contract: `renv::restore()` reproduces
the exact library on any machine.

## One-time setup

Prerequisites: R >= 4.2 on PATH (`R --version` returns a version), and an
internet connection for the initial CRAN install.

```bash
cd CHMM-Model/r_msgarch
Rscript setup.R
```

This installs `renv`, then `MSGARCH` 2.51 from CRAN, then snapshots
`renv.lock`. Total install time: 3 to 5 minutes. After this, no further R
commands need to be issued by hand; `RCall.jl` calls `setup.R`-equivalent
activation automatically through the working-directory `.Rprofile` hook.

## Restoring on a fresh machine

If `renv.lock` is already committed (the expected state for a paper
reviewer cloning the repo):

```bash
cd CHMM-Model/r_msgarch
Rscript -e 'renv::restore(prompt = FALSE)'
```

This installs the pinned MSGARCH version and all transitive dependencies
into a project-local library. No global R packages are touched.

## What `fit_msgarch.R` exposes

Two functions, both stateless:

- `fit_msgarch_ref(returns, K, ...)` returns a fitted-model list including
  a serialised raw vector for round-tripping back into R at simulate time.
- `simulate_msgarch_ref(serialized, n_steps, n_paths, seed)` returns a
  `(n_steps x n_paths)` matrix of simulated returns.

Both require an explicit integer `seed` argument, surfaced from the Julia
caller, so the run is bit-reproducible on a given (R, MSGARCH) version.

## Versioning policy

The pinned versions in `setup.R` (`MSGARCH_VERSION = "2.51"`,
`RENV_VERSION = "1.0.7"`) are the versions the paper Table 2 numbers were
generated against. Bumping either requires re-running the headline
artefacts and updating the version footnote in the manuscript.
