module ContinuousHMM

# Package entry point required by the `name`/`uuid` metadata in Project.toml,
# added so that `Pkg.test()` runs (2026-07 technical-review finding 7).
# The framework itself is script-loaded, not module-loaded: runners and the
# test suite bring the full type/algorithm surface into Main via
# `include("Include.jl")` at the repository root, which enforces the load
# order documented in CLAUDE.md. This module therefore intentionally defines
# and exports nothing: `using ContinuousHMM` succeeds but does not provide the
# framework API (build, MyContinuousHiddenMarkovModel, ...); use
# include("Include.jl") from the repository root instead, as the runners and
# the test suite do. See the Quick Start section of README.md.

end
