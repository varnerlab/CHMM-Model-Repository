# ========================================================================== #
# r_msgarch/setup.R
#
# One-shot R-side environment bootstrap for the reference MS-GARCH baseline
# called from Julia via RCall.jl. Pins:
#
#   - renv             (project-local package library)
#   - MSGARCH 2.51     (Ardia et al. 2019, JSS; CRAN reference implementation)
#   - all transitive dependencies via renv::snapshot()
#
# Run once on a fresh checkout (R >= 4.2 required):
#
#   cd CHMM-Model/r_msgarch
#   Rscript setup.R
#
# This produces:
#   - .Rprofile          renv activator (auto-loads on every Rscript invocation)
#   - renv/              project-local library
#   - renv.lock          pinned package versions (commit this to git)
#
# Subsequent runs (e.g. from RCall.jl) will pick up the pinned library
# automatically via the .Rprofile hook so long as the working directory is
# r_msgarch/ at R startup.
# ========================================================================== #

# --- pinned versions ------------------------------------------------------- #
MSGARCH_VERSION <- "2.51"
RENV_VERSION    <- "1.0.7"

# --- helpers --------------------------------------------------------------- #
.cat_step <- function(msg) cat(sprintf("\n[setup] %s\n", msg))

# --- bootstrap renv -------------------------------------------------------- #
.cat_step("checking R version")
if (getRversion() < "4.2.0") {
  stop("R >= 4.2.0 required (found ", as.character(getRversion()), ")")
}

.cat_step(sprintf("installing renv %s if missing", RENV_VERSION))
if (!requireNamespace("renv", quietly = TRUE) ||
    packageVersion("renv") < RENV_VERSION) {
  install.packages("renv",
    repos = "https://cloud.r-project.org",
    quiet = TRUE
  )
}

.cat_step("initialising renv project (bare, no autodetect)")
renv::init(bare = TRUE, restart = FALSE)

.cat_step(sprintf("installing MSGARCH %s from CRAN", MSGARCH_VERSION))
renv::install(sprintf("MSGARCH@%s", MSGARCH_VERSION), prompt = FALSE)

.cat_step("verifying MSGARCH loads")
suppressPackageStartupMessages({
  library(MSGARCH)
})
cat(sprintf("  MSGARCH %s loaded OK\n", as.character(packageVersion("MSGARCH"))))

.cat_step("snapshotting renv.lock")
renv::snapshot(prompt = FALSE)

.cat_step("done")
cat(sprintf("R-side environment pinned. R %s, MSGARCH %s.\n",
            as.character(getRversion()),
            as.character(packageVersion("MSGARCH"))))
cat("Commit r_msgarch/.Rprofile, r_msgarch/renv.lock, and r_msgarch/renv/activate.R.\n")
