# ========================================================================== #
# r_msgarch/fit_msgarch.R
#
# R-side helpers wrapping MSGARCH (Ardia et al. 2019, JSS) for invocation
# from Julia via RCall.jl. Two entry points:
#
#   fit_msgarch_ref(returns, K, ...)   -> list of fitted-model fields
#   simulate_msgarch_ref(model, ...)   -> matrix of simulated paths
#
# Design: stateless functions, no globals. The fitted-model object is
# round-tripped through R's serialize() into a raw byte vector that the
# Julia caller stores in memory and hands back at simulation time. This
# avoids re-fitting at simulate time and keeps the bridge layer simple.
#
# The script assumes renv has activated the pinned MSGARCH version (see
# setup.R). source() this file once at session start from RCall:
#
#     R"source('r_msgarch/fit_msgarch.R')"
# ========================================================================== #

suppressPackageStartupMessages({
  library(MSGARCH)
})

# --------------------------------------------------------------------------- #
# fit_msgarch_ref
#
# Args:
#   returns       numeric vector of returns (Julia passes the IS series).
#   K             integer >= 1, regime count.
#   variance_spec character: "sGARCH" (default), "eGARCH", "gjrGARCH", "tGARCH".
#                 Replicated K times to give per-regime spec.
#   distribution  character: "norm" (default), "std" (Student-t), "ged".
#                 Replicated K times.
#   fit_method    "MCMC" (default, Bayesian, the Ardia 2019 reference) or
#                 "ML" (frequentist MLE for sanity-check parity with the
#                 in-house Nelder-Mead fit).
#   n_mcmc        total MCMC draws per chain (default 12500).
#   n_burnin      burn-in per chain to discard (default 2500).
#   n_thin        thinning interval (default 10 -> 1000 retained per chain).
#   seed          integer, R-side RNG seed. Must be set explicitly to make
#                 the fit reproducible across machines.
#
# Returns a list (decoded by RCall as a NamedTuple-ish object on the Julia
# side) with the fields:
#
#   K               integer.
#   variance_spec   character vector length K.
#   distribution    character vector length K.
#   fit_method      "MCMC" or "ML".
#   loglik          numeric scalar at the posterior mean / MLE.
#   par_names       character vector of parameter names from MSGARCH.
#   par_post_mean   numeric vector of posterior means / MLE estimates.
#   par_post_sd     numeric vector of posterior SDs (NA for ML).
#   transition      K x K transition matrix at the posterior mean (or MLE).
#   serialized      raw byte vector containing serialize(fit) for round-trip
#                   to simulate_msgarch_ref().
#
# Errors out with a descriptive message on any MSGARCH failure; the Julia
# bridge surfaces the message verbatim.
# --------------------------------------------------------------------------- #
fit_msgarch_ref <- function(returns,
                            K = 2L,
                            variance_spec = "sGARCH",
                            distribution = "norm",
                            fit_method = c("MCMC", "ML"),
                            n_mcmc = 12500L,
                            n_burnin = 2500L,
                            n_thin = 10L,
                            seed = NULL) {

  fit_method <- match.arg(fit_method)
  K <- as.integer(K)
  if (K < 1L) stop("K must be a positive integer (got ", K, ")")

  # MSGARCH::CreateSpec requires EITHER a length-1 model + explicit K in
  # switch.spec (homogeneous regimes), OR a length-K model + no K
  # (heterogeneous regimes). Detect which case the caller asked for.
  vs_len <- length(variance_spec)
  di_len <- length(distribution)
  if (!(vs_len %in% c(1L, K))) stop("variance_spec length must be 1 or K")
  if (!(di_len %in% c(1L, K))) stop("distribution length must be 1 or K")
  homogeneous <- (vs_len == 1L && di_len == 1L)
  if (is.null(seed)) stop("seed must be supplied (integer) for reproducibility")

  set.seed(as.integer(seed))

  spec <- if (homogeneous) {
    CreateSpec(
      variance.spec     = list(model = variance_spec),
      distribution.spec = list(distribution = distribution),
      switch.spec       = list(do.mix = FALSE, K = K)
    )
  } else {
    if (vs_len == 1L) variance_spec <- rep(variance_spec, K)
    if (di_len == 1L) distribution  <- rep(distribution,  K)
    CreateSpec(
      variance.spec     = list(model = variance_spec),
      distribution.spec = list(distribution = distribution),
      switch.spec       = list(do.mix = FALSE)
    )
  }
  # Reflect what we actually fitted in the returned summary
  if (length(variance_spec) == 1L) variance_spec <- rep(variance_spec, K)
  if (length(distribution)  == 1L) distribution  <- rep(distribution,  K)

  fit <- if (fit_method == "MCMC") {
    ctr <- list(nmcmc = as.integer(n_mcmc),
                nburn = as.integer(n_burnin),
                nthin = as.integer(n_thin))
    FitMCMC(spec = spec, data = as.numeric(returns), ctr = ctr)
  } else {
    FitML(spec = spec, data = as.numeric(returns))
  }

  # --- summarise -------------------------------------------------------- #
  if (fit_method == "MCMC") {
    draws         <- as.matrix(fit$par)
    par_post_mean <- colMeans(draws)
    par_post_sd   <- apply(draws, 2, stats::sd)
    par_names     <- colnames(draws)
  } else {
    par_post_mean <- fit$par
    par_post_sd   <- rep(NA_real_, length(par_post_mean))
    par_names     <- names(par_post_mean)
  }

  # Log-likelihood:
  #   - ML fit:    fit$loglik is the MLE log-likelihood (scalar).
  #   - MCMC fit:  no $loglik field; compute the posterior-expected
  #                in-sample LL via PredPdf(fit, x=returns, log=TRUE,
  #                do.its=TRUE), which returns a (T, n_post) matrix of
  #                per-observation log-densities at each posterior draw.
  loglik <- tryCatch(
    {
      if (fit_method == "ML" && !is.null(fit$loglik)) {
        as.numeric(fit$loglik)
      } else {
        pp <- MSGARCH::PredPdf(object = fit,
                               x      = as.numeric(returns),
                               log    = TRUE,
                               do.its = TRUE)
        # pp is (T, n_post) for MCMC, (T,) for ML
        if (is.null(dim(pp))) {
          as.numeric(sum(pp))
        } else {
          as.numeric(mean(colSums(pp)))
        }
      }
    },
    error = function(e) NA_real_
  )

  # Recover the transition matrix at the posterior mean (or MLE).
  # TransMat() in MSGARCH 2.51 only dispatches on MSGARCH_SPEC, not on
  # the fit object directly, so we always go through the spec.
  trans_mean <- tryCatch(
    {
      if (K == 1L) {
        matrix(1.0, nrow = 1L, ncol = 1L)
      } else {
        P <- MSGARCH::TransMat(object = spec, par = par_post_mean)
        matrix(as.numeric(P), nrow = K, ncol = K)
      }
    },
    error = function(e) matrix(NA_real_, nrow = K, ncol = K)
  )

  # Note: do NOT serialize() the fit. MSGARCH stores Rcpp C++ pointers
  # which become NULL after a serialize/unserialize roundtrip, breaking
  # any subsequent simulate() call. Return the fit object directly; the
  # Julia caller holds it as an RCall.RObject for the lifetime of the
  # session.
  list(
    K              = K,
    variance_spec  = variance_spec,
    distribution   = distribution,
    fit_method     = fit_method,
    loglik         = as.numeric(loglik),
    par_names      = as.character(par_names),
    par_post_mean  = as.numeric(par_post_mean),
    par_post_sd    = as.numeric(par_post_sd),
    transition     = trans_mean,
    fit            = fit
  )
}

# --------------------------------------------------------------------------- #
# simulate_msgarch_ref
#
# Args:
#   fit          MSGARCH fit object as returned in fit_msgarch_ref()$fit.
#                Must be the live R-side object (held by Julia as RObject);
#                a serialized/unserialized copy will not work because
#                MSGARCH's Rcpp C++ pointers do not survive the roundtrip.
#   n_steps      integer, horizon per path.
#   n_paths      integer, number of Monte Carlo paths.
#   seed         integer, R-side RNG seed.
#
# Returns:
#   numeric matrix of dimension (n_steps x n_paths). Column j is path j.
#
# For an MCMC fit, MSGARCH::simulate() integrates over the posterior:
# each simulated path uses parameters drawn (with replacement) from the
# posterior sample. For an ML fit, paths are drawn at the MLE point estimate.
# --------------------------------------------------------------------------- #
simulate_msgarch_ref <- function(fit,
                                 n_steps,
                                 n_paths,
                                 seed = NULL) {

  if (is.null(seed)) stop("seed must be supplied (integer) for reproducibility")
  set.seed(as.integer(seed))

  n_steps <- as.integer(n_steps)
  n_paths <- as.integer(n_paths)

  is_mcmc <- inherits(fit, "MSGARCH_MCMC_FIT")

  if (is_mcmc) {
    # MCMC fit. simulate(nsim = 1) returns one path per retained posterior
    # draw, in a (n_steps, n_post) matrix. Sample n_paths columns from it
    # (with replacement only if we need more paths than posterior draws);
    # this is the canonical posterior-predictive integration.
    sim <- simulate(object = fit, nahead = n_steps, nsim = 1L)
    full <- as.matrix(sim$draw)
    n_post <- ncol(full)
    replace <- (n_paths > n_post)
    idx <- sample.int(n_post, n_paths, replace = replace)
    draws <- full[, idx, drop = FALSE]
  } else {
    # ML fit. simulate(nsim = n_paths) returns (n_steps, n_paths) directly.
    sim <- simulate(object = fit, nahead = n_steps, nsim = n_paths)
    draws <- as.matrix(sim$draw)
  }

  # Some versions return (n_paths x n_steps) by mistake; transpose if so.
  if (nrow(draws) != n_steps || ncol(draws) != n_paths) {
    if (nrow(draws) == n_paths && ncol(draws) == n_steps) {
      draws <- t(draws)
    } else {
      stop(sprintf(
        "simulate() returned matrix of shape %d x %d; expected %d x %d",
        nrow(draws), ncol(draws), n_steps, n_paths
      ))
    }
  }
  draws
}

# --------------------------------------------------------------------------- #
# msgarch_version_info
#
# Returns a list documenting the R + MSGARCH versions in use, for embedding
# in the paper artefact alongside the fitted numbers.
# --------------------------------------------------------------------------- #
msgarch_version_info <- function() {
  list(
    r_version       = as.character(getRversion()),
    msgarch_version = as.character(packageVersion("MSGARCH")),
    platform        = R.version$platform,
    os              = R.version$os,
    timestamp_utc   = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
}
