# ========================================================================== #
# MSGARCHReference.jl
#
# Julia bridge to the reference MS-GARCH implementation (Ardia et al. 2019,
# JSS) shipped as the CRAN MSGARCH R package. Wraps RCall.jl with a
# stateless, well-typed interface so callers do not have to touch R.
#
# Peer-review motivation: R2 W1 demands either the reference R-package fit
# in Table 2 or that every "multi-state benefit" claim be qualified to
# "in our re-implementation". This module produces the former. The
# in-house Nelder-Mead frequentist fit in src/MSGARCH.jl is kept as a
# parity / cross-check baseline; it is not replaced.
#
# Setup: run r_msgarch/setup.R once (R >= 4.2 required). The R-side
# environment is pinned via renv (see r_msgarch/README.md). This module
# re-uses the project-local library on every call.
#
# Public API:
#   MyMSGARCHReferenceModel              fitted-model struct
#   fit_msgarch_reference(obs, K; ...)   fit on a return series
#   simulate_msgarch_reference(m, n; .)  draw simulated paths
#   msgarch_reference_versions()         report R + MSGARCH versions
#
# All entry points require an integer `seed` keyword for reproducibility.
# ========================================================================== #

using RCall;

const _MSGARCH_R_DIR = joinpath(_ROOT, "r_msgarch");
const _MSGARCH_R_SCRIPT = joinpath(_MSGARCH_R_DIR, "fit_msgarch.R");

# State flag so the R session is initialised exactly once per Julia process.
const _MSGARCH_R_INIT = Ref(false);

"""
    mutable struct MyMSGARCHReferenceModel

Fitted MS-GARCH model produced by the CRAN `MSGARCH` package via RCall.

The numerical parameter summary (`par_post_mean`, `par_post_sd`, `transition`,
`loglik`) is for inspection and tabulation only. To draw simulated paths,
hand the model back to `simulate_msgarch_reference`; it round-trips the
serialised R object stored in `r_serialized` rather than re-fitting.

### Fields
- `K::Int`                      regime count.
- `variance_spec::Vector{String}`  per-regime variance model (e.g. "sGARCH").
- `distribution::Vector{String}`   per-regime innovation distribution.
- `fit_method::String`          "MCMC" (Bayesian, default) or "ML".
- `loglik::Float64`             posterior-mean log-likelihood (or MLE LL).
- `par_names::Vector{String}`   MSGARCH parameter labels.
- `par_post_mean::Vector{Float64}`   posterior means (or MLE point).
- `par_post_sd::Vector{Float64}`     posterior SDs (NaN entries for ML).
- `transition::Matrix{Float64}`      K x K transition matrix at par_post_mean.
- `fit_robj::RObject`                live R-side MSGARCH fit object. Held
                                     by RCall for the lifetime of the
                                     Julia session; cannot be JLD2-saved
                                     because MSGARCH stores Rcpp C++
                                     pointers that do not survive
                                     serialize/unserialize roundtrips.
"""
mutable struct MyMSGARCHReferenceModel
    K::Int;
    variance_spec::Vector{String};
    distribution::Vector{String};
    fit_method::String;
    loglik::Float64;
    par_names::Vector{String};
    par_post_mean::Vector{Float64};
    par_post_sd::Vector{Float64};
    transition::Matrix{Float64};
    fit_robj::RObject;

    MyMSGARCHReferenceModel() = new();
end

# --------------------------------------------------------------------------- #
# R session bootstrap
# --------------------------------------------------------------------------- #

"""
    _ensure_r_session()

Initialise the R session on first call: cd into r_msgarch/ so the renv
.Rprofile auto-activates, then source fit_msgarch.R. Errors out with a
descriptive message if MSGARCH is not available (telling the user to run
setup.R).
"""
function _ensure_r_session()
    if _MSGARCH_R_INIT[]; return; end

    if !isfile(_MSGARCH_R_SCRIPT)
        error("MSGARCHReference: missing $_MSGARCH_R_SCRIPT. " *
              "Was r_msgarch/ deleted?");
    end

    # The R session is already up by the time we get here, so the renv
    # .Rprofile that fires at R startup hasn't activated. Hop into the
    # r_msgarch/ working directory and source renv/activate.R explicitly.
    activate_path = joinpath(_MSGARCH_R_DIR, "renv", "activate.R");
    if !isfile(activate_path)
        error(
            "MSGARCHReference: renv has not been initialised at " *
            "$activate_path. Run:\n" *
            "    cd r_msgarch && Rscript setup.R\n" *
            "to install MSGARCH and snapshot renv.lock."
        );
    end
    R"""
    setwd($_MSGARCH_R_DIR)
    source($activate_path)
    """;

    # Sanity-check MSGARCH availability before sourcing helpers.
    has_pkg = rcopy(R"requireNamespace('MSGARCH', quietly = TRUE)");
    if !has_pkg
        error(
            "MSGARCHReference: the MSGARCH R package is not available in " *
            "the project-local renv library. Run:\n" *
            "    cd r_msgarch && Rscript setup.R\n" *
            "(or, if renv.lock is already committed, " *
            "Rscript -e 'renv::restore(prompt = FALSE)').\n" *
            "See r_msgarch/README.md for details."
        );
    end

    R"source($_MSGARCH_R_SCRIPT)";

    _MSGARCH_R_INIT[] = true;
    return;
end

# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #

"""
    fit_msgarch_reference(obs::AbstractVector{<:Real}, K::Int;
                          variance_spec="sGARCH",
                          distribution="norm",
                          fit_method="MCMC",
                          n_mcmc=12_500, n_burnin=2_500, n_thin=10,
                          seed::Int) -> MyMSGARCHReferenceModel

Fit a K-regime MS-GARCH model to `obs` using the CRAN `MSGARCH` package.

`fit_method` is `"MCMC"` (Bayesian, the Ardia 2019 reference) by default; pass
`"ML"` for the frequentist MLE cross-check. Reproducibility requires an
explicit integer `seed` (no default).

Throws if R or MSGARCH are missing — see `r_msgarch/README.md` for setup.
"""
function fit_msgarch_reference(obs::AbstractVector{<:Real}, K::Int;
        variance_spec::Union{String,Vector{String}}="sGARCH",
        distribution::Union{String,Vector{String}}="norm",
        fit_method::String="MCMC",
        n_mcmc::Int=12_500,
        n_burnin::Int=2_500,
        n_thin::Int=10,
        seed::Int)::MyMSGARCHReferenceModel

    @assert K >= 1 "K must be a positive integer";
    @assert fit_method in ("MCMC", "ML") "fit_method must be MCMC or ML";

    _ensure_r_session();

    obs_f = convert(Vector{Float64}, collect(obs));
    vs_r = isa(variance_spec, String) ? [variance_spec] : variance_spec;
    di_r = isa(distribution,  String) ? [distribution]  : distribution;

    # The R-side fit object holds Rcpp C++ pointers that don't survive
    # serialize/unserialize, so we stash the result in R's global env and
    # pull each summary field individually (rcopy on a list containing the
    # fit object would also try to copy the fit, which fails the same way).
    R"""
    .last_fit_result <- fit_msgarch_ref(
      returns       = $obs_f,
      K             = $K,
      variance_spec = $vs_r,
      distribution  = $di_r,
      fit_method    = $fit_method,
      n_mcmc        = $n_mcmc,
      n_burnin      = $n_burnin,
      n_thin        = $n_thin,
      seed          = $seed
    )
    """;

    # NA on the R side becomes `missing` on the Julia side; coerce to NaN
    # so every float field is a uniform `Float64`.
    _to_f64(x) = ismissing(x) ? NaN : Float64(x);
    _vec_f64(v) = [_to_f64(x) for x in collect(v)];

    m = MyMSGARCHReferenceModel();
    m.K              = Int(rcopy(R".last_fit_result$K"));
    m.variance_spec  = String.(collect(rcopy(R".last_fit_result$variance_spec")));
    m.distribution   = String.(collect(rcopy(R".last_fit_result$distribution")));
    m.fit_method     = String(rcopy(R".last_fit_result$fit_method"));
    m.loglik         = _to_f64(rcopy(R".last_fit_result$loglik"));
    m.par_names      = String.(collect(rcopy(R".last_fit_result$par_names")));
    m.par_post_mean  = _vec_f64(rcopy(R".last_fit_result$par_post_mean"));
    m.par_post_sd    = _vec_f64(rcopy(R".last_fit_result$par_post_sd"));

    # The transition matrix may come back as Matrix{Union{Missing,Float64}}
    # if MSGARCH's TransMat path failed and the R helper returned NA-filled
    # matrix. Coerce defensively.
    trans_raw = rcopy(R".last_fit_result$transition");
    if isa(trans_raw, Matrix)
        m.transition = Float64.(_to_f64.(trans_raw));
    else
        m.transition = reshape([_to_f64(x) for x in collect(trans_raw)], m.K, m.K);
    end

    # Hold a live reference to the R-side fit object on the Julia side.
    # The RObject keeps the fit alive across GC; passing m.fit_robj back
    # to R via interpolation reattaches it without serialization.
    m.fit_robj = R".last_fit_result$fit";
    return m;
end

# --------------------------------------------------------------------------- #
# Simulate
# --------------------------------------------------------------------------- #

"""
    simulate_msgarch_reference(m::MyMSGARCHReferenceModel, n_steps::Int;
                               n_paths::Int=1, seed::Int) -> Matrix{Float64}

Draw `n_paths` simulated return paths of length `n_steps` from the
posterior-predictive (MCMC) or MLE-plug-in (ML) distribution of the fitted
model. Returns a `(n_steps, n_paths)` matrix.

Reproducibility requires an explicit integer `seed`.
"""
function simulate_msgarch_reference(m::MyMSGARCHReferenceModel, n_steps::Int;
        n_paths::Int=1, seed::Int)::Matrix{Float64}

    @assert n_steps >= 1;
    @assert n_paths >= 1;

    _ensure_r_session();

    fit = m.fit_robj;
    res = R"""
    simulate_msgarch_ref(
      fit     = $fit,
      n_steps = $n_steps,
      n_paths = $n_paths,
      seed    = $seed
    )
    """;

    M = rcopy(res);
    M = isa(M, Matrix{Float64}) ? M : convert(Matrix{Float64}, M);
    @assert size(M) == (n_steps, n_paths) "MSGARCH simulate returned shape $(size(M)) instead of ($n_steps, $n_paths)";
    return M;
end

# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #

"""
    msgarch_reference_versions() -> NamedTuple

Returns a NamedTuple with R version, MSGARCH version, platform, OS, and
timestamp. Embed in the paper artefact alongside the fitted numbers so
reviewers can verify the version used.
"""
function msgarch_reference_versions()
    _ensure_r_session();
    rl = rcopy(R"msgarch_version_info()");
    return (
        r_version       = String(rl[:r_version]),
        msgarch_version = String(rl[:msgarch_version]),
        platform        = String(rl[:platform]),
        os              = String(rl[:os]),
        timestamp_utc   = String(rl[:timestamp_utc]),
    );
end
