# ========================================================================================= #
# SkewEmissions.jl
#
# Fernandez-Steel skew densities for the CHMM emission-family ablation requested by the
# JoFE revision (referee M9).
#
# Fernandez-Steel construction: given a symmetric base density f_0 (standardised Student-t
# or Laplace), the skewed density at skew parameter γ > 0 is
#
#     f(x; μ, σ, γ) = (2 / (σ (γ + 1/γ))) * f_0( (x - μ)/σ * γ^{-sign(x - μ)} ),
#
# which splits the real line into a left half with effective scale σ/γ and a right half
# with effective scale σγ. γ = 1 recovers the symmetric base; γ > 1 produces right-skew,
# γ < 1 left-skew. Sampling: draw Z from f_0 and I ~ Bernoulli(γ²/(γ² + 1)); if I = 1 set
# X = μ + |Z| σ γ, else X = μ - |Z| σ / γ.
#
# Exports logpdf_skewt, logpdf_skewl, sample_skewt, sample_skewl, fit_gamma_mle_skewt,
# fit_gamma_mle_skewl.
# ========================================================================================= #

using Statistics;
using Distributions;
using SpecialFunctions;

# --------------------------------------------------------------------------------------- #
# Log-densities (Fernandez-Steel, location-scale form)
# --------------------------------------------------------------------------------------- #

"""
    logpdf_skewt(x, μ, σ, ν, γ) -> Float64

Log-density of a Fernandez-Steel skew-Student-t with location μ, scale σ, degrees of
freedom ν, skew γ. γ = 1 recovers the symmetric t.
"""
function logpdf_skewt(x::Float64, μ::Float64, σ::Float64, ν::Float64, γ::Float64)::Float64
    z = (x - μ) / σ;
    # Effective scaled z depending on sign
    z_eff = z >= 0 ? z / γ : z * γ;
    # log density of standard Student-t at z_eff
    log_t = loggamma((ν + 1) / 2) - loggamma(ν / 2) - 0.5 * log(π * ν) -
            ((ν + 1) / 2) * log1p(z_eff^2 / ν);
    return log(2.0) - log(σ) - log(γ + 1.0 / γ) + log_t;
end

"""
    logpdf_skewl(x, μ, b, γ) -> Float64

Log-density of a Fernandez-Steel skew-Laplace with location μ, scale b, skew γ.
γ = 1 recovers the symmetric Laplace.
"""
function logpdf_skewl(x::Float64, μ::Float64, b::Float64, γ::Float64)::Float64
    z = (x - μ) / b;
    z_eff = z >= 0 ? z / γ : z * γ;
    log_l = -log(2.0) - abs(z_eff);
    return log(2.0) - log(b) - log(γ + 1.0 / γ) + log_l;
end

# --------------------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------------------- #

function sample_skewt(μ::Float64, σ::Float64, ν::Float64, γ::Float64)::Float64
    Z = rand(TDist(ν));
    # Direction: right-half with probability γ²/(γ² + 1)
    if rand() < γ^2 / (γ^2 + 1)
        return μ + abs(Z) * σ * γ;
    else
        return μ - abs(Z) * σ / γ;
    end
end

function sample_skewl(μ::Float64, b::Float64, γ::Float64)::Float64
    Z = rand(Laplace(0.0, 1.0));
    if rand() < γ^2 / (γ^2 + 1)
        return μ + abs(Z) * b * γ;
    else
        return μ - abs(Z) * b / γ;
    end
end

# --------------------------------------------------------------------------------------- #
# Plug-in γ fit by weighted MLE (golden-section on log γ ∈ log[0.3, 3.0])
# μ, σ, ν are held fixed (from the symmetric CHMM-t/CHMM-L fit); γ is optimised on the
# responsibility-weighted log-likelihood. Each weight γ_t(k) is the posterior probability
# of observation t being in state k from the symmetric EM.
# --------------------------------------------------------------------------------------- #

"""
    fit_gamma_mle_skewt(obs, weights, μ, σ, ν; γ_bounds=(0.3, 3.0), iters=60) -> Float64

Weighted maximum-likelihood estimator for the skew parameter γ of a Fernandez-Steel
skew-t with the location/scale/df held at (μ, σ, ν). Optimisation by golden-section over
log γ to keep the parameterisation symmetric around γ = 1.
"""
function fit_gamma_mle_skewt(obs::AbstractVector{Float64}, weights::AbstractVector{Float64},
                             μ::Float64, σ::Float64, ν::Float64;
                             γ_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                             iters::Int=60)::Float64
    lo_log = log(γ_bounds[1]); hi_log = log(γ_bounds[2]);
    _loglik(log_γ) = begin
        γ = exp(log_γ);
        s = 0.0;
        @inbounds for t in 1:length(obs)
            s += weights[t] * logpdf_skewt(obs[t], μ, σ, ν, γ);
        end
        s;
    end
    φ = (sqrt(5.0) - 1.0) / 2.0;
    a = lo_log; b = hi_log;
    c = b - φ*(b - a); d = a + φ*(b - a);
    fc = _loglik(c); fd = _loglik(d);
    for _ in 1:iters
        if fc > fd
            b = d; d = c; fd = fc;
            c = b - φ*(b - a); fc = _loglik(c);
        else
            a = c; c = d; fc = fd;
            d = a + φ*(b - a); fd = _loglik(d);
        end
    end
    return exp(0.5 * (a + b));
end

"""
    fit_gamma_mle_skewl(obs, weights, μ, b; γ_bounds=(0.3, 3.0), iters=60) -> Float64

Weighted MLE for the skew parameter γ of a Fernandez-Steel skew-Laplace (μ, scale b
held fixed).
"""
function fit_gamma_mle_skewl(obs::AbstractVector{Float64}, weights::AbstractVector{Float64},
                             μ::Float64, b::Float64;
                             γ_bounds::Tuple{Float64,Float64}=(0.3, 3.0),
                             iters::Int=60)::Float64
    lo_log = log(γ_bounds[1]); hi_log = log(γ_bounds[2]);
    _loglik(log_γ) = begin
        γ = exp(log_γ);
        s = 0.0;
        @inbounds for t in 1:length(obs)
            s += weights[t] * logpdf_skewl(obs[t], μ, b, γ);
        end
        s;
    end
    φ = (sqrt(5.0) - 1.0) / 2.0;
    a_val = lo_log; b_val = hi_log;
    c = b_val - φ*(b_val - a_val); d = a_val + φ*(b_val - a_val);
    fc = _loglik(c); fd = _loglik(d);
    for _ in 1:iters
        if fc > fd
            b_val = d; d = c; fd = fc;
            c = b_val - φ*(b_val - a_val); fc = _loglik(c);
        else
            a_val = c; c = d; fc = fd;
            d = a_val + φ*(b_val - a_val); fd = _loglik(d);
        end
    end
    return exp(0.5 * (a_val + b_val));
end
