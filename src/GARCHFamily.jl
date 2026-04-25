# ========================================================================================= #
# GARCHFamily.jl
#
# Additional GARCH-family baselines requested by the JoFE revision (referee M7):
#   - EGARCH(1,1) with Gaussian innovations (asymmetric, leverage effect)
#   - GJR-GARCH(1,1) with Gaussian innovations (asymmetric leverage via sign dummy)
#   - GARCH(1,1) with Student-t innovations (heavy-tailed competitor to CHMM-t)
#   - HAR-RV on daily squared returns (heterogeneous-autoregressive on realised variance)
#
# Each model is fit by grid-initialised Nelder-Mead MLE, matching the style of the
# existing `_fit_garch11` in `Compute.jl`. Parameters are returned as NamedTuples; a
# `simulate_*` helper generates a return path from a fitted tuple.
#
# FIGARCH is documented as a separate follow-up item (see `revision-code-todo.md` M7).
# ========================================================================================= #

using Statistics;
using Distributions;

# ========================================================================================= #
# Shared: grid-initialised Nelder-Mead
# ========================================================================================= #

"""
    _nelder_mead(nll, params; max_iter=2000, tol=1e-8) -> (best_params, best_nll)

Generic Nelder-Mead minimiser for a scalar negative-log-likelihood. Initial simplex is
built by perturbing each dimension by 15% (with an additive floor for zero entries);
reflection / expansion / contraction / shrink operations match the existing GARCH(1,1)
implementation.
"""
function _nelder_mead(nll::Function, params::Vector{Float64};
                     max_iter::Int=2000, tol::Float64=1e-8)
    d = length(params);
    simplex = [copy(params) for _ in 1:(d + 1)];
    for i in 2:(d + 1)
        step = abs(simplex[i][i-1]) > 1e-12 ? simplex[i][i-1] * 0.15 : 0.05;
        simplex[i][i-1] += step;
    end
    for _ in 1:max_iter
        vals = [nll(s) for s in simplex];
        order = sortperm(vals);
        simplex = simplex[order];
        vals = vals[order];
        if abs(vals[end] - vals[1]) < tol; break; end
        centroid = sum(simplex[1:d]) ./ d;
        reflected = centroid .+ (centroid .- simplex[end]);
        f_r = nll(reflected);
        if f_r < vals[1]
            expanded = centroid .+ 2.0 .* (reflected .- centroid);
            f_e = nll(expanded);
            simplex[end] = f_e < f_r ? expanded : reflected;
        elseif f_r < vals[d]
            simplex[end] = reflected;
        else
            contracted = centroid .+ 0.5 .* (simplex[end] .- centroid);
            f_c = nll(contracted);
            if f_c < vals[end]
                simplex[end] = contracted;
            else
                for i in 2:(d + 1)
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1]);
                end
            end
        end
    end
    vals = [nll(s) for s in simplex];
    idx = argmin(vals);
    return simplex[idx], vals[idx];
end

# ========================================================================================= #
# EGARCH(1,1) with Gaussian innovations
# Recursion: log σ²_t = ω + β log σ²_{t-1} + α (|z_{t-1}| - E[|z|]) + γ z_{t-1}
# where z_t = (r_t - μ) / σ_t and E[|z|] = sqrt(2/π) under Gaussian z.
# ========================================================================================= #

const _SQRT_2_OVER_PI = sqrt(2.0 / π);

function _egarch11_nll(p::Vector{Float64}, obs::Vector{Float64})::Float64
    ω, α, γ, β, μ = p[1], p[2], p[3], p[4], p[5];
    if abs(β) >= 0.999; return 1e10; end
    N = length(obs);
    log_σ2 = log(max(var(obs), 1e-8));
    ll = 0.0;
    for t in 1:N
        σ2 = exp(log_σ2);
        r = obs[t] - μ;
        ll += -0.5 * (log(2π) + log_σ2 + r^2 / σ2);
        z = r / sqrt(σ2);
        log_σ2_next = ω + β * log_σ2 + α * (abs(z) - _SQRT_2_OVER_PI) + γ * z;
        log_σ2 = clamp(log_σ2_next, -30.0, 30.0);
    end
    return isfinite(ll) ? -ll : 1e10;
end

"""
    fit_egarch11(obs) -> NamedTuple

Fits EGARCH(1,1) by grid-initialised Nelder-Mead MLE. Returns
`(ω, α, γ, β, μ, log_sigma2_hist, ll)`.
"""
function fit_egarch11(obs::Vector{Float64})
    μ_init = mean(obs);
    best_nll = Inf; best_params = zeros(5);
    for α_try in [0.05, 0.10, 0.15]
        for β_try in [0.90, 0.95]
            for γ_try in [-0.10, -0.05, 0.0]
                ω_try = (1 - β_try) * log(var(obs));
                p = [ω_try, α_try, γ_try, β_try, μ_init];
                nll = _egarch11_nll(p, obs);
                if nll < best_nll; best_nll = nll; best_params = copy(p); end
            end
        end
    end
    best, _ = _nelder_mead(p -> _egarch11_nll(p, obs), best_params; max_iter=3000);
    ω, α, γ, β, μ = best[1], best[2], best[3], best[4], best[5];
    N = length(obs);
    log_σ2_hist = zeros(N);
    log_σ2_hist[1] = log(max(var(obs), 1e-8));
    for t in 2:N
        r_prev = obs[t-1] - μ;
        σ_prev = sqrt(exp(log_σ2_hist[t-1]));
        z_prev = r_prev / σ_prev;
        log_σ2_hist[t] = ω + β * log_σ2_hist[t-1] + α * (abs(z_prev) - _SQRT_2_OVER_PI) + γ * z_prev;
        log_σ2_hist[t] = clamp(log_σ2_hist[t], -30.0, 30.0);
    end
    ll = -_egarch11_nll(best, obs);
    return (ω=ω, α=α, γ=γ, β=β, μ=μ, log_σ2_hist=log_σ2_hist, ll=ll);
end

"""
    simulate_egarch(fit, n_steps) -> Vector{Float64}
"""
function simulate_egarch(fit, n_steps::Int)::Vector{Float64}
    returns = zeros(n_steps);
    log_σ2 = fit.log_σ2_hist[end];
    for t in 1:n_steps
        σ = sqrt(exp(log_σ2));
        z = randn();
        returns[t] = fit.μ + σ * z;
        log_σ2 = fit.ω + fit.β * log_σ2 + fit.α * (abs(z) - _SQRT_2_OVER_PI) + fit.γ * z;
        log_σ2 = clamp(log_σ2, -30.0, 30.0);
    end
    return returns;
end

# ========================================================================================= #
# GJR-GARCH(1,1) with Gaussian innovations
# σ²_t = ω + α ε²_{t-1} + γ ε²_{t-1} I(ε_{t-1} < 0) + β σ²_{t-1}
# Stationarity: α + β + γ/2 < 1.
# ========================================================================================= #

function _gjr11_nll(p::Vector{Float64}, obs::Vector{Float64})::Float64
    ω, α, γ, β, μ = p[1], p[2], p[3], p[4], p[5];
    if ω <= 0 || α < 0 || β < 0 || (α + γ) < 0 || (α + β + γ/2) >= 0.999; return 1e10; end
    N = length(obs);
    σ2 = ω / max(1.0 - α - β - γ/2, 1e-6);
    ll = 0.0;
    for t in 1:N
        r = obs[t] - μ;
        ll += -0.5 * (log(2π) + log(max(σ2, 1e-12)) + r^2 / max(σ2, 1e-12));
        ind = r < 0 ? 1.0 : 0.0;
        σ2 = ω + α * r^2 + γ * r^2 * ind + β * σ2;
        σ2 = max(σ2, 1e-12);
    end
    return isfinite(ll) ? -ll : 1e10;
end

function fit_gjr11(obs::Vector{Float64})
    μ_init = mean(obs);
    var_obs = var(obs);
    best_nll = Inf; best_params = zeros(5);
    for α_try in [0.02, 0.05]
        for β_try in [0.85, 0.90]
            for γ_try in [0.05, 0.10, 0.15]
                denom = 1 - α_try - β_try - γ_try/2;
                if denom > 0.01
                    ω_try = var_obs * denom;
                    p = [ω_try, α_try, γ_try, β_try, μ_init];
                    nll = _gjr11_nll(p, obs);
                    if nll < best_nll; best_nll = nll; best_params = copy(p); end
                end
            end
        end
    end
    best, _ = _nelder_mead(p -> _gjr11_nll(p, obs), best_params; max_iter=3000);
    ω, α, γ, β, μ = best[1], best[2], best[3], best[4], best[5];
    N = length(obs);
    σ2_hist = zeros(N);
    σ2_hist[1] = ω / max(1.0 - α - β - γ/2, 1e-6);
    for t in 2:N
        r_prev = obs[t-1] - μ;
        ind = r_prev < 0 ? 1.0 : 0.0;
        σ2_hist[t] = ω + α * r_prev^2 + γ * r_prev^2 * ind + β * σ2_hist[t-1];
        σ2_hist[t] = max(σ2_hist[t], 1e-12);
    end
    ll = -_gjr11_nll(best, obs);
    return (ω=ω, α=α, γ=γ, β=β, μ=μ, σ2_hist=σ2_hist, ll=ll);
end

function simulate_gjr(fit, n_steps::Int)::Vector{Float64}
    returns = zeros(n_steps);
    σ2 = fit.σ2_hist[end];
    for t in 1:n_steps
        σ = sqrt(max(σ2, 1e-12));
        returns[t] = fit.μ + σ * randn();
        r = returns[t] - fit.μ;
        ind = r < 0 ? 1.0 : 0.0;
        σ2 = fit.ω + fit.α * r^2 + fit.γ * r^2 * ind + fit.β * σ2;
        σ2 = max(σ2, 1e-12);
    end
    return returns;
end

# ========================================================================================= #
# GARCH(1,1) with Student-t innovations
# σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}; ε_t = σ_t * z_t with z_t ~ t_ν standardised to unit variance.
# ν > 2; z_t = η_t * sqrt((ν-2)/ν) with η_t ~ t_ν (non-standardised).
# ========================================================================================= #

function _garcht11_nll(p::Vector{Float64}, obs::Vector{Float64})::Float64
    ω, α, β, μ, ν = p[1], p[2], p[3], p[4], p[5];
    if ω <= 0 || α < 0 || β < 0 || (α + β) >= 0.999 || ν <= 2.1 || ν > 200; return 1e10; end
    N = length(obs);
    σ2 = ω / max(1.0 - α - β, 1e-6);
    ll = 0.0;
    # Student-t log density with scale σ*sqrt((ν-2)/ν) so that Var(ε) = σ²
    c = lgamma((ν + 1) / 2) - lgamma(ν / 2) - 0.5 * log(π * (ν - 2));
    for t in 1:N
        r = obs[t] - μ;
        σ = sqrt(max(σ2, 1e-12));
        z = r / σ;
        ll += c - log(σ) - 0.5 * (ν + 1) * log(1 + z^2 / (ν - 2));
        σ2 = ω + α * r^2 + β * σ2;
        σ2 = max(σ2, 1e-12);
    end
    return isfinite(ll) ? -ll : 1e10;
end

function fit_garcht11(obs::Vector{Float64})
    μ_init = mean(obs);
    var_obs = var(obs);
    best_nll = Inf; best_params = zeros(5);
    for α_try in [0.05, 0.10]
        for β_try in [0.85, 0.90]
            for ν_try in [4.0, 6.0, 10.0]
                ω_try = var_obs * (1 - α_try - β_try);
                p = [ω_try, α_try, β_try, μ_init, ν_try];
                nll = _garcht11_nll(p, obs);
                if nll < best_nll; best_nll = nll; best_params = copy(p); end
            end
        end
    end
    best, _ = _nelder_mead(p -> _garcht11_nll(p, obs), best_params; max_iter=3000);
    ω, α, β, μ, ν = best[1], best[2], best[3], best[4], best[5];
    N = length(obs);
    σ2_hist = zeros(N);
    σ2_hist[1] = ω / max(1.0 - α - β, 1e-6);
    for t in 2:N
        r_prev = obs[t-1] - μ;
        σ2_hist[t] = ω + α * r_prev^2 + β * σ2_hist[t-1];
        σ2_hist[t] = max(σ2_hist[t], 1e-12);
    end
    ll = -_garcht11_nll(best, obs);
    return (ω=ω, α=α, β=β, μ=μ, ν=ν, σ2_hist=σ2_hist, ll=ll);
end

function simulate_garcht(fit, n_steps::Int)::Vector{Float64}
    returns = zeros(n_steps);
    σ2 = fit.σ2_hist[end];
    # Standardised t_ν innovation: draw η from TDist(ν), divide by sqrt(ν/(ν-2)).
    td = TDist(fit.ν);
    scale_to_unit_var = sqrt((fit.ν - 2.0) / fit.ν);
    for t in 1:n_steps
        σ = sqrt(max(σ2, 1e-12));
        z = rand(td) * scale_to_unit_var;
        returns[t] = fit.μ + σ * z;
        r = returns[t] - fit.μ;
        σ2 = fit.ω + fit.α * r^2 + fit.β * σ2;
        σ2 = max(σ2, 1e-12);
    end
    return returns;
end

# ========================================================================================= #
# HAR-RV on daily squared returns
# log RV_t = β_0 + β_d log RV_{t-1} + β_w log RV^{(5)}_{t-1} + β_m log RV^{(22)}_{t-1} + η_t
# with RV^{(h)}_t = (1/h) Σ_{s=t-h+1}^{t} RV_s, using daily squared returns as the RV proxy.
# Simulation: at each t draw η_t ~ N(0, σ_η²) fit from residuals, roll forward the recursion,
# then r_t = μ + sqrt(RV_t) * z_t with z_t ~ N(0, 1).
# ========================================================================================= #

function fit_harrv(obs::Vector{Float64})
    μ = mean(obs);
    RV = (obs .- μ).^2;
    RV_log = log.(max.(RV, 1e-12));
    N = length(obs);
    # Build lagged regressors starting at t = 23 (need 22 prior days)
    t_start = 23;
    n = N - t_start + 1;
    y = Vector{Float64}(undef, n);
    X = Matrix{Float64}(undef, n, 4);
    for i in 1:n
        t = t_start + i - 1;
        y[i] = RV_log[t];
        X[i, 1] = 1.0;
        X[i, 2] = RV_log[t-1];
        X[i, 3] = mean(RV_log[(t-5):(t-1)]);
        X[i, 4] = mean(RV_log[(t-22):(t-1)]);
    end
    β = (X' * X) \ (X' * y);
    resid = y .- X * β;
    σ_η = std(resid);
    return (β=β, μ=μ, σ_η=σ_η, RV_log_hist=RV_log, σ2_hist=exp.(RV_log), ll=NaN);
end

function simulate_harrv(fit, n_steps::Int)::Vector{Float64}
    returns = zeros(n_steps);
    # Seed the lag buffer with the last 22 log-RV values from the fit window.
    buf = copy(fit.RV_log_hist[(end-21):end]);
    for t in 1:n_steps
        # Predict log RV_t
        log_rv_pred = fit.β[1] +
                      fit.β[2] * buf[end] +
                      fit.β[3] * mean(buf[(end-4):end]) +
                      fit.β[4] * mean(buf);
        log_rv_pred += fit.σ_η * randn();
        log_rv_pred = clamp(log_rv_pred, -30.0, 30.0);
        σ = sqrt(exp(log_rv_pred));
        returns[t] = fit.μ + σ * randn();
        push!(buf, log_rv_pred);
        popfirst!(buf);
    end
    return returns;
end
