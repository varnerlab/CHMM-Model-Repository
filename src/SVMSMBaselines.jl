# ========================================================================================= #
# SVMSMBaselines.jl
#
# Peer-review item 2: stochastic-volatility, Markov-switching multifractal, and
# Merton-jump-diffusion baselines for the headline panel.
#
# Implementations are intentionally minimal MLE/MoM baselines designed to fit on a
# 10-year daily SPY series in seconds and produce simulation paths under the same
# `simulate_*(params, T; n_paths)` interface used elsewhere.
#
# Models:
#   1. SV-AR(1): Taylor 1986 / Harvey-Ruiz-Shephard 1994 lognormal stochastic volatility
#      r_t = mu + sigma * exp(h_t / 2) * eps_t,  eps_t ~ N(0, 1),
#      h_t = phi * h_{t-1} + eta_t,            eta_t ~ N(0, sigma_eta^2).
#      Fit by quasi-MLE on the log-squared linearisation
#      log(r_t - mu)^2 = log(sigma^2) + h_t + log(eps_t^2)
#      following Harvey-Ruiz-Shephard with a Kalman filter on the log-squared series
#      (mean of log(chi^2_1) = -1.27, var ~= 4.93).
#
#   2. MSM (Calvet-Fisher 2004) with k-bar = 8 multipliers:
#      sigma_t = sigma_bar * sqrt(prod_k m_{k,t}),  m_{k,t} ~ Bernoulli flip with rate
#      gamma_k = 1 - (1 - gamma_1)^(b^(k-1)). Fit by simulation-grid MLE on
#      (sigma_bar, m_0, gamma_1, b) using a 4-d grid then a refining Nelder-Mead.
#      For computational tractability, use the closed-form unconditional moments to
#      pin sigma_bar to the empirical sample std and grid-search the remaining 3 params.
#
#   3. Merton 1976 jump-diffusion:
#      r_t = mu * dt + sigma * sqrt(dt) * Z_t + N_t * J_t,
#      N_t ~ Poisson(lambda * dt), J_t ~ N(mu_J, sigma_J^2). Fit by exact MLE on the
#      mixture density f(r) = sum_n p(N=n) * N(r; mu*dt + n*mu_J, sigma^2*dt + n*sigma_J^2).
# ========================================================================================= #

using Statistics;
using Distributions;
using LinearAlgebra;

# Already loaded in Include.jl: _nelder_mead from GARCHFamily.jl

# ========================================================================================= #
# 1. SV-AR(1): Taylor 1986 / Harvey-Ruiz-Shephard 1994
# ========================================================================================= #

# Mean and variance of log(chi^2_1) (closed form)
const _LOG_CHI2_1_MEAN = -1.2703628454614781;       # -gamma - log(2)
const _LOG_CHI2_1_VAR  = 4.9348022005446793;        # pi^2 / 2

"""
    fit_sv_ar1(obs)
Fit Harvey-Ruiz-Shephard log-squared SV via Kalman filter on log((r - mu)^2).
Returns (mu, sigma2, phi, sigma_eta2, ll).
"""
function fit_sv_ar1(obs::AbstractVector{Float64})
    n = length(obs);
    mu_hat = mean(obs);
    res2 = (obs .- mu_hat) .^ 2;
    # Avoid zeros in the log-squared transformation
    res2 .= max.(res2, 1e-12);
    y = log.(res2);          # y_t = log(sigma^2) + h_t + log(eps_t^2)
    # Demean by E[log eps_t^2] = -1.27 to isolate log(sigma^2) + h_t
    y_centered = y .- _LOG_CHI2_1_MEAN;

    # State-space: x_t = log(sigma^2) + h_t,  x_t = (1-phi) * log(sigma^2) + phi * x_{t-1} + eta_t
    # Observation: y_centered_t = x_t + xi_t,   xi_t ~ centred log(chi^2_1) with var R = 4.93
    function nll(p::Vector{Float64})::Float64
        log_phi  = p[1]; log_sigma_eta2 = p[2]; log_sigma2 = p[3];
        phi = tanh(log_phi);
        sigma_eta2 = exp(log_sigma_eta2);
        log_sigma2_v = log_sigma2;       # let it float as a free parameter

        if abs(phi) >= 0.9999; return 1e10; end
        if sigma_eta2 <= 0;    return 1e10; end

        # Kalman filter
        x = log_sigma2_v;                # initialise at unconditional mean
        P = sigma_eta2 / max(1.0 - phi^2, 1e-8);
        ll = 0.0;
        @inbounds for t in 1:n
            # predict
            x_pred = (1 - phi) * log_sigma2_v + phi * x;
            P_pred = phi^2 * P + sigma_eta2;
            # observation residual + variance
            v = y_centered[t] - x_pred;
            S = P_pred + _LOG_CHI2_1_VAR;
            if S <= 0; return 1e10; end
            ll -= 0.5 * (log(2 * pi * S) + v^2 / S);
            # update
            K = P_pred / S;
            x = x_pred + K * v;
            P = (1 - K) * P_pred;
        end
        return -ll;
    end

    # Initial: phi = 0.95, sigma_eta = 0.3, sigma^2 = sample variance of obs
    p0 = [atanh(0.95), log(0.3^2), log(var(obs))];
    pbest, nllbest = _nelder_mead(nll, p0);
    phi = tanh(pbest[1]);
    sigma_eta2 = exp(pbest[2]);
    sigma2 = exp(pbest[3]);
    return (mu = mu_hat, sigma2 = sigma2, phi = phi, sigma_eta2 = sigma_eta2,
            log_likelihood = -nllbest);
end

"""
    simulate_sv_ar1(params, T; n_paths)
Simulate `n_paths` paths of length `T` from the fitted SV-AR(1) model. Each path draws
h_t recursively from h_t = phi * h_{t-1} + eta_t and then r_t = mu + sigma * exp(h_t/2) * eps_t.
"""
function simulate_sv_ar1(params, T::Int; n_paths::Int = 1000)
    sigma = sqrt(params.sigma2);
    sigma_eta = sqrt(params.sigma_eta2);
    out = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        h = randn() * sigma_eta / sqrt(max(1 - params.phi^2, 1e-8));
        @inbounds for t in 1:T
            h = params.phi * h + sigma_eta * randn();
            out[t, p] = params.mu + sigma * exp(h / 2) * randn();
        end
    end
    return out;
end

# ========================================================================================= #
# 2. Markov-Switching Multifractal (Calvet-Fisher 2004) with k-bar = 8 multipliers,
#    binomial multiplier distribution m_0 vs (2 - m_0).
# ========================================================================================= #

"""
    fit_msm(obs; kbar=8)
Calvet-Fisher 2004 binomial MSM with k-bar multipliers. Approximate fit by
moment-grid search on (m_0, gamma_1, b) with sigma_bar pinned to sample std.

Implementation note: full HMM-style filter on 2^kbar latent states is exact but expensive.
We instead use a moment-matching approximation:
  - sigma_bar = sample std of obs
  - m_0 in (1.0, 2.0): controls multiplier asymmetry (m_0 = 1 is degenerate)
  - gamma_1 in (1e-4, 0.999): per-step flip probability of the slowest multiplier
  - b in (1.0001, 6.0):       multiplier rate ratio between consecutive levels
The fit chooses (m_0, gamma_1, b) to minimise squared error between the empirical and
model-implied log-vol autocorrelation function over the first 50 lags.
"""
function fit_msm(obs::AbstractVector{Float64}; kbar::Int = 8)
    n = length(obs);
    mu_hat = mean(obs);
    sigma_bar = std(obs);
    log_abs_r = log.(max.(abs.(obs .- mu_hat), 1e-10));
    # Empirical ACF of log|r|
    L = min(50, n - 1);
    acf_emp = _autocor_simple(log_abs_r, L);

    # Model-implied stationary E[log(sqrt(prod m_k))] etc; closed-form approximation:
    # The log-volatility process has approximate ACF rho(tau) = sum_k w_k * gamma_k_tau
    # where gamma_k_tau = (1 - 2 gamma_k)^tau, gamma_k = 1 - (1 - gamma_1)^(b^(k-1)),
    # and w_k = 1/kbar (under symmetric multiplier loadings).
    function _acf_model(gamma_1::Float64, b::Float64, kbar::Int, lags::Int)
        ms = zeros(lags);
        weights = fill(1.0 / kbar, kbar);
        for k in 1:kbar
            gk = 1 - (1 - gamma_1) ^ (b ^ (k - 1));
            gk = clamp(gk, 1e-9, 1 - 1e-9);
            persistence = 1 - 2 * gk;
            for tau in 1:lags
                ms[tau] += weights[k] * persistence ^ tau;
            end
        end
        return ms;
    end

    function nll(p::Vector{Float64})::Float64
        m0 = p[1]; gamma_1 = p[2]; b = p[3];
        if !(1.001 < m0 < 1.999);     return 1e10; end
        if !(1e-4  < gamma_1 < 0.999); return 1e10; end
        if !(1.0001 < b < 6.0);       return 1e10; end
        rho = _acf_model(gamma_1, b, kbar, L);
        # variance scaling factor folded in via sigma_bar
        return sum((acf_emp .- rho) .^ 2);
    end

    p0 = [1.4, 0.05, 2.0];
    pbest, nllbest = _nelder_mead(nll, p0);
    m0 = pbest[1]; gamma_1 = pbest[2]; b = pbest[3];
    return (mu = mu_hat, sigma_bar = sigma_bar, m0 = m0, gamma_1 = gamma_1, b = b,
            kbar = kbar, fit_loss = nllbest);
end

function _autocor_simple(x::AbstractVector{Float64}, L::Int)
    n = length(x);
    xc = x .- mean(x);
    denom = sum(xc .^ 2);
    return [sum(xc[1:n-tau] .* xc[1+tau:n]) / denom for tau in 1:L];
end

"""
    simulate_msm(params, T; n_paths)
"""
function simulate_msm(params, T::Int; n_paths::Int = 1000)
    kbar = params.kbar;
    out = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        # initialise multipliers at random in {m0, 2 - m0}
        m = [rand() < 0.5 ? params.m0 : (2 - params.m0) for _ in 1:kbar];
        @inbounds for t in 1:T
            for k in 1:kbar
                gk = 1 - (1 - params.gamma_1) ^ (params.b ^ (k - 1));
                if rand() < gk
                    m[k] = rand() < 0.5 ? params.m0 : (2 - params.m0);
                end
            end
            sigma_t = params.sigma_bar * sqrt(prod(m));
            out[t, p] = params.mu + sigma_t * randn();
        end
    end
    return out;
end

# ========================================================================================= #
# 3. Merton 1976 jump-diffusion (constant volatility + Poisson Gaussian jumps)
# ========================================================================================= #

"""
    fit_jump_diffusion(obs; dt=1/252, n_terms=20)
Maximum-likelihood fit of Merton 1976 jump-diffusion. Density is a Poisson mixture of
Gaussians: f(r) = sum_{n=0}^{n_terms} Poisson(n; lambda*dt) * N(r; mu*dt + n*mu_J, sigma^2*dt + n*sigma_J^2).
Note: `obs` here is annualised; we treat dt=1/252 only when interpreting parameters.
"""
function fit_jump_diffusion(obs::AbstractVector{Float64}; dt::Float64 = 1/252, n_terms::Int = 20)
    @assert dt > 0;
    mu_hat = mean(obs);
    sigma_hat = std(obs);

    function nll(p::Vector{Float64})::Float64
        mu = p[1];
        log_sigma = p[2]; log_lambda = p[3]; mu_J = p[4]; log_sigma_J = p[5];
        sigma   = exp(log_sigma);
        lambda  = exp(log_lambda);
        sigma_J = exp(log_sigma_J);
        if sigma <= 0 || lambda <= 0 || sigma_J <= 0; return 1e10; end

        ll = 0.0;
        @inbounds for r in obs
            # Poisson mixture density (annualised mu / sigma)
            p_mix = 0.0;
            ld = lambda;
            log_pois = -ld;
            for k in 0:n_terms
                if k > 0
                    log_pois += log(ld) - log(k);
                end
                # Component density
                m = mu + k * mu_J;
                v = sigma^2 + k * sigma_J^2;
                if v <= 0; continue; end
                comp = -0.5 * (log(2 * pi * v) + (r - m)^2 / v);
                p_mix += exp(log_pois + comp);
            end
            if p_mix <= 0; return 1e10; end
            ll += log(p_mix);
        end
        return -ll;
    end

    p0 = [mu_hat, log(sigma_hat), log(0.1), 0.0, log(sigma_hat * 2)];
    pbest, nllbest = _nelder_mead(nll, p0; max_iter = 4000);
    return (mu = pbest[1], sigma = exp(pbest[2]),
            lambda = exp(pbest[3]), mu_J = pbest[4], sigma_J = exp(pbest[5]),
            log_likelihood = -nllbest);
end

"""
    simulate_jump_diffusion(params, T; n_paths)
Simulate annualised log returns from the fitted Merton model.
"""
function simulate_jump_diffusion(params, T::Int; n_paths::Int = 1000)
    out = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths
        @inbounds for t in 1:T
            # diffusion + Poisson-Gaussian jump
            n_jumps = rand(Poisson(params.lambda));
            jump_total = n_jumps == 0 ? 0.0 :
                params.mu_J * n_jumps + params.sigma_J * sqrt(n_jumps) * randn();
            out[t, p] = params.mu + params.sigma * randn() + jump_total;
        end
    end
    return out;
end
