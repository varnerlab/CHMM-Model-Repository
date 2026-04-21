# ========================================================================================= #
# CrossAsset.jl: Cross-asset simulation via Single Index Model (SIM) and copulas
# ========================================================================================= #
#
# Two families of multi-asset generators that preserve per-asset CHMM marginals:
#
#   1. Single Index Model (SIM):
#        G_{i,t} = α_i + β_i * G_{M,t} + η_{i,t}
#      α_i, β_i fitted by OLS on in-sample data; η_{i,t} resampled from empirical
#      residuals. The market factor G_{M,t} is drawn from the market-asset CHMM.
#
#   2. Gaussian and Student-t copulas:
#      Per-asset CHMM paths are independently simulated, then re-ordered by rank
#      so that their ranks match a multivariate copula sample. This preserves
#      each asset's exact fitted marginal while injecting cross-asset dependence.
#
# ========================================================================================= #


# --- PROBABILITY-INTEGRAL TRANSFORM ------------------------------------------ #

"""
    _pit_ranks(X::Matrix{Float64}) -> Matrix{Float64}

Pseudo-uniform observations via the nonparametric PIT u = rank(x) / (n + 1).
Columns are treated as independent margins.
"""
function _pit_ranks(X::Matrix{Float64})::Matrix{Float64}
    T, d = size(X);
    U = zeros(T, d);
    for j in 1:d
        r = ordinalrank(X[:, j]);
        U[:, j] = r ./ (T + 1.0);
    end
    return U;
end

"""
    _kendall_tau_matrix(X::Matrix{Float64}) -> Matrix{Float64}

Pairwise Kendall's τ matrix for columns of X. Diagonal is 1.
"""
function _kendall_tau_matrix(X::Matrix{Float64})::Matrix{Float64}
    d = size(X, 2);
    τ = Matrix{Float64}(I, d, d);
    for i in 1:d
        for j in (i+1):d
            τij = corkendall(X[:, i], X[:, j]);
            τ[i, j] = τij;
            τ[j, i] = τij;
        end
    end
    return τ;
end

"""
    _nearest_psd(A::Matrix{Float64}; eps_floor::Float64=1e-8) -> Matrix{Float64}

Project a symmetric matrix to the nearest positive semi-definite matrix
by clipping eigenvalues from below. Preserves diagonal at 1 after projection.
"""
function _nearest_psd(A::Matrix{Float64}; eps_floor::Float64=1e-8)::Matrix{Float64}
    S = 0.5 .* (A .+ A');
    F = eigen(Symmetric(S));
    λ = max.(F.values, eps_floor);
    B = F.vectors * Diagonal(λ) * F.vectors';
    dinv = 1.0 ./ sqrt.(diag(B));
    return Symmetric(Diagonal(dinv) * B * Diagonal(dinv)) |> Matrix;
end


# --- SIM: SINGLE INDEX MODEL ------------------------------------------------- #

"""
    build(::Type{MySingleIndexModel}, data::NamedTuple) -> MySingleIndexModel

Fits the Single Index Model G_{i,t} = α_i + β_i G_{M,t} + η_{i,t} by OLS on
in-sample data, one regression per asset.

### NamedTuple keys
- `returns::Matrix{Float64}`: T × d in-sample return matrix (each column an asset)
- `market::Vector{Float64}`: Length-T market factor return series
- `tickers::Vector{String}`: d asset tickers in column order
"""
function build(model::Type{MySingleIndexModel}, data::NamedTuple)::MySingleIndexModel

    R = data.returns;
    M = data.market;
    tickers = data.tickers;
    T, d = size(R);

    alphas = zeros(d);
    betas = zeros(d);
    residuals = zeros(T, d);
    r2 = zeros(d);

    M_mean = mean(M);
    M_var = var(M);

    for j in 1:d
        rj = R[:, j];
        rj_mean = mean(rj);
        cov_jm = mean((rj .- rj_mean) .* (M .- M_mean));
        β = cov_jm / M_var;
        α = rj_mean - β * M_mean;

        fitted = α .+ β .* M;
        resid = rj .- fitted;

        ss_tot = sum((rj .- rj_mean) .^ 2);
        ss_res = sum(resid .^ 2);
        r2_j = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;

        alphas[j] = α;
        betas[j] = β;
        residuals[:, j] = resid;
        r2[j] = r2_j;
    end

    m = model();
    m.tickers = tickers;
    m.alphas = alphas;
    m.betas = betas;
    m.residuals = residuals;
    m.r2 = r2;
    return m;
end

"""
    simulate(model::MySingleIndexModel, market_paths::Matrix{Float64}) -> Array{Float64,3}

Propagates simulated market-factor paths to a multi-asset universe via the SIM
factor equation. Idiosyncratic shocks η_{i,t} are resampled with replacement
from each asset's empirical in-sample residual distribution.

### Arguments
- `model::MySingleIndexModel`: Fitted SIM
- `market_paths::Matrix{Float64}`: T_sim × n_paths matrix of market factor paths

### Returns
- `Array{Float64,3}`: T_sim × d × n_paths tensor of simulated asset returns
"""
function simulate(model::MySingleIndexModel, market_paths::Matrix{Float64})::Array{Float64,3}
    T_sim, np = size(market_paths);
    d = length(model.tickers);
    T_resid = size(model.residuals, 1);
    sim = zeros(T_sim, d, np);
    for p in 1:np
        for j in 1:d
            η_idx = rand(1:T_resid, T_sim);
            sim[:, j, p] = model.alphas[j] .+ model.betas[j] .* market_paths[:, p] .+ model.residuals[η_idx, j];
        end
    end
    return sim;
end


# --- GAUSSIAN COPULA --------------------------------------------------------- #

"""
    build(::Type{MyGaussianCopulaModel}, data::NamedTuple) -> MyGaussianCopulaModel

Fits a Gaussian copula from in-sample returns. The correlation matrix is
estimated from Kendall's τ via ρ = sin(πτ/2). This rank-based estimator is
robust to outliers and marginal misspecification.

### NamedTuple keys
- `returns::Matrix{Float64}`: T × d in-sample return matrix
- `tickers::Vector{String}`: d asset tickers
- `marginals::Vector{AbstractMarkovModel}`: Per-asset fitted CHMMs
"""
function build(model::Type{MyGaussianCopulaModel}, data::NamedTuple)::MyGaussianCopulaModel

    R = data.returns;
    τ = _kendall_tau_matrix(R);
    Σ = sin.(π .* τ ./ 2.0);
    Σ = _nearest_psd(Σ);

    m = model();
    m.tickers = data.tickers;
    m.Sigma = Σ;
    m.marginals = data.marginals;
    return m;
end


# --- STUDENT-t COPULA -------------------------------------------------------- #

"""
    _tcopula_profile_loglik(U::Matrix{Float64}, Σ::Matrix{Float64}, ν::Float64) -> Float64

Student-t copula log-likelihood evaluated on pseudo-uniform PIT data U,
given correlation Σ and degrees-of-freedom ν.
"""
function _tcopula_profile_loglik(U::Matrix{Float64}, Σ::Matrix{Float64}, ν::Float64)::Float64
    T, d = size(U);
    X = similar(U);
    for j in 1:d
        X[:, j] = quantile.(TDist(ν), U[:, j]);
    end

    Σchol = cholesky(Symmetric(Σ));
    logdetΣ = 2.0 * sum(log.(diag(Σchol.U)));

    c = lgamma((ν + d) / 2) + (d - 1) * lgamma(ν / 2) - d * lgamma((ν + 1) / 2);
    c -= 0.5 * logdetΣ;

    ll = T * c;
    Σinv = inv(Symmetric(Σ));
    for t in 1:T
        x = X[t, :];
        q = dot(x, Σinv * x);
        ll += -((ν + d) / 2) * log(1.0 + q / ν);
        for j in 1:d
            ll += ((ν + 1) / 2) * log(1.0 + x[j]^2 / ν);
        end
    end
    return ll;
end

"""
    build(::Type{MyStudentTCopulaModel}, data::NamedTuple) -> MyStudentTCopulaModel

Fits a Student-t copula. Correlation matrix Σ is estimated from Kendall's τ
(same as the Gaussian copula). Degrees of freedom ν are selected by profile
maximum likelihood over ν ∈ {2, 3, 4, 5, 6, 8, 10, 15, 20, 30}.

### NamedTuple keys
- `returns::Matrix{Float64}`: T × d in-sample return matrix
- `tickers::Vector{String}`: d asset tickers
- `marginals::Vector{AbstractMarkovModel}`: Per-asset fitted CHMMs
- `nu_grid::Vector{Float64}` (optional): Grid for ν profile search
"""
function build(model::Type{MyStudentTCopulaModel}, data::NamedTuple)::MyStudentTCopulaModel

    R = data.returns;
    τ = _kendall_tau_matrix(R);
    Σ = sin.(π .* τ ./ 2.0);
    Σ = _nearest_psd(Σ);

    U = _pit_ranks(R);

    ν_grid = haskey(data, :nu_grid) ? data.nu_grid : Float64[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0];

    best_ν = ν_grid[1];
    best_ll = -Inf;
    for ν in ν_grid
        ll = _tcopula_profile_loglik(U, Σ, ν);
        if ll > best_ll
            best_ll = ll;
            best_ν = ν;
        end
    end

    m = model();
    m.tickers = data.tickers;
    m.Sigma = Σ;
    m.nu = best_ν;
    m.marginals = data.marginals;
    return m;
end


# --- COPULA SIMULATION VIA RANK REORDERING ----------------------------------- #

"""
    _sample_gaussian_copula(Σ::Matrix{Float64}, T::Int64) -> Matrix{Float64}

Draws T samples from a d-variate Gaussian copula with correlation Σ.
Returns T × d uniform variates.
"""
function _sample_gaussian_copula(Σ::Matrix{Float64}, T::Int64)::Matrix{Float64}
    d = size(Σ, 1);
    L = cholesky(Symmetric(Σ)).L;
    Z = randn(T, d) * L';
    U = cdf.(Normal(), Z);
    return U;
end

"""
    _sample_t_copula(Σ::Matrix{Float64}, ν::Float64, T::Int64) -> Matrix{Float64}

Draws T samples from a d-variate Student-t copula with correlation Σ and
degrees of freedom ν. Returns T × d uniform variates.
"""
function _sample_t_copula(Σ::Matrix{Float64}, ν::Float64, T::Int64)::Matrix{Float64}
    d = size(Σ, 1);
    L = cholesky(Symmetric(Σ)).L;
    Z = randn(T, d) * L';
    w = rand(Chisq(ν), T) ./ ν;
    X = Z ./ sqrt.(w);
    U = cdf.(TDist(ν), X);
    return U;
end

"""
    _simulate_chmm_marginal(chmm::MyContinuousHiddenMarkovModel, start_dist::Categorical, T::Int) -> Vector{Float64}

Simulate a T-step return path from a fitted CHMM.
"""
function _simulate_chmm_marginal(chmm::MyContinuousHiddenMarkovModel, start_dist::Categorical, T::Int)::Vector{Float64}
    s0 = rand(start_dist);
    states = chmm(s0, T);
    out = Vector{Float64}(undef, T);
    for t in 1:T
        out[t] = rand(chmm.emission[states[t]]);
    end
    return out;
end

"""
    _stationary(chmm::MyContinuousHiddenMarkovModel) -> Categorical

Compute the stationary distribution of the CHMM transition matrix by power
iteration, returned as a Categorical.
"""
function _stationary(chmm::MyContinuousHiddenMarkovModel)::Categorical
    K = length(chmm.states);
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = chmm.transition[i].p;
    end
    π_stat = (T_mat ^ 1000)[1, :];
    π_stat .= max.(π_stat, 1e-12);
    π_stat ./= sum(π_stat);
    return Categorical(π_stat);
end

"""
    simulate(model::Union{MyGaussianCopulaModel, MyStudentTCopulaModel},
             T_sim::Int, n_paths::Int) -> Array{Float64,3}

Generates multi-asset paths that preserve each asset's fitted CHMM marginal
while injecting cross-asset dependence via the copula. Procedure:

1. Simulate n_paths independent CHMM paths for each asset j ∈ {1, ..., d}.
2. Draw T_sim uniform variates U ∼ C from the fitted copula for each path.
3. Reorder the simulated observations so their ranks match rank(U_j).

Returns a T_sim × d × n_paths tensor.
"""
function simulate(model::MyGaussianCopulaModel, T_sim::Int, n_paths::Int)::Array{Float64,3}
    d = length(model.tickers);
    out = zeros(T_sim, d, n_paths);
    starts = [ _stationary(model.marginals[j]) for j in 1:d ];
    for p in 1:n_paths
        U = _sample_gaussian_copula(model.Sigma, T_sim);
        for j in 1:d
            g = _simulate_chmm_marginal(model.marginals[j], starts[j], T_sim);
            ord_g = sortperm(g);
            ord_u = ordinalrank(U[:, j]);
            reordered = Vector{Float64}(undef, T_sim);
            reordered[1:T_sim] = g[ord_g];
            out[:, j, p] = reordered[ord_u];
        end
    end
    return out;
end

function simulate(model::MyStudentTCopulaModel, T_sim::Int, n_paths::Int)::Array{Float64,3}
    d = length(model.tickers);
    out = zeros(T_sim, d, n_paths);
    starts = [ _stationary(model.marginals[j]) for j in 1:d ];
    for p in 1:n_paths
        U = _sample_t_copula(model.Sigma, model.nu, T_sim);
        for j in 1:d
            g = _simulate_chmm_marginal(model.marginals[j], starts[j], T_sim);
            ord_g = sortperm(g);
            ord_u = ordinalrank(U[:, j]);
            reordered = Vector{Float64}(undef, T_sim);
            reordered[1:T_sim] = g[ord_g];
            out[:, j, p] = reordered[ord_u];
        end
    end
    return out;
end


# --- CROSS-ASSET EVALUATION METRICS ------------------------------------------ #

"""
    correlation_reproduction(R_obs::Matrix{Float64},
                             sim::Array{Float64,3}) -> NamedTuple

Compares observed and simulated cross-asset correlation structure.
Returns:
- `frob_mean`: Mean Frobenius norm ‖Σ_sim - Σ_obs‖_F across paths
- `frob_std`: Standard deviation of the same
- `offdiag_mae`: Mean absolute error of off-diagonal correlations (averaged over paths)
"""
function correlation_reproduction(R_obs::Matrix{Float64}, sim::Array{Float64,3})
    d = size(R_obs, 2);
    Σ_obs = cor(R_obs);
    np = size(sim, 3);
    frobs = zeros(np);
    maes = zeros(np);
    for p in 1:np
        Σp = cor(sim[:, :, p]);
        frobs[p] = norm(Σp .- Σ_obs);
        mask = .!I(d);
        maes[p] = mean(abs.((Σp .- Σ_obs)[mask]));
    end
    return (frob_mean=mean(frobs), frob_std=std(frobs),
            offdiag_mae=mean(maes));
end

"""
    per_asset_ks_pass_rates(R_obs::Matrix{Float64}, sim::Array{Float64,3};
                            α::Float64=0.05) -> Vector{Float64}

Computes per-asset KS pass rates over the n_paths cross-asset simulations.
Returns a length-d vector where entry j is the fraction of paths for which
sim[:, j, p] fails to reject KS equivalence with R_obs[:, j].
"""
function per_asset_ks_pass_rates(R_obs::Matrix{Float64}, sim::Array{Float64,3};
                                 α::Float64=0.05)::Vector{Float64}
    d = size(R_obs, 2);
    np = size(sim, 3);
    rates = zeros(d);
    for j in 1:d
        obs_j = R_obs[:, j];
        passes = 0;
        for p in 1:np
            pv = pvalue(ApproximateTwoSampleKSTest(obs_j, sim[:, j, p]));
            if pv > α
                passes += 1;
            end
        end
        rates[j] = 100.0 * passes / np;
    end
    return rates;
end

# ----------------------------------------------------------------------------- #


# ========================================================================================= #
# VIX-Conditioned Regime Coupling (Phases 2 + 3 of the VIX-CHMM forecasting plan)
# ========================================================================================= #

"""
    fit_coupling(vix_states, equity_states, K_vix, K_eq; lag=1, alpha=1.0) -> Matrix{Float64}

Empirical row-stochastic coupling matrix `C[s^V, s^E] = P(S^E_t = s^E | S^V_{t-h} = s^V)`
estimated from Viterbi-decoded state sequences on a shared in-sample window.

### Arguments
- `vix_states::Vector{Int}`: Viterbi-decoded VIX regime sequence.
- `equity_states::Vector{Int}`: Viterbi-decoded equity regime sequence on the same dates.
- `K_vix::Int`, `K_eq::Int`: number of states in each model.

### Keyword arguments
- `lag::Int=1`: non-negative integer; `lag=1` means the coupling relates the equity regime
  at day `t` to the VIX regime at day `t - 1`, giving VIX leading-indicator semantics and
  avoiding the contemporaneous circularity flagged in the implementation plan.
- `alpha::Float64=1.0`: Laplace-smoothing pseudo-count per cell (protects against VIX
  regimes that appear rarely on the in-sample window).
"""
function fit_coupling(vix_states::Vector{Int}, equity_states::Vector{Int},
    K_vix::Int, K_eq::Int; lag::Int=1, alpha::Float64=1.0)::Matrix{Float64}

    @assert length(vix_states) == length(equity_states) "state vectors must be aligned and equal-length";
    @assert lag >= 0 "lag must be non-negative";
    counts = alpha .* ones(K_vix, K_eq);
    for t in (lag + 1):length(vix_states)
        s_v = vix_states[t - lag];
        s_e = equity_states[t];
        counts[s_v, s_e] += 1.0;
    end
    for i in 1:K_vix
        row_sum = sum(counts[i, :]);
        counts[i, :] ./= max(row_sum, 1e-12);
    end
    return counts;
end

"""
    coupling_entropy(C::Matrix{Float64}) -> Vector{Float64}

Per-row Shannon entropy `H(C[i, :])` in nats. Lower entropy indicates a more informative
coupling at that VIX state; values near `log(K^E)` indicate near-uniform rows.
"""
function coupling_entropy(C::Matrix{Float64})::Vector{Float64}
    K_v = size(C, 1);
    h = zeros(K_v);
    for i in 1:K_v
        row = view(C, i, :);
        acc = 0.0;
        for p in row
            if p > 0.0
                acc -= p * log(p);
            end
        end
        h[i] = acc;
    end
    return h;
end

"""
    simulate_prices_vix_conditioned(chmm_equity, chmm_vix, C, S0, n_steps;
        pi_V_init=nothing, Δt=1/252, risk_free_rate=0.0, n_paths=1) -> Vector or Matrix

VIX-conditioned equity price simulator (Phase 3, simplified). At each step:

1. Sample the next VIX regime via the VIX transition matrix (initial state drawn from
   `pi_V_init`, defaulting to the stationary distribution of the VIX CHMM).
2. Given the sampled VIX regime, sample the equity regime from the coupling row
   `C[s^V, :]`.
3. Sample the return from the equity CHMM's per-state emission distribution.
4. Roll the price forward through the project convention
   `P_t = P_{t-1} * exp((G_t + r_f) * Δt)`.

This produces equity price paths whose regime dynamics are driven by the simulated
VIX state, while preserving each equity CHMM's fitted per-state emission exactly.

### Arguments
- `chmm_equity::_ContinuousCHMM`: trained per-ticker equity CHMM.
- `chmm_vix::_ContinuousCHMM`: trained VIX CHMM.
- `C::Matrix{Float64}`: `K^V × K^E` coupling matrix from `fit_coupling`.
- `S0::Real`: initial spot price.
- `n_steps::Int`: number of return steps (output path length is `n_steps + 1`).

### Keyword arguments
- `pi_V_init::Union{Nothing,Vector{Float64}}=nothing`: initial VIX-state distribution;
  falls back to stationary when `nothing`.
- `Δt::Float64=1/252`, `risk_free_rate::Float64=0.0`: return convention.
- `n_paths::Int=1`: number of independent paths.
"""
function simulate_prices_vix_conditioned(
    chmm_equity::_ContinuousCHMM,
    chmm_vix::_ContinuousCHMM,
    C::Matrix{Float64},
    S0::Real,
    n_steps::Int;
    pi_V_init::Union{Nothing,Vector{Float64}}=nothing,
    Δt::Float64=1.0/252.0,
    risk_free_rate::Float64=0.0,
    n_paths::Int=1,
)

    K_vix = length(chmm_vix.states);
    K_eq  = length(chmm_equity.states);
    @assert size(C) == (K_vix, K_eq) "C must be K^V × K^E";

    T_V = _transition_matrix(chmm_vix);
    π0  = pi_V_init === nothing ? probs(_stationary_distribution(chmm_vix)) : copy(pi_V_init);
    π0 = max.(π0, 1e-12); π0 ./= sum(π0);
    init_dist = Categorical(π0);

    vix_row_dists = [Categorical(T_V[i, :] ./ sum(T_V[i, :])) for i in 1:K_vix];
    coup_row_dists = [Categorical(C[i, :] ./ sum(C[i, :])) for i in 1:K_vix];

    function _one_path()
        path = Vector{Float64}(undef, n_steps + 1);
        path[1] = Float64(S0);
        s_V = rand(init_dist);
        @inbounds for t in 1:n_steps
            s_E = rand(coup_row_dists[s_V]);
            r   = rand(chmm_equity.emission[s_E]);
            path[t + 1] = path[t] * exp((r + risk_free_rate) * Δt);
            s_V = rand(vix_row_dists[s_V]);
        end
        return path;
    end

    if n_paths == 1
        return _one_path();
    end

    paths = Matrix{Float64}(undef, n_steps + 1, n_paths);
    for p in 1:n_paths
        paths[:, p] = _one_path();
    end
    return paths;
end
