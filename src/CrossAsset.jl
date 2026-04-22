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


# --- TRUNCATED LEVEL-1 C-VINE COPULA ---------------------------------------- #

"""
    _gaussian_pair_loglik(u::AbstractVector, v::AbstractVector, ρ::Float64) -> Float64

Log-likelihood of a bivariate Gaussian copula evaluated on pseudo-uniform data.
"""
function _gaussian_pair_loglik(u::AbstractVector, v::AbstractVector, ρ::Float64)::Float64
    ρ = clamp(ρ, -0.995, 0.995);
    x = quantile.(Normal(), u);
    y = quantile.(Normal(), v);
    denom = 1.0 - ρ^2;
    ll = 0.0;
    for i in eachindex(x)
        ll += -0.5 * log(denom) - (ρ^2 * (x[i]^2 + y[i]^2) - 2.0 * ρ * x[i] * y[i]) / (2.0 * denom);
    end
    return ll;
end

"""
    _t_pair_loglik(u::AbstractVector, v::AbstractVector, ρ::Float64, ν::Float64) -> Float64

Log-likelihood of a bivariate Student-t copula evaluated on pseudo-uniform data.
"""
function _t_pair_loglik(u::AbstractVector, v::AbstractVector, ρ::Float64, ν::Float64)::Float64
    ρ = clamp(ρ, -0.995, 0.995);
    x = quantile.(TDist(ν), u);
    y = quantile.(TDist(ν), v);
    Σ = [1.0 ρ; ρ 1.0];
    mv = MvTDist(ν, zeros(2), Σ);
    uni = TDist(ν);
    ll = 0.0;
    for i in eachindex(x)
        ll += logpdf(mv, [x[i], y[i]]) - logpdf(uni, x[i]) - logpdf(uni, y[i]);
    end
    return ll;
end

"""
    _fit_pair_copula(u::AbstractVector, v::AbstractVector;
                     nu_grid::Vector{Float64}=...) -> NamedTuple

Fit one bivariate pair-copula by AIC, choosing between Gaussian and Student-t.
The correlation parameter is estimated from Kendall's τ via the usual inversion.
"""
function _fit_pair_copula(u::AbstractVector, v::AbstractVector;
                          nu_grid::Vector{Float64}=Float64[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0])
    τ = corkendall(u, v);
    ρ = clamp(sin(π * τ / 2.0), -0.995, 0.995);

    ll_g = _gaussian_pair_loglik(u, v, ρ);
    best = (family=:gaussian, rho=ρ, nu=Inf, ll=ll_g, aic=2.0 - 2.0 * ll_g);

    for ν in nu_grid
        ll_t = _t_pair_loglik(u, v, ρ, ν);
        aic_t = 4.0 - 2.0 * ll_t;
        if aic_t < best.aic
            best = (family=:student_t, rho=ρ, nu=ν, ll=ll_t, aic=aic_t);
        end
    end
    return best;
end

"""
    build(::Type{MyTruncatedCVineCopulaModel}, data::NamedTuple) -> MyTruncatedCVineCopulaModel

Fits a truncated level-1 C-vine:

1. Pick the root asset with the largest sum of absolute Kendall's τ values.
2. Fit one bivariate pair-copula from the root to every remaining asset.
3. Choose the pair family (`:gaussian` or `:student_t`) by AIC.
"""
function build(model::Type{MyTruncatedCVineCopulaModel}, data::NamedTuple)::MyTruncatedCVineCopulaModel
    R = data.returns;
    tickers = data.tickers;
    marginals = data.marginals;
    ν_grid = haskey(data, :nu_grid) ? data.nu_grid : Float64[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0];

    U = _pit_ranks(R);
    τ = _kendall_tau_matrix(R);
    root_scores = [sum(abs.(τ[j, setdiff(1:size(τ, 1), [j])])) for j in 1:size(τ, 1)];
    root_index = argmax(root_scores);
    children = [j for j in 1:size(R, 2) if j != root_index];

    families = Symbol[];
    rhos = Float64[];
    nus = Float64[];
    aic = Float64[];

    u_root = U[:, root_index];
    for j in children
        fit = _fit_pair_copula(u_root, U[:, j]; nu_grid=ν_grid);
        push!(families, fit.family);
        push!(rhos, fit.rho);
        push!(nus, fit.nu);
        push!(aic, fit.aic);
    end

    m = model();
    m.tickers = tickers;
    m.root_index = root_index;
    m.children = children;
    m.families = families;
    m.rhos = rhos;
    m.nus = nus;
    m.aic = aic;
    m.marginals = marginals;
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
    _sample_truncated_cvine(model::MyTruncatedCVineCopulaModel, T::Int) -> Matrix{Float64}

Sample a T × d pseudo-uniform matrix from the truncated level-1 C-vine.
All non-root assets are conditionally sampled from the root under their fitted
pair-copula. This is a scalable one-factor vine approximation.
"""
function _sample_truncated_cvine(model::MyTruncatedCVineCopulaModel, T::Int)::Matrix{Float64}
    d = length(model.tickers);
    U = zeros(T, d);
    root_u = rand(T);
    U[:, model.root_index] = root_u;

    for (edge, j) in enumerate(model.children)
        ρ = model.rhos[edge];
        fam = model.families[edge];
        if fam === :gaussian
            z_root = quantile.(Normal(), root_u);
            z_child = ρ .* z_root .+ sqrt(1.0 - ρ^2) .* randn(T);
            U[:, j] = cdf.(Normal(), z_child);
        else
            ν = model.nus[edge];
            x_root = quantile.(TDist(ν), root_u);
            x_child = similar(x_root);
            for t in eachindex(x_root)
                scale = sqrt((ν + x_root[t]^2) * (1.0 - ρ^2) / (ν + 1.0));
                x_child[t] = ρ * x_root[t] + scale * rand(TDist(ν + 1.0));
            end
            U[:, j] = cdf.(TDist(ν), x_child);
        end
    end
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

function simulate(model::MyTruncatedCVineCopulaModel, T_sim::Int, n_paths::Int)::Array{Float64,3}
    d = length(model.tickers);
    out = zeros(T_sim, d, n_paths);
    starts = [_stationary(model.marginals[j]) for j in 1:d];
    for p in 1:n_paths
        U = _sample_truncated_cvine(model, T_sim);
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
