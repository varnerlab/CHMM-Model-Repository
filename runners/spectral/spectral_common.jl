# ========================================================================================= #
# spectral_common.jl — shared helpers for the spectral effective-rank runners.
#
# Third-review item 7: the previous diagnostics counted every complex eigenvalue as its
# own mode and defined the share budget on |w_k λ_k| per eigenvalue. For a complex-
# conjugate pair the real ACF contribution is 2 Re(w λ^τ), not 2 |w λ^τ|: phase and
# cancellation matter, and an eigenvalue-level modulus budget double-counts a pair while
# ignoring its oscillation. These helpers therefore:
#   1. group complex-conjugate eigenvalue pairs into single REAL damped-oscillatory
#      components before ranking (component count ≤ K - 1);
#   2. rank components by the absolute value of their SIGNED real lag-1 contribution and
#      define shares on the "absolute component-magnitude budget"
#      B = Σ_c |contrib_c(1)| — which is NOT a percentage of ρ(1) itself, since signed
#      contributions can cancel; ρ(1) = Σ_c contrib_c(1) is reported alongside;
#   3. reconstruct the theoretical ACF from the component sum and compare it against the
#      direct matrix formula ρ(τ) = (m' diag(π̄) T^τ m − (π̄'m)²) / σ²_|G| over
#      τ = 1..252, reporting the maximum absolute error as a numerical-fidelity check.
# ========================================================================================= #

using Random, LinearAlgebra, Statistics

"""
    _stationary_pi(T) -> Vector{Float64}

Stationary distribution of a row-stochastic transition matrix by a constrained
left-eigenvector linear solve (see CHANGELOG.md: fixed
matrix powers such as `(T^2000)[1, :]` carry no residual check and can retain
starting-row dependence for highly persistent chains). Asserts uniqueness of
the unit eigenvalue, non-negativity, normalization, and a small stationarity
residual before returning.
"""
function _stationary_pi(T::AbstractMatrix{Float64})
    K = size(T, 1);
    # uniqueness of the unit eigenvalue (irreducible + aperiodic chain)
    λ = eigvals(Matrix(T));
    n_unit = count(l -> abs(l - 1.0) < 1e-8, λ);
    n_unit == 1 || error("_stationary_pi: expected a unique unit eigenvalue, found $n_unit");
    # constrained solve: π' (T - I) = 0 with sum(π) = 1
    A = vcat(transpose(T) - I, ones(1, K));
    b = vcat(zeros(K), 1.0);
    π̄ = A \ b;
    @assert minimum(π̄) > -1e-10 "_stationary_pi: negative stationary mass $(minimum(π̄))";
    π̄ = max.(π̄, 0.0);
    π̄ ./= sum(π̄);
    resid = norm(transpose(π̄) * T - transpose(π̄), Inf);
    @assert resid < 1e-10 "_stationary_pi: stationarity residual $resid";
    return π̄;
end

"""
    _T_pibar_m(model, K; n_draw, seed) -> (T, π̄, m, M)

Transition matrix, stationary vector, and per-state moments
m_k = E[|G| | s = k], M_k = E[G² | s = k] for a fitted CHMM. For Normal
emissions the moments are closed-form (folded-normal mean; μ² + σ²), removing
Monte-Carlo variation from the model-vs-sample comparison (see CHANGELOG.md); non-Normal emissions fall back to `n_draw` sampled draws per
state under `seed`.
"""
function _T_pibar_m(model, K::Int; n_draw::Int, seed::Int=0)
    Random.seed!(seed);
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π̄ = _stationary_pi(T);
    m = zeros(K); M = zeros(K);
    for k in 1:K
        e = model.emission[k];
        if e isa Normal
            μ = mean(e); σ = std(e);
            z = μ / σ;
            m[k] = σ * sqrt(2 / π) * exp(-z^2 / 2) + μ * (1 - 2 * cdf(Normal(), -z));
            M[k] = μ^2 + σ^2;
        else
            s = [rand(e) for _ in 1:n_draw];
            m[k] = mean(abs.(s));
            M[k] = mean(s .^ 2);
        end
    end
    return T, π̄, m, M;
end

"""
    _spectral_components(T, π̄, m, M; lags=(1, 5, 20, 50), recon_maxlag=252)
        -> (σ²_G, components, κ_V, recon_err)

Grouped real spectral components of the absolute-return ACF identity.

Each component is a NamedTuple with fields
- `kind`      : `:real` or `:pair` (complex-conjugate pair combined)
- `abs_lam`   : |λ| of the component (shared by both members of a pair)
- `theta`     : oscillation angle (0.0 for real modes)
- `contrib`   : NamedTuple of SIGNED real contributions at each lag in `lags`
- `mag_t1`    : |contrib(1)|, the lag-1 ranking key
- `int_abs`   : Σ_{τ=1..recon_maxlag} |contrib(τ)|, the horizon-aware ranking
                key (integrated absolute contribution over the reported lag
                band; see CHANGELOG.md: a component small at lag 1 can
                matter at later lags if its eigenvalue is larger or its phase
                differs, so a lag-1 budget alone is not an effective-rank
                diagnostic for the full curve)

`recon_err` is max_{τ ≤ recon_maxlag} |Σ_c contrib_c(τ) − ρ_direct(τ)| with
ρ_direct from the matrix-power formula; it validates the eigendecomposition
numerically (diagonalizable T assumed, as in theory.tex).
"""
function _spectral_components(T::AbstractMatrix, π̄::AbstractVector, m::AbstractVector,
                              M::AbstractVector; lags=(1, 5, 20, 50), recon_maxlag::Int=252)
    K = length(m);
    F = eigen(T);
    λ = F.values;
    V = F.vectors;
    W = inv(V);
    κ_V = cond(V);

    σ²_G = π̄' * M - (π̄' * m)^2;
    μ_G  = π̄' * m;

    idx_one = argmin(abs.(λ .- 1.0));
    rest = setdiff(1:K, idx_one);

    # Normalised complex modal weights a_k = c_k / σ²_G. NOTE: the left-
    # eigenvector factor is the plain (unconjugated) product w_k' m — Julia's
    # dot() conjugates its first argument, which is wrong for complex modes.
    a = Dict{Int,ComplexF64}();
    for k in rest
        c_k = (transpose(m) * Diagonal(π̄) * V[:, k]) * (transpose(W[k, :]) * m);
        a[k] = c_k / σ²_G;
    end

    imag_tol = 1e-10;
    used = Set{Int}();
    components = NamedTuple[];
    _int_abs(f) = sum(abs(f(τ)) for τ in 1:recon_maxlag);
    for k in rest
        k in used && continue;
        if abs(imag(λ[k])) <= imag_tol
            push!(used, k);
            contrib = NamedTuple{Tuple(Symbol.("t", lags))}(
                Tuple(real(a[k]) * real(λ[k])^τ for τ in lags));
            push!(components, (kind=:real, abs_lam=abs(λ[k]), theta=0.0,
                               contrib=contrib, mag_t1=abs(contrib[1]),
                               int_abs=_int_abs(τ -> real(a[k]) * real(λ[k])^τ),
                               members=(k,)));
        else
            # Find the conjugate partner among unused non-unit eigenvalues.
            j = findfirst(j -> !(j in used) && j != k && abs(λ[j] - conj(λ[k])) < 1e-8, rest);
            if j === nothing
                # No clean partner (should not happen for a real T); fall back to
                # the real part of this eigenvalue's own contribution.
                push!(used, k);
                contrib = NamedTuple{Tuple(Symbol.("t", lags))}(
                    Tuple(real(a[k] * λ[k]^τ) for τ in lags));
                push!(components, (kind=:real, abs_lam=abs(λ[k]), theta=angle(λ[k]),
                                   contrib=contrib, mag_t1=abs(contrib[1]),
                                   int_abs=_int_abs(τ -> real(a[k] * λ[k]^τ)),
                                   members=(k,)));
            else
                jj = rest[j];
                push!(used, k); push!(used, jj);
                contrib = NamedTuple{Tuple(Symbol.("t", lags))}(
                    Tuple(real(a[k] * λ[k]^τ + a[jj] * λ[jj]^τ) for τ in lags));
                push!(components, (kind=:pair, abs_lam=abs(λ[k]), theta=abs(angle(λ[k])),
                                   contrib=contrib, mag_t1=abs(contrib[1]),
                                   int_abs=_int_abs(τ -> real(a[k] * λ[k]^τ + a[jj] * λ[jj]^τ)),
                                   members=(k, jj)));
            end
        end
    end
    sort!(components, by = c -> -c.mag_t1);

    # Reconstruction check: spectral sum vs direct matrix formula.
    recon_err = 0.0;
    Tτ = Matrix{Float64}(I, K, K);
    for τ in 1:recon_maxlag
        Tτ *= T;
        ρ_direct = (m' * Diagonal(π̄) * Tτ * m - μ_G^2) / σ²_G;
        ρ_spec = sum(real(a[k] * λ[k]^τ) for k in rest);
        recon_err = max(recon_err, abs(ρ_spec - ρ_direct));
    end

    return σ²_G, components, κ_V, recon_err;
end

"""
    _component_summary(components) -> NamedTuple

Budget shares over the grouped components: dom_share, n95, n99, n>1%, the
absolute budget B = Σ|contrib(1)|, and the signed ρ(1) = Σ contrib(1).
Horizon-aware counterparts (`dom_share_int`, `n_for_95_int`) rank by
`int_abs` = Σ_{τ ≤ 252} |contrib(τ)| instead of the lag-1 magnitude.
"""
function _component_summary(components)
    mags = [c.mag_t1 for c in components];
    B = sum(mags);
    rho1 = sum(c.contrib[1] for c in components);
    sorted = sort(mags; rev=true);
    cum = cumsum(sorted) ./ B;
    n95 = findfirst(>=(0.95), cum);
    n99 = findfirst(>=(0.99), cum);
    ints = sort([c.int_abs for c in components]; rev=true);
    B_int = sum(ints);
    cum_int = cumsum(ints) ./ B_int;
    n95_int = findfirst(>=(0.95), cum_int);
    return (dom_share = sorted[1] / B,
            n_for_95 = isnothing(n95) ? length(mags) : n95,
            n_for_99 = isnothing(n99) ? length(mags) : n99,
            n_above_1pct = count(x -> x / B > 0.01, sorted),
            budget = B,
            rho1 = rho1,
            dom_share_int = ints[1] / B_int,
            n_for_95_int = isnothing(n95_int) ? length(ints) : n95_int,
            budget_int = B_int);
end

"""
    _theoretical_acf(T, π̄, m, M, maxlag) -> Vector{Float64}

Fitted population |G| ACF from the direct matrix formula
ρ(τ) = (m' diag(π̄) T^τ m − (π̄'m)²) / σ²_|G| for τ = 1..maxlag, with
σ²_|G| = π̄'M − (π̄'m)² (m_k = E[|G| | s=k], M_k = E[G² | s=k]).
"""
function _theoretical_acf(T::AbstractMatrix, π̄::AbstractVector, m::AbstractVector,
                          M::AbstractVector, maxlag::Int)
    σ²_G = π̄' * M - (π̄' * m)^2;
    μ_G  = π̄' * m;
    K = length(m);
    ρ = zeros(maxlag);
    Tτ = Matrix{Float64}(I, K, K);
    for τ in 1:maxlag
        Tτ *= T;
        ρ[τ] = (m' * Diagonal(π̄) * Tτ * m - μ_G^2) / σ²_G;
    end
    return ρ;
end
