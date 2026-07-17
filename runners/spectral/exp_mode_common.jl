# ========================================================================================= #
# exp_mode_common.jl — unrestricted exponential-mode diagnostic core (include-able).
#
# Mixed dictionary of exponential mode shapes spanning every K-state HMM ACF component
# shape: positive real (cost 1), negative real (cost 1), damped-oscillatory conjugate
# pair (cost 2). Budget-aware greedy matching pursuit with exact LLS refit and local
# lambda-refinement sweeps. Heuristic fit — no global certificate; see
# run_exp_mode_diagnostic.jl for scope and run_hmm_acf_capacity.jl for the realizable
# (valid-HMM) attainability experiment.
# ========================================================================================= #

using LinearAlgebra, Statistics, Printf

const LAM_MAX   = 0.9995;
const THETAS    = [pi/12, pi/6, pi/4, pi/3, pi/2, 2pi/3, 5pi/6];
const N_SWEEPS  = 5;

_lambda_grid(n) = clamp.(exp.(log(0.5) ./ exp.(range(log(0.4), log(600.0), length=n))), 1e-6, LAM_MAX);

# A component is (kind, λ, θ) with kind ∈ (:pos, :neg, :osc); cost 1, 1, 2 modes.
_comp_cost(kind) = kind == :osc ? 2 : 1;

function _comp_basis(kind::Symbol, λ::Float64, θ::Float64, maxlag::Int)
    if kind == :pos
        return reshape([λ^τ for τ in 1:maxlag], maxlag, 1);
    elseif kind == :neg
        return reshape([(-λ)^τ for τ in 1:maxlag], maxlag, 1);
    else
        return hcat([λ^τ * cos(θ*τ) for τ in 1:maxlag],
                    [λ^τ * sin(θ*τ) for τ in 1:maxlag]);
    end
end

function _build_dictionary(n_lam::Int, maxlag::Int)
    lams = _lambda_grid(n_lam);
    comps = Tuple{Symbol,Float64,Float64}[];
    for λ in lams
        push!(comps, (:pos, λ, 0.0));
        push!(comps, (:neg, λ, 0.0));
        for θ in THETAS
            push!(comps, (:osc, λ, θ));
        end
    end
    bases = [_comp_basis(k, λ, θ, maxlag) for (k, λ, θ) in comps];
    return comps, bases;
end

# Exact LLS on the concatenated bases of the selected components.
function _lls_multi(ρ::Vector{Float64}, sel::Vector{Tuple{Symbol,Float64,Float64}}, maxlag::Int)
    G = hcat([_comp_basis(k, λ, θ, maxlag) for (k, λ, θ) in sel]...);
    a = G \ ρ;
    return a, ρ .- G * a;
end

# Projection score of residual r on a candidate basis (SSE explained per mode).
function _proj_score(r::Vector{Float64}, B::Matrix{Float64}, cost::Int)
    c = B \ r;
    return sum(abs2, B * c) / cost;
end

"""
Budget-aware greedy matching pursuit over the mixed dictionary: select components
(cost-weighted) up to m modes, exact LLS refit after each selection, then N_SWEEPS
local λ-refinement sweeps (multiplicative half-life factors; kind and θ kept fixed).
Returns (selected components, coefficients aligned with the concatenated bases,
fitted curve).
"""
function _fit_m_modes(ρ::Vector{Float64}, m::Int, comps, bases, maxlag::Int)
    sel = Tuple{Symbol,Float64,Float64}[];
    used = 0;
    resid = copy(ρ);
    while used < m
        best_i = 0; best_score = -Inf;
        for (i, c) in enumerate(comps)
            cost = _comp_cost(c[1]);
            cost > m - used && continue;
            c in sel && continue;
            sc = _proj_score(resid, bases[i], cost);
            if sc > best_score; best_score = sc; best_i = i; end
        end
        best_i == 0 && break;
        push!(sel, comps[best_i]);
        used += _comp_cost(comps[best_i][1]);
        _, resid = _lls_multi(ρ, sel, maxlag);
    end
    # local λ refinement (kind/θ fixed): multiplicative half-life perturbations
    factors = [0.90, 0.95, 0.98, 1.02, 1.05, 1.10];
    _, r0 = _lls_multi(ρ, sel, maxlag);
    sse = sum(abs2, r0);
    for _ in 1:N_SWEEPS
        for i in 1:length(sel)
            (k, λ, θ) = sel[i];
            h_i = log(0.5) / log(λ);
            for f in factors
                cand = copy(sel);
                cand[i] = (k, clamp(exp(log(0.5) / (h_i * f)), 1e-6, LAM_MAX), θ);
                _, r = _lls_multi(ρ, cand, maxlag);
                s = sum(abs2, r);
                if s < sse - 1e-14; sse = s; sel = cand; end
            end
        end
    end
    a, resid = _lls_multi(ρ, sel, maxlag);
    return sel, a, ρ .- resid;
end

function _sample_acf_abs(x::AbstractVector, maxlag::Int)
    a = abs.(x); n = length(a); μ = mean(a);
    v = sum((a .- μ).^2);
    return [sum((a[1:n-τ] .- μ) .* (a[1+τ:n] .- μ)) / v for τ in 1:maxlag];
end

function _band_maes(fit::Vector{Float64}, ρ::Vector{Float64})
    return mean(abs.(fit[1:63] .- ρ[1:63])), mean(abs.(fit[64:end] .- ρ[64:end]));
end

# Serialize a selection: "kind:lambda:theta:coeffs" segments joined by "|".
function _comp_string(sel, a)
    parts = String[];
    j = 1;
    for (k, λ, θ) in sel
        if k == :osc
            push!(parts, @sprintf("osc:%.4f:%.4f:%.5f:%.5f", λ, θ, a[j], a[j+1]));
            j += 2;
        else
            push!(parts, @sprintf("%s:%.4f:0:%.5f", String(k), λ, a[j]));
            j += 1;
        end
    end
    return join(parts, "|");
end

