# =========================================================================== #
# run_tail_index_families.jl
#
# Hill tail-index (alpha) for EVERY CHMM emission variant at K*=3, so we can say
# per family which are genuinely heavy-tailed (finite power-law tail index, in
# Cont's (2,5) band) vs merely leptokurtic (exponential-type tail, alpha -> inf).
#
# Theory: only the Student-t family has a genuine power-law tail (asymptotic
# alpha = nu). Gaussian, Laplace, and GED have exponential-type tails, so their
# asymptotic tail index is infinite; any tail fatness is a finite-sample mixture
# effect (leptokurtic, not heavy-tailed). This runner measures alpha_hat directly
# with the same Hill estimator applied to every family and to observed SPY.
#
# Fit + simulate protocol matches run_kstar3_headline.jl (seed SEED for the fit,
# SEED+1 for the 1000-path simulation). The shared-nu CHMM-t row reuses the saved
# results/chmm_t_shared_nu/sims_K3.jld2 (no refit; keeps continuity with the
# committed run_tail_index_hill.jl number, alpha_hat ~ 3.67 at top 5%).
#
# Output: results/diagnostics/tail_index_families.{txt,csv}
# Usage:  julia --project=. runners/robustness/run_tail_index_families.jl
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, StatsBase, Printf, Distributions, JLD2;

const SEED       = 20260420;
const RISK_FREE  = 0.0;
const ΔT         = 1/252;
const MAX_ITER   = 60;
const N_PATHS    = 1000;
const K_STAR     = 3;
const LAMBDA_T   = 20.0;
const TAIL_FRACS = [0.02, 0.05, 0.10];
const OUT_DIR    = joinpath(_ROOT, "results", "diagnostics");
mkpath(OUT_DIR);

# --- Hill upper-tail index (Paper I port). Returns xi = 1/alpha. --------------
function hill_xi(x::AbstractVector{<:Real}; tail_frac::Float64 = 0.05)
    @assert 0.0 < tail_frac < 1.0
    sx = sort(x; rev = true)
    k  = max(2, floor(Int, tail_frac * length(sx)))
    @assert sx[k] > 0.0
    s = 0.0
    for i in 1:(k - 1); s += log(sx[i] / sx[k]); end
    return s / (k - 1)
end
_exkurt(x) = sum(((x .- mean(x)) ./ std(x)).^4) / length(x) - 3.0;

# One-sided Hill: left tail (loss magnitudes) and right tail (gains), with the SAME
# exceedance count k = tail_frac * n on each side so the two indices are comparable.
# Separating them accounts for Cont's gain/loss asymmetry (the loss tail is usually
# heavier, i.e. a lower alpha, than the gain tail).
function _lr_xi(g::AbstractVector{<:Real}; tail_frac::Float64 = 0.05)
    n = length(g); k = max(2, floor(Int, tail_frac * n))
    Lv = sort(.-(filter(<(0.0), g)); rev = true)   # loss magnitudes, largest first
    Rv = sort(filter(>(0.0), g);     rev = true)    # gains, largest first
    _xi(v) = (kk = min(k, length(v)); s = 0.0; for i in 1:(kk-1); s += log(v[i]/v[kk]); end; s/(kk-1))
    return (_xi(Lv), _xi(Rv))
end

# --- Fit / simulate helpers (verbatim from run_kstar3_headline.jl) ------------
function _train_headline(family::Symbol, obs::Vector{Float64}, K::Int, max_iter::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t_pen
        return build(MyStudentTHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter, ν_shrink_rate=LAMBDA_T));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :ged
        return build(MyGEDHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    else
        error("Unknown emission family: $family")
    end
end

function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    return Categorical((T_mat^1000)[1, :]);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos::Int, n_paths::Int)
    sim_is  = Array{Float64,2}(undef, n_is,  n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is;  sim_is[j,i]  = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

# Per-path Hill on a (T x paths) matrix: mean alpha_hat + across-path 95% band.
function _sim_alpha(sim::AbstractMatrix; tail_frac = 0.05)
    xis = [hill_xi(abs.(@view sim[:, p]); tail_frac = tail_frac) for p in 1:size(sim, 2)]
    return (a = 1/mean(xis), lo = 1/quantile(xis, 0.975), hi = 1/quantile(xis, 0.025))
end
function _sim_kurt(sim::AbstractMatrix)
    return mean(_exkurt(@view sim[:, p]) for p in 1:size(sim, 2))
end

# Per-path left/right Hill: average xi over paths, then invert to alpha.
function _sim_lr(sim::AbstractMatrix; tail_frac = 0.05)
    xls = Float64[]; xrs = Float64[]
    for p in 1:size(sim, 2)
        xl, xr = _lr_xi(sim[:, p]; tail_frac = tail_frac)
        push!(xls, xl); push!(xrs, xr)
    end
    return (aL = 1/mean(xls), aR = 1/mean(xrs))
end

# --- Load SPY IS / OoS -------------------------------------------------------
println("[load] SPY IS / OoS (VWAP excess log growth) ...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt = ΔT, risk_free_rate = RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is  = all_R[:, idx_spy];
R_oos = log_growth_matrix(MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"], "SPY"; Δt = ΔT, risk_free_rate = RISK_FREE);
n_is = length(R_is); n_oos = length(R_oos);
println("  IS = $n_is  OoS = $n_oos")

# --- Assemble rows: label, IS/OoS sim matrices, sim kurtosis -----------------
rows = NamedTuple[];

# Observed (reference).
push!(rows, (label = "Observed SPY", is = nothing, oos = nothing,
             kurt_is = round(_exkurt(R_is), digits=2), kurt_oos = round(_exkurt(R_oos), digits=2),
             obs = true));

# Shared-nu CHMM-t from saved sims (no refit).
sims_path = joinpath(_ROOT, "results", "chmm_t_shared_nu", "sims_K3.jld2");
if isfile(sims_path)
    sims = JLD2.load(sims_path);
    push!(rows, (label = "CHMM-t (shared-nu)", is = sims["is"], oos = sims["oos"],
                 kurt_is = round(_sim_kurt(sims["is"]), digits=2),
                 kurt_oos = round(_sim_kurt(sims["oos"]), digits=2), obs = false));
else
    @warn "sims_K3.jld2 missing; skipping shared-nu row (run run_chmm_t_shared_nu.jl first)";
end

# Fit + simulate the four headline families at K*=3.
FAMILIES = [(:gaussian, "CHMM-N"), (:student_t_pen, "CHMM-t (pen, lambda=20)"),
            (:laplace, "CHMM-L"), (:ged, "CHMM-GED")];
for (fam, label) in FAMILIES
    println("[fit] $label (K=$K_STAR) ...")
    Random.seed!(SEED);
    model = _train_headline(fam, R_is, K_STAR, MAX_ITER);
    sd = _stationary(model, K_STAR);
    Random.seed!(SEED + 1);
    sim_is, sim_oos = _simulate_paths(model, sd, n_is, n_oos, N_PATHS);
    push!(rows, (label = label, is = sim_is, oos = sim_oos,
                 kurt_is = round(_sim_kurt(sim_is), digits=2),
                 kurt_oos = round(_sim_kurt(sim_oos), digits=2), obs = false));
end

# --- Emit --------------------------------------------------------------------
function _alpha_cells(row, tf)
    if row.obs
        return (1/hill_xi(abs.(R_is); tail_frac=tf), 1/hill_xi(abs.(R_oos); tail_frac=tf), NaN, NaN)
    else
        ai = _sim_alpha(row.is; tail_frac=tf); ao = _sim_alpha(row.oos; tail_frac=tf)
        return (ai.a, ao.a, ai.lo, ai.hi)
    end
end

function _emit(io)
    println(io, "="^100)
    println(io, "Hill tail index (alpha) by CHMM variant at K*=3  vs  observed SPY")
    println(io, "="^100)
    println(io)
    println(io, "Estimator: upper-tail Hill on |G_t| (Paper I port), same estimator for every series.")
    println(io, "alpha_hat = 1/xi_hat; for a Pareto tail alpha_hat -> tail index. Seed $SEED, $N_PATHS paths.")
    println(io, "Cont (2001): daily-equity tail index alpha in (2,5), often 3-4. Student-t emission: alpha = nu.")
    println(io, "Gaussian / Laplace / GED emissions: exponential-type tail, asymptotic alpha = infinity.")
    println(io)
    @printf(io, "%-24s %8s | %-14s | %-14s | %-14s | %-22s\n",
            "Variant", "ExKurt", "alpha (top 2%)", "alpha (top 5%)", "alpha (top 10%)", "alpha 5% IS 95% band")
    @printf(io, "%-24s %8s | %-14s | %-14s | %-14s | %-22s\n",
            "", "IS/OoS", "IS / OoS", "IS / OoS", "IS / OoS", "[lo, hi]")
    println(io, "-"^100)
    for r in rows
        a2 = _alpha_cells(r, 0.02); a5 = _alpha_cells(r, 0.05); a10 = _alpha_cells(r, 0.10)
        band = r.obs ? "     (observed)     " : @sprintf("[%.2f, %.2f]", a5[3], a5[4])
        @printf(io, "%-24s %3.2f/%-4.2f | %4.2f / %-4.2f  | %4.2f / %-4.2f  | %4.2f / %-4.2f  | %-22s\n",
                r.label, r.kurt_is, r.kurt_oos,
                a2[1], a2[2], a5[1], a5[2], a10[1], a10[2], band)
    end
    println(io)
    println(io, "-"^100)
    println(io, "Gain/loss asymmetry (Cont SF): Hill alpha for the LEFT tail (losses) vs RIGHT tail (gains),")
    println(io, "top 5%, same exceedance count each side. Lower alpha = heavier tail. Observed loss tail is")
    println(io, "heavier than the gain tail; symmetric emissions can only get asymmetry from the regime means.")
    println(io)
    @printf(io, "%-24s | %-18s | %-18s | %-10s\n",
            "Variant", "LEFT alpha (loss)", "RIGHT alpha (gain)", "L - R (IS)")
    @printf(io, "%-24s | %-18s | %-18s | %-10s\n", "", "IS / OoS", "IS / OoS", "asymmetry")
    println(io, "-"^100)
    for r in rows
        if r.obs
            lL_is, rR_is = _lr_xi(R_is);  lL_oos, rR_oos = _lr_xi(R_oos)
            aLi, aRi, aLo, aRo = 1/lL_is, 1/rR_is, 1/lL_oos, 1/rR_oos
        else
            li = _sim_lr(r.is); lo = _sim_lr(r.oos)
            aLi, aRi, aLo, aRo = li.aL, li.aR, lo.aL, lo.aR
        end
        @printf(io, "%-24s | %4.2f / %-4.2f       | %4.2f / %-4.2f       | %+5.2f\n",
                r.label, aLi, aLo, aRi, aRo, aLi - aRi)
    end
    println(io)
    println(io, "Reading. In-band (2,5) and stable across fractions => genuine power-law heavy tail.")
    println(io, "High alpha and/or rising with depth => thin (exponential) tail: leptokurtic only.")
    println(io, "L - R < 0 => loss tail heavier than gain tail (the observed asymmetry); ~0 => symmetric.")
end

_emit(stdout)
open(joinpath(OUT_DIR, "tail_index_families.txt"), "w") do io; _emit(io); end
open(joinpath(OUT_DIR, "tail_index_families.csv"), "w") do io
    println(io, "variant,exkurt_is,exkurt_oos,tail_frac,alpha_is,alpha_oos,alpha_is_lo,alpha_is_hi")
    for r in rows, tf in TAIL_FRACS
        a = _alpha_cells(r, tf)
        @printf(io, "\"%s\",%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f\n",
                r.label, r.kurt_is, r.kurt_oos, tf, a[1], a[2], a[3], a[4])
    end
end
println("\n[done] $(joinpath(OUT_DIR, "tail_index_families.txt"))")
