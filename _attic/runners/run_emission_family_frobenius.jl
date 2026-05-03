# ========================================================================================= #
# run_emission_family_frobenius.jl
#
# Closes peer-review item R2 W4 / R2 RE3: report the per-state location, scale, and
# transition-matrix Frobenius distances between the four emission families (CHMM-N,
# CHMM-t penalised at lambda = 20, CHMM-L, CHMM-GED) on the SPY headline and on the
# 30-ticker sector-balanced panel. The reviewer's framing: if these distances are below
# the per-state Monte Carlo standard errors, the four-family narrative collapses to a
# one-parameter shape axis. We fit each family independently per ticker, canonicalise
# states by ascending sigma, and report pairwise Frobenius distances.
#
# All fits at K*=3 (the body headline) so the comparison is between families at the
# same state resolution. Multi-seed averaging (3 seeds per family) gives a rough
# Monte Carlo SE for context.
#
# Output:
#   results/emission_family_frobenius/{spy.txt, panel.txt, summary.csv}
#   ../CHMM-paper/results/robustness/emission_family_frobenius.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Printf
const SEED       = 20260420;
const TICKER_SPY = "SPY";
const RISK_FREE  = 0.0;
const DT         = 1/252;
const MAX_ITER   = 60;
const K_MAIN     = 3;
const LAMBDA     = 20.0;
const N_SEEDS    = 3;             # refit each family 3 times for MC SE

const OUT_DIR = joinpath(_ROOT, "results", "emission_family_frobenius");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

const SECTOR_PANEL = [
    ("Information Technology",   ["AAPL", "MSFT", "NVDA"]),
    ("Health Care",              ["JNJ",  "UNH",  "LLY"]),
    ("Financials",               ["JPM",  "BAC",  "WFC"]),
    ("Consumer Discretionary",   ["AMZN", "HD",   "MCD"]),
    ("Communication Services",   ["NFLX", "VZ",   "DIS"]),
    ("Industrials",              ["CAT",  "BA",   "HON"]),
    ("Consumer Staples",         ["PG",   "KO",   "WMT"]),
    ("Energy",                   ["XOM",  "CVX",  "COP"]),
    ("Utilities",                ["NEE",  "DUK",  "SO" ]),
    ("Materials",                ["FCX",  "NEM",  "APD"]),
];

println("="^80);
println("  Per-state Frobenius distances across four emission families");
println("  R2 W4 / R2 RE3, K* = $K_MAIN, $N_SEEDS seeds per family");
println("="^80);

# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY closes + 30-ticker panel...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String, DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

# --------------------------------------------------------------------------------------- #
function _fit(family::Symbol, R::Vector{Float64}, K::Int, seed::Int)
    Random.seed!(seed);
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=R, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :student_t_pen
        return build(MyStudentTHiddenMarkovModel,
            (observations=R, number_of_states=K, max_iter=MAX_ITER, ν_shrink_rate=LAMBDA));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=R, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :ged
        return build(MyGEDHiddenMarkovModel,
            (observations=R, number_of_states=K, max_iter=MAX_ITER));
    end
end

# Extract canonical (μ, σ, T) under ascending-σ canonicalisation
function _extract_canonical(model, K::Int)
    μ = Vector{Float64}(undef, K);
    σ = Vector{Float64}(undef, K);
    for s in 1:K
        em = model.emission[s];
        if isa(em, Distributions.Normal)
            μ[s] = em.μ; σ[s] = em.σ;
        elseif isa(em, Distributions.Laplace)
            μ[s] = em.μ; σ[s] = em.θ;
        else
            μ[s] = mean(em); σ[s] = sqrt(var(em));
        end
    end
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    order = sortperm(σ);
    μ_c = μ[order]; σ_c = σ[order];
    T_c = zeros(K, K);
    for i in 1:K, j in 1:K; T_c[i, j] = T[order[i], order[j]]; end
    return μ_c, σ_c, T_c;
end

# --------------------------------------------------------------------------------------- #
const FAMILIES = [(:gaussian, "N"), (:student_t_pen, "t"), (:laplace, "L"), (:ged, "GED")];

function _fit_all_seeds(R::Vector{Float64}, K::Int, base_seed::Int)
    by_family = Dict{Symbol, NamedTuple}();
    for (fam, _) in FAMILIES
        μs = Vector{Vector{Float64}}();
        σs = Vector{Vector{Float64}}();
        Ts = Vector{Matrix{Float64}}();
        for s in 1:N_SEEDS
            m = _fit(fam, R, K, base_seed + 100*s + Int(hash(fam) >> 32));
            μ, σ, T = _extract_canonical(m, K);
            push!(μs, μ); push!(σs, σ); push!(Ts, T);
        end
        # mean-of-seed estimate + per-seed SD (rough Monte Carlo SE)
        μ_mean = mean(hcat(μs...); dims=2)[:];
        σ_mean = mean(hcat(σs...); dims=2)[:];
        T_mean = mean(cat(Ts...; dims=3); dims=3)[:, :, 1];
        μ_sd   = N_SEEDS > 1 ? std(hcat(μs...); dims=2)[:] : zeros(K);
        σ_sd   = N_SEEDS > 1 ? std(hcat(σs...); dims=2)[:] : zeros(K);
        T_sd   = N_SEEDS > 1 ? std(cat(Ts...; dims=3); dims=3)[:, :, 1] : zeros(K, K);
        by_family[fam] = (μ=μ_mean, σ=σ_mean, T=T_mean, μ_sd=μ_sd, σ_sd=σ_sd, T_sd=T_sd);
    end
    return by_family;
end

function _frobenius(a::AbstractArray, b::AbstractArray)
    return sqrt(sum((a .- b) .^ 2));
end

# --------------------------------------------------------------------------------------- #
println("\n[SPY] fitting all 4 families x $N_SEEDS seeds at K = $K_MAIN ...");
idx_spy = findfirst(==(TICKER_SPY), all_tickers);
R_spy = Vector{Float64}(all_R[:, idx_spy]);
spy_fams = _fit_all_seeds(R_spy, K_MAIN, SEED);

println("\n[SPY] mean parameter estimates (canonical sigma-ascending order):");
for (fam, label) in FAMILIES
    f = spy_fams[fam];
    @printf("  CHMM-%-3s  μ = [%s]   σ = [%s]   diag T = [%s]\n",
            label,
            join([@sprintf("%+.4f", x) for x in f.μ], ", "),
            join([@sprintf("%.4f",  x) for x in f.σ], ", "),
            join([@sprintf("%.3f",  f.T[k, k]) for k in 1:K_MAIN], ", "));
end

println("\n[SPY] pairwise Frobenius distances between families:");
spy_pairs = Vector{NamedTuple}();
for i in eachindex(FAMILIES), j in (i+1):lastindex(FAMILIES)
    fi, li = FAMILIES[i]; fj, lj = FAMILIES[j];
    d_μ = _frobenius(spy_fams[fi].μ, spy_fams[fj].μ);
    d_σ = _frobenius(spy_fams[fi].σ, spy_fams[fj].σ);
    d_T = _frobenius(spy_fams[fi].T, spy_fams[fj].T);
    push!(spy_pairs, (label="$li-$lj", d_μ=d_μ, d_σ=d_σ, d_T=d_T));
    @printf("  CHMM-%s vs CHMM-%-3s  ‖Δμ‖_F = %.5f   ‖Δσ‖_F = %.5f   ‖ΔT‖_F = %.5f\n",
            li, lj, d_μ, d_σ, d_T);
end
# Also Monte Carlo SE on each family
spy_se = Dict{Symbol, NamedTuple}();
for (fam, label) in FAMILIES
    f = spy_fams[fam];
    spy_se[fam] = (
        μ_se=_frobenius(f.μ_sd, zeros(K_MAIN)),
        σ_se=_frobenius(f.σ_sd, zeros(K_MAIN)),
        T_se=_frobenius(f.T_sd, zeros(K_MAIN, K_MAIN)),
    );
    @printf("  CHMM-%-3s self MC SE  ‖σ_μ‖_F = %.5f   ‖σ_σ‖_F = %.5f   ‖σ_T‖_F = %.5f\n",
            label, spy_se[fam].μ_se, spy_se[fam].σ_se, spy_se[fam].T_se);
end

# --------------------------------------------------------------------------------------- #
println("\n[panel] fitting all 4 families across 30 tickers...");
panel_rows = Vector{NamedTuple}();
let ticker_idx = 0
    for (sector, tickers) in SECTOR_PANEL
        for tk in tickers
            ti = findfirst(==(tk), all_tickers);
            if isnothing(ti); continue; end
            ticker_idx += 1;
            R_t = Vector{Float64}(all_R[:, ti]);
            fams = _fit_all_seeds(R_t, K_MAIN, SEED + 1000 * ticker_idx);
            for i in eachindex(FAMILIES), j in (i+1):lastindex(FAMILIES)
                fi, li = FAMILIES[i]; fj, lj = FAMILIES[j];
                d_μ = _frobenius(fams[fi].μ, fams[fj].μ);
                d_σ = _frobenius(fams[fi].σ, fams[fj].σ);
                d_T = _frobenius(fams[fi].T, fams[fj].T);
                push!(panel_rows, (sector=sector, ticker=tk, pair="$li-$lj",
                                   d_μ=d_μ, d_σ=d_σ, d_T=d_T));
            end
            @printf("  [%2d] %-5s  done\n", ticker_idx, tk);
        end
    end
end

# --------------------------------------------------------------------------------------- #
println("\n[aggregate] panel-wide distribution by pair...");
pair_rows = unique([r.pair for r in panel_rows]);
agg = Vector{NamedTuple}();
for p in pair_rows
    sub = filter(r -> r.pair == p, panel_rows);
    push!(agg, (pair=p,
        d_μ_med=median([r.d_μ for r in sub]), d_μ_q1=quantile([r.d_μ for r in sub], 0.25),
            d_μ_q3=quantile([r.d_μ for r in sub], 0.75),
        d_σ_med=median([r.d_σ for r in sub]), d_σ_q1=quantile([r.d_σ for r in sub], 0.25),
            d_σ_q3=quantile([r.d_σ for r in sub], 0.75),
        d_T_med=median([r.d_T for r in sub]), d_T_q1=quantile([r.d_T for r in sub], 0.25),
            d_T_q3=quantile([r.d_T for r in sub], 0.75)));
    @printf("  %s  ‖Δμ‖ med = %.4f [%.4f, %.4f]  ‖Δσ‖ med = %.4f [%.4f, %.4f]  ‖ΔT‖ med = %.4f [%.4f, %.4f]\n",
            p,
            agg[end].d_μ_med, agg[end].d_μ_q1, agg[end].d_μ_q3,
            agg[end].d_σ_med, agg[end].d_σ_q1, agg[end].d_σ_q3,
            agg[end].d_T_med, agg[end].d_T_q1, agg[end].d_T_q3);
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
spy_path = joinpath(OUT_DIR, "spy.txt");
open(spy_path, "w") do io
    println(io, "="^110);
    println(io, "Per-state Frobenius distances between four emission families on SPY (R2 W4)");
    println(io, "K = $K_MAIN, λ = $LAMBDA, $N_SEEDS seeds per family");
    println(io, "="^110);
    println(io);
    println(io, "Mean parameter estimates (canonical sigma-ascending order, mean over $N_SEEDS seeds):");
    for (fam, label) in FAMILIES
        f = spy_fams[fam];
        @printf(io, "  CHMM-%-3s  μ = [%s]   σ = [%s]   diag T = [%s]\n",
                label,
                join([@sprintf("%+.4f", x) for x in f.μ], ", "),
                join([@sprintf("%.4f",  x) for x in f.σ], ", "),
                join([@sprintf("%.3f",  f.T[k, k]) for k in 1:K_MAIN], ", "));
    end
    println(io);
    println(io, "Pairwise Frobenius distances:");
    @printf(io, "  %-7s  %-12s  %-12s  %-12s\n", "pair", "‖Δμ‖_F", "‖Δσ‖_F", "‖ΔT‖_F");
    for r in spy_pairs
        @printf(io, "  %-7s  %-12.5f  %-12.5f  %-12.5f\n", r.label, r.d_μ, r.d_σ, r.d_T);
    end
    println(io);
    println(io, "Per-family Monte Carlo SE (Frobenius norm of per-coord SD across seeds):");
    @printf(io, "  %-9s  %-12s  %-12s  %-12s\n", "family", "‖σ_μ‖_F", "‖σ_σ‖_F", "‖σ_T‖_F");
    for (fam, label) in FAMILIES
        s = spy_se[fam];
        @printf(io, "  CHMM-%-3s  %-12.5f  %-12.5f  %-12.5f\n",
                label, s.μ_se, s.σ_se, s.T_se);
    end
end

panel_path = joinpath(OUT_DIR, "panel.txt");
open(panel_path, "w") do io
    println(io, "="^110);
    println(io, "Per-state Frobenius distances on the 30-ticker panel (R2 RE3)");
    println(io, "K = $K_MAIN, λ = $LAMBDA, $N_SEEDS seeds per family per ticker");
    println(io, "="^110);
    println(io);
    println(io, "Aggregate distribution across 30 tickers:");
    @printf(io, "  %-7s  %-22s  %-22s  %-22s\n", "pair",
            "‖Δμ‖_F med [Q1, Q3]", "‖Δσ‖_F med [Q1, Q3]", "‖ΔT‖_F med [Q1, Q3]");
    for r in agg
        @printf(io, "  %-7s  %.4f [%.4f, %.4f]  %.4f [%.4f, %.4f]  %.4f [%.4f, %.4f]\n",
                r.pair,
                r.d_μ_med, r.d_μ_q1, r.d_μ_q3,
                r.d_σ_med, r.d_σ_q1, r.d_σ_q3,
                r.d_T_med, r.d_T_q1, r.d_T_q3);
    end
    println(io);
    println(io, "Per-ticker per-pair distances (full panel):");
    @printf(io, "  %-26s  %-7s  %-7s  %-12s  %-12s  %-12s\n",
            "Sector", "Ticker", "Pair", "‖Δμ‖_F", "‖Δσ‖_F", "‖ΔT‖_F");
    for r in panel_rows
        @printf(io, "  %-26s  %-7s  %-7s  %-12.5f  %-12.5f  %-12.5f\n",
                r.sector, r.ticker, r.pair, r.d_μ, r.d_σ, r.d_T);
    end
end

# Combined CSV
csv_path = joinpath(OUT_DIR, "summary.csv");
open(csv_path, "w") do io
    println(io, "scope,sector,ticker,pair,d_mu_F,d_sigma_F,d_T_F");
    for r in spy_pairs
        @printf(io, "spy,,SPY,%s,%.6f,%.6f,%.6f\n", r.label, r.d_μ, r.d_σ, r.d_T);
    end
    for r in panel_rows
        @printf(io, "panel,%s,%s,%s,%.6f,%.6f,%.6f\n",
                r.sector, r.ticker, r.pair, r.d_μ, r.d_σ, r.d_T);
    end
end
cp(csv_path, joinpath(PAPER_ROBUSTNESS_DIR, "emission_family_frobenius.csv"); force=true);

println("\n" * "="^80);
println("  Per-state Frobenius distances complete.");
@printf("  SPY      : %s\n", spy_path);
@printf("  Panel    : %s\n", panel_path);
@printf("  CSV      : %s\n", csv_path);
println("="^80);
