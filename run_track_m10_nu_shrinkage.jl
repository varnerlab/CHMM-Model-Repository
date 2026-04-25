# ========================================================================================= #
# run_track_m10_nu_shrinkage.jl
#
# Track M10 (revision response to referee comment M10):
# Penalised ECM for CHMM-t with an exponential shrinkage prior on 1/ν_k, targeting the
# IS excess-kurtosis overshoot of the unpenalised CHMM-t (14.57 simulated vs 7.68 observed).
#
# For each shrinkage rate in a grid, we (i) refit the flat CHMM-t on SPY IS under the
# penalised ECM (see `baum_welch_student_t` with new kwarg `ν_shrink_rate`); (ii) simulate
# N_paths IS return series from the fitted model and record the mean simulated excess
# kurtosis and IS KS pass rate against the observed SPY IS series; (iii) run the
# filter-based one-step-ahead VaR backtest from Track C3b (M3) and report Kupiec /
# Christoffersen LR statistics at α ∈ {0.01, 0.05}.
#
# Outputs:
#   results/track_m10/NU_Shrinkage_Sweep.txt
#   ../CHMM-paper/results/revision/M10_nu_shrinkage.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const N_PATHS   = 1000;

const SHRINK_GRID = [0.0, 5.0, 20.0, 50.0, 100.0, 200.0];

const TRACK_M10_DIR      = joinpath(_ROOT, "results", "track_m10");
const PAPER_REVISION_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "revision"));
mkpath(TRACK_M10_DIR);
mkpath(PAPER_REVISION_DIR);

println("="^72)
println("  Track M10: CHMM-t penalised ECM with 1/ν shrinkage (referee M10)")
println("  Seed $SEED, K=$K_MAIN, rates=$(SHRINK_GRID)")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);

obs_kurt = kurtosis(R_is);  # Distributions.kurtosis returns excess kurtosis
println("  IS $n_is (obs excess kurt $(round(obs_kurt, digits=3))), OoS $n_oos");

# --------------------------------------------------------------------------------------- #
# Forward filter + mixture quantile (copied from run_track_c3_filter_var.jl so this
# script is self-contained; if either changes, update both together).
# --------------------------------------------------------------------------------------- #
function forward_filter(observations::Vector{Float64},
                        model::Union{MyContinuousHiddenMarkovModel,
                                     MyStudentTHiddenMarkovModel,
                                     MyLaplaceHiddenMarkovModel})::Matrix{Float64}
    N = length(observations);
    K = length(model.states);
    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(model.transition[i]);
    end
    log_T = log.(T_mat);
    log_pi = zeros(N, K);
    log_pi[1, :] .= log(1.0 / K) .+ [logpdf(model.emission[k], observations[1]) for k in 1:K];
    log_pi[1, :] .-= _logsumexp_vec(log_pi[1, :]);
    for t in 2:N
        for k in 1:K
            log_pi[t, k] = _logsumexp_vec(log_pi[t-1, :] .+ log_T[:, k]) +
                           logpdf(model.emission[k], observations[t]);
        end
        log_pi[t, :] .-= _logsumexp_vec(log_pi[t, :]);
    end
    return exp.(log_pi);
end

function mixture_quantile(α::Float64, weights::AbstractVector{Float64},
                          components::AbstractVector;
                          tol::Float64=1e-8, max_iter::Int=200)::Float64
    K = length(weights);
    qs = [quantile(components[k], α) for k in 1:K];
    lo = minimum(qs); hi = maximum(qs);
    hi - lo < tol && return 0.5 * (lo + hi);
    mixture_cdf(r) = sum(weights[k] * cdf(components[k], r) for k in 1:K);
    for _ in 1:max_iter
        mid = 0.5 * (lo + hi);
        if mixture_cdf(mid) < α; lo = mid; else; hi = mid; end
        hi - lo < tol && break;
    end
    return 0.5 * (lo + hi);
end

function filter_var_series(model, observations::Vector{Float64}, α::Float64)::Vector{Float64}
    π_mat = forward_filter(observations, model);
    N, K = size(π_mat);
    components = [model.emission[k] for k in 1:K];
    out = Vector{Float64}(undef, N);
    for t in 1:N
        out[t] = mixture_quantile(α, view(π_mat, t, :), components);
    end
    return out;
end

# --------------------------------------------------------------------------------------- #
# Helpers: KS / AD pass rate across simulated paths
# --------------------------------------------------------------------------------------- #
function ks_pass_rate(R_ref::Vector{Float64}, sim::Matrix{Float64}; α::Float64=0.05)
    n_sim = size(sim, 2);
    n_pass = 0;
    for p in 1:n_sim
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        if pv >= α; n_pass += 1; end
    end
    return n_pass / n_sim;
end

# --------------------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------------------- #
results = NamedTuple[];

for rate in SHRINK_GRID
    println("\n[sweep] ν_shrink_rate = $rate");
    Random.seed!(SEED + 8);
    m = build(MyStudentTHiddenMarkovModel,
        (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER,
         ν_shrink_rate=rate));

    ν_vec = [params(m.emission[k].ρ)[1] for k in 1:K_MAIN];  # TDist(ν).ν
    ν_med = median(ν_vec); ν_min = minimum(ν_vec); ν_max = maximum(ν_vec);

    Random.seed!(SEED + 100);
    sim_is = simulate_returns(m, n_is; n_paths=N_PATHS);
    sim_kurt = mean([kurtosis(sim_is[:, p]) for p in 1:N_PATHS]);
    is_ks = ks_pass_rate(R_is, sim_is);

    v01 = filter_var_series(m, R_oos, 0.01);
    v05 = filter_var_series(m, R_oos, 0.05);
    br01 = R_oos .<= v01;
    br05 = R_oos .<= v05;
    k01 = kupiec_lr(br01, 0.01);
    k05 = kupiec_lr(br05, 0.05);
    c01 = christoffersen_lr(br01);
    c05 = christoffersen_lr(br05);

    r = (
        rate=rate,
        ν_min=ν_min, ν_med=ν_med, ν_max=ν_max,
        sim_kurt=sim_kurt, is_ks=is_ks,
        br01=k01.breach_rate, LRuc01=k01.LR, pUC01=k01.pvalue,
        LRind01=c01.LR, pInd01=c01.pvalue,
        br05=k05.breach_rate, LRuc05=k05.LR, pUC05=k05.pvalue,
        LRind05=c05.LR, pInd05=c05.pvalue,
    );
    push!(results, r);

    println("  ν [min, med, max] = [$(round(ν_min, digits=2)), $(round(ν_med, digits=2)), $(round(ν_max, digits=2))]");
    println("  sim IS kurt = $(round(sim_kurt, digits=2)) (obs $(round(obs_kurt, digits=2)))   IS KS pass = $(round(100*is_ks, digits=1))%");
    println("  filter VaR 1%: br $(round(100*k01.breach_rate, digits=2))%  LR_uc $(round(k01.LR, digits=2))  LR_ind $(round(c01.LR, digits=2))");
    println("  filter VaR 5%: br $(round(100*k05.breach_rate, digits=2))%  LR_uc $(round(k05.LR, digits=2))  LR_ind $(round(c05.LR, digits=2))");
end

# --------------------------------------------------------------------------------------- #
# Output: text report
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_M10_DIR, "NU_Shrinkage_Sweep.txt"), "w") do io
    println(io, "="^140);
    println(io, "Track M10. CHMM-t penalised ECM with exponential 1/ν shrinkage. Sweep over shrinkage rate.");
    println(io, "="^140);
    println(io, "");
    println(io, "Penalty   : Q_pen(ν) = Q(ν) - rate / ν, maximised by golden-section search.");
    println(io, "            Rate = 0 recovers the unpenalised ECM of Peel & McLachlan (2000).");
    println(io, "Target    : bring simulated IS excess kurtosis from 14.57 (unpenalised) toward observed $(round(obs_kurt, digits=2)) without degrading IS KS.");
    println(io, "Data      : SPY IS ($n_is obs), OoS ($n_oos obs). N_paths = $N_PATHS.");
    println(io, "Filter VaR: one-step-ahead forward-filter mixture α-quantile, as in Track C3b (referee M3 response).");
    println(io, "");
    println(io, rpad("rate", 7), " | ",
                rpad("ν_med", 6), " | ",
                rpad("ν_min", 6), " | ",
                rpad("ν_max", 6), " | ",
                rpad("sim_kurt", 8), " | ",
                rpad("IS_KS%", 6), " | ",
                rpad("br%01", 6), " | ",
                rpad("LRuc01", 7), " | ",
                rpad("LRind01", 7), " | ",
                rpad("br%05", 6), " | ",
                rpad("LRuc05", 7), " | ",
                rpad("LRind05", 7));
    println(io, "-"^140);
    for r in results
        println(io, rpad(r.rate, 7), " | ",
                    rpad(round(r.ν_med, digits=2), 6), " | ",
                    rpad(round(r.ν_min, digits=2), 6), " | ",
                    rpad(round(r.ν_max, digits=2), 6), " | ",
                    rpad(round(r.sim_kurt, digits=2), 8), " | ",
                    rpad(round(100*r.is_ks, digits=1), 6), " | ",
                    rpad(round(100*r.br01, digits=1), 6), " | ",
                    rpad(round(r.LRuc01, digits=2), 7), " | ",
                    rpad(round(r.LRind01, digits=2), 7), " | ",
                    rpad(round(100*r.br05, digits=1), 6), " | ",
                    rpad(round(r.LRuc05, digits=2), 7), " | ",
                    rpad(round(r.LRind05, digits=2), 7));
    end
    println(io, "="^140);
    println(io, "");
    println(io, "Reading: IS KS% is the fraction of simulated paths passing two-sample KS against observed SPY IS at α=0.05;");
    println(io, "         sim_kurt is the mean across paths of simulated excess kurtosis; target observed $(round(obs_kurt, digits=2)).");
    println(io, "         LR_uc / LR_ind are Kupiec / Christoffersen statistics on the filter-based one-step-ahead breach sequence.");
    println(io, "         χ²(1) 95% critical value is 3.84; lower is better for LR_uc and LR_ind.");
end

# --------------------------------------------------------------------------------------- #
# Output: CSV for paper import
# --------------------------------------------------------------------------------------- #
open(joinpath(PAPER_REVISION_DIR, "M10_nu_shrinkage.csv"), "w") do io
    println(io, "rate,nu_med,nu_min,nu_max,sim_kurt,IS_KS_pct,br01_pct,LRuc01,pUC01,LRind01,pInd01,br05_pct,LRuc05,pUC05,LRind05,pInd05");
    for r in results
        println(io, "$(r.rate),$(round(r.ν_med, digits=3)),$(round(r.ν_min, digits=3)),$(round(r.ν_max, digits=3)),$(round(r.sim_kurt, digits=3)),$(round(100*r.is_ks, digits=2)),$(round(100*r.br01, digits=2)),$(round(r.LRuc01, digits=3)),$(round(r.pUC01, digits=4)),$(round(r.LRind01, digits=3)),$(round(r.pInd01, digits=4)),$(round(100*r.br05, digits=2)),$(round(r.LRuc05, digits=3)),$(round(r.pUC05, digits=4)),$(round(r.LRind05, digits=3)),$(round(r.pInd05, digits=4))");
    end
end

println("\n" * "="^72);
println("  Track M10 complete.");
println("  Text report : $(joinpath(TRACK_M10_DIR, "NU_Shrinkage_Sweep.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_REVISION_DIR, "M10_nu_shrinkage.csv"))");
println("="^72);
