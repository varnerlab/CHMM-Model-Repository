# ========================================================================================= #
# run_kurtosis_ci_placement.jl
#
# Reviewer Round 2 / Item B3 (peer-review.md R2#req-2).
#
# The body penalised CHMM-t at λ = 20 reports aggregate simulated IS kurtosis 8.56 (K=18)
# and 14.91 (K*=3) against observed 7.68. The L=20 stationary block bootstrap CI on the
# observed IS kurtosis is [2.17, 12.40] (Appendix sec:kurtosis_bootstrap_ci, Table
# tab:kurtosis_bootstrap). R2's concern: at K*=3 the body penalised CHMM-t aggregate
# sim kurt 14.91 is *above* the bootstrap CI upper bound of 12.40 — i.e. the headline
# heavy-tail row is statistically distinguishable from the observed bootstrap envelope
# at the 5% level.
#
# This runner records the per-path simulated IS excess kurtosis distribution for the
# penalised CHMM-t at λ = 20 at three operating points (K* = 3, K* = 6, K = 18) and
# reports:
#   (a) the fraction of simulated paths whose IS kurtosis falls inside the L=20
#       bootstrap CI [2.17, 12.40];
#   (b) the fraction below the upper bound 12.40 (the relevant one-sided test for
#       overshoot);
#   (c) per-path mean / median / sd / [Q5, Q95].
#
# Output:
#   results/kurtosis_ci_placement/kurtosis_ci_placement.csv
#   results/kurtosis_ci_placement/kurtosis_ci_placement.txt
#   results/kurtosis_ci_placement/kurtosis_per_path_K{k}.csv  (per-path kurt)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
using StatsBase
using Printf

const SEED      = 20260420;
const N_PATHS   = 1000;
const MAX_ITER  = 60;
const DT        = 1/252;
const RISK_FREE = 0.0;
const SHRINK_RATE = 20.0;

# L=20 bootstrap CI on observed IS excess kurtosis (Table tab:kurtosis_bootstrap)
const CI_LO = 2.167;
const CI_HI = 12.400;

Random.seed!(SEED);

const OUT_DIR = joinpath(_ROOT, "results", "kurtosis_ci_placement");
mkpath(OUT_DIR);

println("="^80)
println("  Bootstrap-CI placement of penalised CHMM-t IS kurtosis  (R2 / Item B3)")
println("  Seed: $SEED  Paths: $N_PATHS  λ = $SHRINK_RATE")
println("  IS bootstrap CI [$(CI_LO), $(CI_HI)]")
println("="^80)

# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY IS...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);
μ_obs = mean(R_is); σ_obs = std(R_is);
kurt_obs = sum(((R_is .- μ_obs) ./ σ_obs).^4) / n_is - 3.0;
println("  IS = $n_is  obs excess kurt = $(round(kurt_obs, digits=3))")

# ----------------------------------------------------------------------------------------- #
function _stationary(model, K::Int)
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :];
    return Categorical(π);
end

function _sim_paths(model, sd, n::Int, np::Int)
    sim = Matrix{Float64}(undef, n, np);
    for p in 1:np
        s0 = rand(sd);
        st = model(s0, n);
        for j in 1:n; sim[j, p] = rand(model.emission[st[j]]); end
    end
    return sim;
end

function per_path_excess_kurt(sim::AbstractMatrix)
    np = size(sim, 2);
    n = size(sim, 1);
    out = zeros(np);
    for i in 1:np
        s = view(sim, :, i);
        μ = mean(s); σ = std(s);
        out[i] = sum(((s .- μ) ./ σ).^4) / n - 3.0;
    end
    return out;
end

# ----------------------------------------------------------------------------------------- #
results = NamedTuple[]

for K in [3, 6, 18]
    println("\n[fit] CHMM-t penalised λ=$SHRINK_RATE on SPY IS, K = $K ...")
    chmm_t_pen = build(MyStudentTHiddenMarkovModel,
        (observations=R_is, number_of_states=K, max_iter=MAX_ITER,
         ν_shrink_rate=SHRINK_RATE));
    sd_t = _stationary(chmm_t_pen, K);
    println("[sim] $N_PATHS paths length $n_is ...")
    sim_is = _sim_paths(chmm_t_pen, sd_t, n_is, N_PATHS);

    kurt_pp = per_path_excess_kurt(sim_is);

    n_in_ci   = count(k -> CI_LO <= k <= CI_HI, kurt_pp);
    n_below   = count(<(CI_LO), kurt_pp);
    n_above   = count(>(CI_HI), kurt_pp);
    frac_in   = round(100*n_in_ci/N_PATHS, digits=1);
    frac_below_hi = round(100*(n_in_ci+n_below)/N_PATHS, digits=1);

    push!(results, (
        K=K,
        agg_kurt = round(mean(kurt_pp), digits=3),
        median   = round(median(kurt_pp), digits=3),
        sd       = round(std(kurt_pp), digits=3),
        q05      = round(quantile(kurt_pp, 0.05), digits=3),
        q95      = round(quantile(kurt_pp, 0.95), digits=3),
        n_in_ci  = n_in_ci,
        n_below  = n_below,
        n_above  = n_above,
        frac_in  = frac_in,
        frac_below_hi = frac_below_hi,
    ))

    # save per-path kurt
    pp_path = joinpath(OUT_DIR, "kurtosis_per_path_K$(K).csv")
    open(pp_path, "w") do io
        println(io, "path_idx,excess_kurt")
        for (i, k) in enumerate(kurt_pp)
            @printf(io, "%d,%.6f\n", i, k)
        end
    end

    @printf("  K=%d : agg %.2f  med %.2f  [Q5,Q95]=[%.2f,%.2f]  in_CI=%d/%d (%.1f%%)  below_HI=%.1f%%\n",
            K, results[end].agg_kurt, results[end].median, results[end].q05, results[end].q95,
            n_in_ci, N_PATHS, frac_in, frac_below_hi)
end

# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "kurtosis_ci_placement.csv")
open(csv_path, "w") do io
    println(io, "K,agg_kurt,median,sd,q05,q95,n_in_ci,n_below_lo,n_above_hi,pct_in_ci,pct_below_hi")
    for r in results
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%.1f,%.1f\n",
                r.K, r.agg_kurt, r.median, r.sd, r.q05, r.q95,
                r.n_in_ci, r.n_below, r.n_above, r.frac_in, r.frac_below_hi)
    end
end

txt_path = joinpath(OUT_DIR, "kurtosis_ci_placement.txt")
open(txt_path, "w") do io
    println(io, "="^110)
    println(io, "Bootstrap-CI placement of penalised CHMM-t IS kurtosis  (Reviewer Round 2 / Item B3)")
    println(io, "="^110)
    println(io)
    @printf(io, "Setup     : SPY IS, T = %d, paths = %d, seed = %d, penalty rate λ = %.1f\n", n_is, N_PATHS, SEED, SHRINK_RATE)
    @printf(io, "CI source : Stationary block bootstrap of Politis-Romano 1994 at L=20, B=5000 (Appendix sec:kurtosis_bootstrap_ci)\n")
    @printf(io, "CI used   : [%.3f, %.3f]   (observed IS excess kurt = %.3f)\n", CI_LO, CI_HI, kurt_obs)
    println(io)
    @printf(io, "  %-3s  %-9s  %-7s  %-6s  %-7s  %-7s  %-9s  %-12s  %-12s\n",
            "K", "agg_kurt", "median", "sd", "Q5", "Q95", "in_CI", "≤ CI_HI(%)", "in_CI(%)")
    println(io, "  ", "-"^100)
    for r in results
        @printf(io, "  %-3d  %-9.3f  %-7.3f  %-6.3f  %-7.3f  %-7.3f  %d/%d   %12.1f   %12.1f\n",
                r.K, r.agg_kurt, r.median, r.sd, r.q05, r.q95,
                r.n_in_ci, N_PATHS, r.frac_below_hi, r.frac_in)
    end
    println(io)
    println(io, "Reading.")
    println(io, "  At each operating point, 'in_CI(%)' is the fraction of simulated IS paths whose")
    println(io, "  per-path excess kurtosis falls inside the L=20 bootstrap CI on observed [2.17, 12.40].")
    println(io, "  '≤ CI_HI(%)' is the fraction below the upper bound (the relevant one-sided test for")
    println(io, "  overshoot, since at this λ the constraint is not undershoot).")
    println(io)
    println(io, "  R2 concern: at K*=3 the aggregate sim kurt 14.91 sits ABOVE the upper bound 12.40, so")
    println(io, "  the headline heavy-tail row could be statistically distinguishable from observed at α=0.05")
    println(io, "  on a one-sided over-shoot test. The per-path 'in_CI(%)' and '≤ CI_HI(%)' columns above")
    println(io, "  quantify this directly: if 'in_CI(%)' is well below 50%, the penalised CHMM-t IS kurtosis")
    println(io, "  distribution is not centred inside the bootstrap envelope and the headline framing should")
    println(io, "  be revised. If 'in_CI(%)' is above 50% even with aggregate kurt above the upper bound,")
    println(io, "  the result is a heavy-right-tail-of-paths effect with most of the mass inside the CI.")
end

println("\n[done] $csv_path")
println("[done] $txt_path")
