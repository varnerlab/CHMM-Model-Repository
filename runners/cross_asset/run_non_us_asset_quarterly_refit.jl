# =========================================================================== #
# run_non_us_asset_quarterly_refit.jl
#
# Review-response Item 11: GLD periodic-refit follow-up. The static-IS-fit GLD
# OoS KS in run_non_us_asset.jl collapses to 0%; the body conclusion claims
# "the static IS-fitted CHMM does not transfer outside this asset class." This
# runner tests whether quarterly refit on a 5y rolling window recovers any of
# that gap.
#
# Recipe: mirror run_sector_panel_quarterly_refit.jl but on the single GLD
# series. For each OoS quarter, refit CHMM-N (or -t / -L per CLI) on the
# preceding 5y rolling window, simulate the quarter under the refitted
# parameters, and concatenate the per-quarter paths. Score KS, kurtosis, and
# |G_t| ACF-MAE on the stitched OoS path.
#
# Output: results/non_us_asset/Non_US_Asset_QuarterlyRefit.{txt,csv}
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, HypothesisTests, StatsBase, Printf

const SEED          = 20260422;
const K_MAIN        = parse(Int, get(ENV, "GLD_REFIT_K", "3"));
const N_PATHS       = 200;
const MAX_ITER      = 60;
const DT            = 1/252;
const RISK_FREE     = 0.0;
const REFIT_CADENCE = 63;        # ~quarterly trading days
const TRAIN_LEN     = 1260;       # 5y rolling estimation window
const L_LAGS        = 252;
const ALPHA_KS      = 0.05;
const FAMILY        = get(ENV, "GLD_REFIT_FAMILY", "N");  # "N", "t", "L"

const OUT_DIR  = joinpath(_ROOT, "results", "non_us_asset");
mkpath(OUT_DIR);
const OUT_TXT  = joinpath(OUT_DIR, "Non_US_Asset_QuarterlyRefit.txt");
const OUT_CSV  = joinpath(OUT_DIR, "Non_US_Asset_QuarterlyRefit.csv");

println("="^88)
println("  Item 11: GLD quarterly-refit Pipeline-A check")
println("  K = $K_MAIN, family = CHMM-$FAMILY, refit cadence = $REFIT_CADENCE days,")
println("  rolling train = $TRAIN_LEN days, $N_PATHS paths, seed = $SEED")
println("="^88)

# -------- data --------
train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet()["dataset"];

# Trim training set to common max-day rows (mirrors run_non_us_asset.jl)
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
@assert haskey(dataset, "GLD")     "GLD missing from train dataset"
@assert haskey(oos_dataset, "GLD") "GLD missing from OoS dataset"

R_is  = Vector{Float64}(log_growth_matrix(dataset, "GLD";  Δt=DT, risk_free_rate=RISK_FREE));
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "GLD"; Δt=DT, risk_free_rate=RISK_FREE));
n_is = length(R_is); n_oos = length(R_oos);
println("[setup] IS = $n_is, OoS = $n_oos days for GLD")

# -------- family-dispatched CHMM build --------
function _build_chmm(obs::AbstractVector, K::Int)
    if FAMILY == "N"
        return build(MyContinuousHiddenMarkovModel,
            (observations=Vector{Float64}(obs), number_of_states=K, max_iter=MAX_ITER));
    elseif FAMILY == "t"
        return build(MyStudentTHiddenMarkovModel,
            (observations=Vector{Float64}(obs), number_of_states=K, max_iter=MAX_ITER));
    elseif FAMILY == "L"
        return build(MyLaplaceHiddenMarkovModel,
            (observations=Vector{Float64}(obs), number_of_states=K, max_iter=MAX_ITER));
    else
        error("Unknown CHMM family: $FAMILY")
    end
end

function _stationary_start_dist(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return Categorical(π_stat);
end

# -------- quarterly-refit OoS paths --------
# For each quarter q starting at OoS day 1, 64, 127, ...:
#   train_window = last TRAIN_LEN observations of (R_is ++ R_oos[1:q-1])
#   fit CHMM on train_window
#   simulate REFIT_CADENCE days starting from stationary distribution
#   concatenate
function _quarterly_refit_paths()
    sim_paths = zeros(n_oos, N_PATHS);
    full = vcat(R_is, R_oos);
    quarter_starts = collect(1:REFIT_CADENCE:n_oos);
    println("[refit] $(length(quarter_starts)) quarterly refits scheduled")
    for (qi, q_start) in enumerate(quarter_starts)
        q_end = min(q_start + REFIT_CADENCE - 1, n_oos);
        q_len = q_end - q_start + 1;
        # train window: last TRAIN_LEN observations strictly before q_start in OoS
        win_end_global = n_is + q_start - 1;
        win_start_global = max(1, win_end_global - TRAIN_LEN + 1);
        train_win = full[win_start_global:win_end_global];
        Random.seed!(SEED + 1000 * qi);
        model = _build_chmm(train_win, K_MAIN);
        K_actual = length(model.states);
        sd = _stationary_start_dist(model, K_actual);
        for p in 1:N_PATHS
            s0 = rand(sd);
            st = model(s0, q_len);
            for j in 1:q_len
                sim_paths[q_start + j - 1, p] = rand(model.emission[st[j]]);
            end
        end
        @printf("  quarter %d/%d  days %d-%d  K_actual=%d\n",
            qi, length(quarter_starts), q_start, q_end, K_actual);
    end
    return sim_paths;
end

println("\n[run] Building quarterly-refit OoS paths...")
sim_oos = _quarterly_refit_paths();

# -------- score the refitted OoS paths --------
function _ks_pass_rate(R_obs::AbstractVector, R_sim::AbstractMatrix; α::Float64=ALPHA_KS)
    np = size(R_sim, 2);
    pass = 0;
    for i in 1:np
        ks = HypothesisTests.ApproximateTwoSampleKSTest(R_obs, R_sim[:, i]);
        if pvalue(ks) >= α; pass += 1; end
    end
    return 100.0 * pass / np;
end

function _mean_excess_kurt(R_sim::AbstractMatrix)
    np = size(R_sim, 2);
    s = 0.0;
    for i in 1:np; s += kurtosis(R_sim[:, i]); end
    return s / np;
end

function _acf_mae(R_obs::AbstractVector, R_sim::AbstractMatrix; L::Int=L_LAGS)
    obs_acf = autocor(abs.(R_obs), 1:L);
    np = size(R_sim, 2);
    sim_acf_mean = zeros(L);
    for i in 1:np
        sim_acf_mean .+= autocor(abs.(R_sim[:, i]), 1:L);
    end
    sim_acf_mean ./= np;
    return mean(abs.(sim_acf_mean .- obs_acf));
end

oos_ks    = _ks_pass_rate(R_oos, sim_oos);
oos_kurt  = _mean_excess_kurt(sim_oos);
oos_acf   = _acf_mae(R_oos, sim_oos);
obs_kurt  = kurtosis(R_oos);

println("\n" * "="^88)
println("  GLD quarterly-refit OoS results (CHMM-$FAMILY, K = $K_MAIN)")
println("="^88)
println(@sprintf("  OoS KS pass rate     : %.2f %%", oos_ks))
println(@sprintf("  OoS sim kurtosis     : %.3f  (observed %.3f)", oos_kurt, obs_kurt))
println(@sprintf("  OoS |G_t| ACF-MAE    : %.4f", oos_acf))
println("  Compare against the static-IS-fit baseline reported in")
println("  results/non_us_asset/Non_US_Asset.txt (typically 0% OoS KS).")

open(OUT_TXT, "w") do io
    println(io, "Item 11: GLD quarterly-refit Pipeline-A back-test")
    println(io, "  CHMM-$FAMILY, K = $K_MAIN, refit cadence = $REFIT_CADENCE days,")
    println(io, "  rolling train = $TRAIN_LEN days, $N_PATHS paths, seed = $SEED")
    println(io, "")
    println(io, @sprintf("  OoS KS pass rate     : %.2f %%", oos_ks))
    println(io, @sprintf("  OoS sim kurtosis     : %.3f  (observed %.3f)", oos_kurt, obs_kurt))
    println(io, @sprintf("  OoS |G_t| ACF-MAE    : %.4f", oos_acf))
    println(io, "")
    println(io, "Compare against Non_US_Asset.txt for the static-IS-fit row")
    println(io, "(static OoS KS typically 0%, motivating the periodic-refit recipe).")
end

open(OUT_CSV, "w") do io
    write(io, "metric,value\n")
    write(io, @sprintf("oos_ks_pct,%.4f\n",      oos_ks))
    write(io, @sprintf("oos_sim_kurtosis,%.4f\n", oos_kurt))
    write(io, @sprintf("obs_oos_kurtosis,%.4f\n", obs_kurt))
    write(io, @sprintf("oos_acf_mae,%.6f\n",     oos_acf))
end

println("\n[done] Output: $OUT_TXT, $OUT_CSV")
