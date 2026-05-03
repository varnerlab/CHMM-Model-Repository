# ========================================================================================= #
# run_walkforward_cond_var_refit_cadence.jl
#
# Reviewer Round 2 / Item B5 (peer-review.md R1#3, P7).
#
# The body walk-forward conditional-VaR table (run_walkforward_conditional_var.jl)
# uses fold-IS-fixed parameters (no refit within the fold). The body sees 5/24
# Christoffersen-cc rejections at α=0.05 concentrated on W2 (COVID) and W4 (2022
# rate-hike). R1#3 asks whether monthly or weekly refit *within* each fold's test
# slice closes 2-3 of those rejections, in which case faster refit is the production
# recipe.
#
# Scoped at K = 3 (body headline), α = 0.05 (the body's substantive diagnostic per
# var_backtest.tex), CHMM-N. Two refit cadences: monthly (21 days) and weekly (5
# days). The fold-IS-fixed baseline (cadence = ∞) is read from the cached
# results/walkforward/walkforward_conditional_var.csv.
#
# Output:
#   results/walkforward_refit_cadence/walkforward_refit_cadence.csv
#   results/walkforward_refit_cadence/walkforward_refit_cadence.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates
using CSV
using DataFrames

const SEED      = 20260420;
const K         = 3;
const ALPHA     = 0.05;
const MAX_ITER  = 60;
const RISK_FREE = 0.0;
const DT        = 1/252;
const TICKER    = "SPY";
const TRAIN_LEN = 1260;          # 5y rolling train window (matches body)
const CADENCES  = [21, 5];       # monthly and weekly (trading days)

const OUT_DIR = joinpath(_ROOT, "results", "walkforward_refit_cadence");
mkpath(OUT_DIR);

# Same fold calendar as run_walkforward_conditional_var.jl
const FOLDS = [
    ("W1", Date(2014,1,1), Date(2019,1,1), Date(2019,1,1), Date(2020,1,1)),
    ("W2", Date(2015,1,1), Date(2020,1,1), Date(2020,1,1), Date(2021,1,1)),
    ("W3", Date(2016,1,1), Date(2021,1,1), Date(2021,1,1), Date(2022,1,1)),
    ("W4", Date(2017,1,1), Date(2022,1,1), Date(2022,1,1), Date(2023,1,1)),
    ("W5", Date(2018,1,1), Date(2023,1,1), Date(2023,1,1), Date(2024,1,1)),
    ("W6", Date(2019,1,1), Date(2024,1,1), Date(2024,1,1), Date(2025,1,1)),
];

println("="^80)
println("  Walk-forward conditional-VaR refit-cadence sweep  (R1#3 / Item B5)")
println("  K = $K  α = $ALPHA  cadences = $CADENCES  folds = $(length(FOLDS))")
println("="^80)

# ----------------------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------------------- #
println("\n[setup] Loading SPY full timeline...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
df_train = train_dataset[TICKER];
df_oos   = oos_dataset[TICKER];
df_full = [df_train; df_oos];
df_full = df_full[sortperm(df_full.timestamp), :];
println("  $(TICKER) full timeline: $(df_full[1, :timestamp]) → $(df_full[end, :timestamp]) ($(nrow(df_full)) rows)")

# Index helper: convert any (start, end_excl) date range to indices into df_full's
# log-return series, where return j corresponds to row (j+1) of df_full (since the
# first return is between rows 1 and 2). We build the full log-return series once.
ts_full = Date.(df_full.timestamp);
P_full  = Vector{Float64}(df_full.volume_weighted_average_price);
R_full  = Vector{Float64}((1/DT) .* (log.(P_full[2:end] ./ P_full[1:end-1])));
ts_full_R = ts_full[2:end];   # the date associated with each return is the row-2 date
n_full = length(R_full);
@printf("  R_full series: %d log returns from %s to %s\n", n_full, ts_full_R[1], ts_full_R[end])

function _idx_range(t_start::Date, t_end_excl::Date)
    mask = (ts_full_R .>= t_start) .& (ts_full_R .< t_end_excl)
    idx_first = findfirst(mask); idx_last = findlast(mask)
    return (idx_first, idx_last)
end

# ----------------------------------------------------------------------------------------- #
# CHMM-N fit + predictive helpers (mirrors body runner for parity)
# ----------------------------------------------------------------------------------------- #
function _fit_chmm_n(R_train::AbstractVector, K::Int)
    Random.seed!(SEED + 7 * K)
    chmm = build(MyContinuousHiddenMarkovModel,
                 (observations=R_train, number_of_states=K, max_iter=MAX_ITER))
    T_mat = zeros(K, K)
    for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
    π̄ = (T_mat^2000)[1, :]; π̄ ./= sum(π̄)
    μ = zeros(K); σ = zeros(K)
    for k in 1:K
        d = chmm.emission[k]
        μ[k] = mean(d); σ[k] = std(d)
    end
    return (T=T_mat, π̄=π̄, μ=μ, σ=σ)
end

function _filter_one_step(prev_pred::AbstractVector, y::Real, fit)
    K = length(fit.μ)
    b = [pdf(Normal(fit.μ[k], fit.σ[k]), y) for k in 1:K]
    post = prev_pred .* b
    Z = sum(post)
    if Z <= 0; post .= prev_pred; else; post ./= Z; end
    return vec(post' * fit.T)
end

function _filter_through(R::AbstractVector, fit)
    K = length(fit.μ)
    pred = zeros(length(R) + 1, K)
    pred[1, :] = fit.π̄
    for t in eachindex(R)
        pred[t + 1, :] = _filter_one_step(pred[t, :], R[t], fit)
    end
    return pred
end

function _mixture_quantile(α::Float64, weights::AbstractVector, μ::AbstractVector,
                            σ::AbstractVector; lo::Float64 = -50.0, hi::Float64 = 50.0,
                            tol::Float64 = 1e-7, max_iter::Int = 80)
    cdf_at(x) = sum(w * cdf(Normal(μk, σk), x) for (w, μk, σk) in zip(weights, μ, σ))
    a = lo; b = hi
    for _ in 1:max_iter
        m = (a + b) / 2
        fm = cdf_at(m) - α
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return (a + b) / 2
end

# ----------------------------------------------------------------------------------------- #
# Refit-cadence pass on a single fold
# ----------------------------------------------------------------------------------------- #
function refit_cadence_pass(fold_id::String,
                            train_start_date::Date, train_end_date::Date,
                            test_start_date::Date,  test_end_date::Date,
                            cadence::Int)
    train_idx = _idx_range(train_start_date, train_end_date)
    test_idx  = _idx_range(test_start_date,  test_end_date)
    train_first, train_last = train_idx
    test_first,  test_last  = test_idx
    n_test = test_last - test_first + 1
    @printf("  [%s] train idx [%d, %d] (T=%d)  test idx [%d, %d] (T=%d)  cadence=%d\n",
            fold_id, train_first, train_last, train_last - train_first + 1,
            test_first, test_last, n_test, cadence)

    var_thr = zeros(n_test)
    breaches = falses(n_test)

    last_refit_offset = -10_000   # j-offset (1-based within test slice) of most recent refit
    fit = nothing
    pred_state = nothing

    for j in 1:n_test
        # decide refit
        need_refit = (j == 1) || ((j - last_refit_offset) >= cadence)
        if need_refit
            # train_end is calendar index test_first + j - 2 (the day BEFORE test[j])
            calendar_train_end = test_first + j - 2
            calendar_train_start = max(1, calendar_train_end - TRAIN_LEN + 1)
            R_train_now = R_full[calendar_train_start:calendar_train_end]
            fit = _fit_chmm_n(R_train_now, K)
            pred_train = _filter_through(R_train_now, fit)
            pred_state = vec(pred_train[end, :])
            last_refit_offset = j
        end

        # predictive at day j
        var_α = _mixture_quantile(ALPHA, pred_state, fit.μ, fit.σ)
        y_j = R_full[test_first + j - 1]
        var_thr[j] = var_α
        breaches[j] = y_j < var_α

        # update for j+1
        pred_state = _filter_one_step(pred_state, y_j, fit)
    end

    return var_thr, breaches, n_test
end

# ----------------------------------------------------------------------------------------- #
# Main loop
# ----------------------------------------------------------------------------------------- #
panels = NamedTuple[]

for (fid, ts, te, vs, ve) in FOLDS
    println("\n" * "="^60)
    println("  Fold $fid  train [$ts, $te)  test [$vs, $ve)")
    println("="^60)
    for cad in CADENCES
        var_thr, br, n_test = refit_cadence_pass(fid, ts, te, vs, ve, cad)
        n_br = sum(br); br_rate = n_br / n_test
        med_var = median(var_thr)
        k_uc = kupiec_lr(br, ALPHA)
        c_in = christoffersen_lr(br)
        c_cc = christoffersen_cc(br, ALPHA)
        @printf("    cadence=%2d  breaches=%3d/%3d (%.2f%%)  med VaR=%7.3f  LR_uc=%5.3f  LR_ind=%5.3f  LR_cc=%5.3f  p_cc=%.3f\n",
                cad, n_br, n_test, 100*br_rate, med_var,
                k_uc.LR, c_in.LR, c_cc.LR, c_cc.pvalue)
        push!(panels, (fold=fid, cadence=cad, K=K, α=ALPHA,
            n_test=n_test, breaches=n_br, breach_rate=br_rate, median_var=med_var,
            LR_uc=k_uc.LR, p_uc=k_uc.pvalue,
            LR_ind=c_in.LR, p_ind=c_in.pvalue,
            LR_cc=c_cc.LR, p_cc=c_cc.pvalue))
    end
end

# ----------------------------------------------------------------------------------------- #
# Pull the fold-IS-fixed baseline rows from the cached CSV
# ----------------------------------------------------------------------------------------- #
println("\n[baseline] Reading cached fold-IS-fixed rows from results/walkforward/walkforward_conditional_var.csv ...")
baseline_csv = joinpath(_ROOT, "results", "walkforward", "walkforward_conditional_var.csv")
baseline_rows = NamedTuple[]
if isfile(baseline_csv)
    bdf = DataFrame(CSV.File(baseline_csv))
    for r in eachrow(bdf)
        if r.K == K && abs(r.alpha - ALPHA) < 1e-6
            push!(baseline_rows, (fold=r.fold, cadence=Int(typemax(Int)), K=K, α=ALPHA,
                n_test=r.T_test, breaches=r.breaches, breach_rate=r.breach_rate, median_var=r.median_var,
                LR_uc=r.LR_uc, p_uc=r.p_uc, LR_ind=r.LR_ind, p_ind=r.p_ind,
                LR_cc=r.LR_cc, p_cc=r.p_cc))
        end
    end
    println("  read $(length(baseline_rows)) fold-IS-fixed rows.")
else
    println("  baseline CSV not found; skipping baseline panel.")
end

# Concatenate baseline rows alongside the new cadence rows.
all_rows = vcat(baseline_rows, panels)

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "walkforward_refit_cadence.csv")
open(csv_path, "w") do io
    println(io, "fold,cadence,K,alpha,n_test,breaches,breach_rate,median_var,LR_uc,p_uc,LR_ind,p_ind,LR_cc,p_cc")
    for r in all_rows
        cad_label = r.cadence > 1000 ? "fold-IS-fixed" : string(r.cadence)
        @printf(io, "%s,%s,%d,%.2f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            r.fold, cad_label, r.K, r.α, r.n_test, r.breaches, r.breach_rate, r.median_var,
            r.LR_uc, r.p_uc, r.LR_ind, r.p_ind, r.LR_cc, r.p_cc)
    end
end

txt_path = joinpath(OUT_DIR, "walkforward_refit_cadence.txt")
open(txt_path, "w") do io
    println(io, "="^120)
    println(io, "Walk-forward conditional-VaR refit-cadence sweep, CHMM-N at K=$K, α=$ALPHA  (R1#3 / Item B5)")
    println(io, "="^120)
    println(io, "Setup: $(length(FOLDS)) folds (train 5y / test 1y); refit at cadences ∈ {monthly=21, weekly=5}")
    println(io, "       trading days. Fold-IS-fixed baseline (cadence = ∞) read from cached")
    println(io, "       results/walkforward/walkforward_conditional_var.csv.")
    println(io, "       At each refit, CHMM-N is re-fit on the most recent $(TRAIN_LEN)-obs window ending the day before")
    println(io, "       the refit-target test day; the predictive filter is re-primed by running through the new train")
    println(io, "       window. The forward filter then continues on the test slice from that posterior.")
    println(io, "Tests: Christoffersen-cc joint conditional coverage; critical χ²₂(0.05) = 5.991 (p_cc > 0.05 = pass).")
    println(io)
    @printf(io, "%-4s | %-13s | %-9s | %-10s | %-9s | %-7s %-6s | %-7s %-6s\n",
        "fold", "cadence", "breaches", "br rate", "med VaR", "LR_uc", "p", "LR_cc", "p_cc")
    println(io, "-"^110)
    for r in all_rows
        cad_label = r.cadence > 1000 ? "fold-IS-fixed" : "$(r.cadence)d"
        @printf(io, "%-4s | %-13s | %-9d | %-10.4f | %-9.3f | %-7.3f %-6.3f | %-7.3f %-6.3f\n",
            r.fold, cad_label, r.breaches, r.breach_rate, r.median_var,
            r.LR_uc, r.p_uc, r.LR_cc, r.p_cc)
    end
    println(io)
    println(io, "Reading.")
    println(io, "  Each row passes Christoffersen-cc at α=0.05 when p_cc > 0.05. The body walk-forward result")
    println(io, "  (fold-IS-fixed) sees rejections concentrated on W2 (COVID) and W4 (2022 rate-hike).")
    println(io, "  Question: do faster refits close 2-3 of those rejections?")
    println(io, "    - If monthly or weekly refit raises p_cc above 0.05 on W2 / W4, faster refit is the")
    println(io, "      production recipe.")
    println(io, "    - If p_cc remains below 0.05 on W2 / W4 across all cadences, the failures are intrinsic")
    println(io, "      regime breaks no refit cadence can adapt to in time.")
end

println("\n[done] $csv_path")
println("[done] $txt_path")
