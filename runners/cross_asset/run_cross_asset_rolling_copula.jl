# ========================================================================================= #
# run_cross_asset_rolling_copula.jl
#
# Paired static-versus-rolling refit comparison of the Pipeline-B Student-t copula on the
# six-asset SPY universe. Third-review item 2 rebuild: the previous version of this runner
# compared a full-window static MAE (one 572-day simulation against the single full-OoS
# correlation matrix) with the mean of quarter-specific rolling MAEs (63-day simulations
# against quarter-specific realised matrices), used K = 18 marginals although the paper's
# headline cross-asset table uses K* = 3, and silently dropped the last 5 OoS observations.
# Those two arms score different estimands, so their difference was not identified.
#
# Design (paired, like-for-like):
# - Per-asset CHMM-N marginals at K* = 3 (the Table tab:cross_asset headline state count),
#   fitted once on IS and held fixed in BOTH arms. Only the copula differs between arms.
# - The OoS span is partitioned into quarters of STEP = 63 trading days; the final partial
#   block (5 days at T_OoS = 572) is included as a short quarter, so every OoS observation
#   is scored. Quarter q's target is its realised correlation matrix.
# - STATIC arm: the copula is fit once on the IS window; for each quarter we simulate a
#   panel of the quarter's length and score it against that quarter's target.
# - ROLLING arm: for each quarter the copula (Sigma, nu*) is refit on the trailing
#   WINDOW = 252 days ending the session before the quarter starts (accruing OoS data as
#   the window slides); the simulated panel is scored against the SAME quarter target.
# - Both arms therefore face the identical sequence of targets, horizons, and marginals;
#   the per-quarter simulations share a per-quarter seed (common-random-number pairing at
#   the seed level; the nu-mixing draws still differ where the fitted nu* differs).
# - Inference: paired per-quarter loss differences d_q = MAE_static,q - MAE_rolling,q,
#   with a paired t interval and sign counts across quarters. Uncertainty is across
#   quarters (time blocks), not Monte-Carlo paths.
# - A full-window static MAE (the old Table tab:cross_asset anchor) is reported once,
#   descriptively, so the paper can reconcile against the main-text static number; it
#   takes no part in the paired comparison.
#
# Outputs:
#   results/cross_asset/Rolling_Copula_OoS.txt
#   ../CHMM-Paper-Repository/results/robustness/rolling_copula_oos.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random
using Statistics
using LinearAlgebra
using Dates
using Distributions

const SEED      = 20260422;
const RISK_FREE = 0.0;
const DT        = 1/252;
const K         = 3;          # paper headline K* = 3 (Table tab:cross_asset marginals)
const MAX_ITER  = 60;
const N_PATHS   = 200;
const ASSETS    = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const WINDOW    = 252;        # rolling-window length (1 trading year)
const STEP      = 63;         # quarterly refit cadence (1 trading quarter)

const OUT_DIR             = joinpath(_ROOT, "results", "cross_asset");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-Paper-Repository", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72);
println("  Paired static-vs-rolling cross-asset Student-t copula comparison");
println("  K = $K marginals  Window $WINDOW d  Step $STEP d  Seed $SEED");
println("="^72);

Random.seed!(SEED);

# --------------------------------------------------------------------------------------- #
# Data: IS + OoS for the six-asset universe; concatenate into a single time series for the
# rolling-window machinery (the boundary t = n_is corresponds to 2024-01-04).
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading IS + OoS six-asset returns...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end

R_is_list  = [log_growth_matrix(filtered, t; Δt=DT, risk_free_rate=RISK_FREE) for t in ASSETS];
R_oos_list = [log_growth_matrix(oos_dataset, t; Δt=DT, risk_free_rate=RISK_FREE) for t in ASSETS];
n_is  = minimum(length.(R_is_list));
n_oos = minimum(length.(R_oos_list));
R_is  = hcat([r[1:n_is]  for r in R_is_list]...);
R_oos = hcat([r[1:n_oos] for r in R_oos_list]...);
R_full = vcat(R_is, R_oos);
n_full = size(R_full, 1);
d = length(ASSETS);
println("  IS $n_is days,  OoS $n_oos days,  combined $n_full days,  d = $d assets");

# --------------------------------------------------------------------------------------- #
# Per-asset CHMM-N marginals at K* = 3 (held fixed in both arms)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Per-asset CHMM-N marginals at K = $K on IS (held fixed in both arms)...");
chmms = Vector{AbstractMarkovModel}(undef, d);
for j in 1:d
    Random.seed!(SEED + j);
    chmms[j] = build(MyContinuousHiddenMarkovModel,
        (observations=R_is[:, j], number_of_states=K, max_iter=MAX_ITER));
    println("  $(ASSETS[j]) done.");
end

function _offdiag_mae(A::Matrix{Float64}, B::Matrix{Float64})
    @assert size(A) == size(B);
    d = size(A, 1);
    mask = .!I(d);
    return mean(abs.((A .- B)[mask]));
end

# --------------------------------------------------------------------------------------- #
# Quarter partition of the OoS span: complete 63-day blocks plus the final partial block,
# so every OoS observation is scored (stated convention: the last block is short).
# --------------------------------------------------------------------------------------- #
quarter_bounds = Tuple{Int,Int}[];
let q_start = n_is + 1
    while q_start <= n_full
        push!(quarter_bounds, (q_start, min(q_start + STEP - 1, n_full)));
        q_start += STEP;
    end
end
n_q = length(quarter_bounds);
println("\n[quarters] $n_q OoS blocks (final block $(quarter_bounds[end][2] - quarter_bounds[end][1] + 1) days)");

# --------------------------------------------------------------------------------------- #
# Static arm copula: fit once on IS.
# --------------------------------------------------------------------------------------- #
println("\n[static] Fitting the static copula on the IS window...");
Random.seed!(SEED);
copula_static = build(MyStudentTCopulaModel,
    (returns = R_is, tickers = ASSETS, marginals = chmms));
println("  static nu* = $(copula_static.nu)");

# Descriptive full-window anchor (NOT part of the paired comparison): one n_oos-day panel
# against the single full-OoS correlation matrix, the老 Table tab:cross_asset construction.
Random.seed!(SEED);
sim_full = simulate(copula_static, n_oos, N_PATHS);
cor_full = correlation_reproduction(R_oos, sim_full);
println("  [anchor] static full-window OoS off-diag MAE = $(round(cor_full.offdiag_mae, digits=4))");

# --------------------------------------------------------------------------------------- #
# Paired per-quarter comparison: identical targets, horizons, marginals, and per-quarter
# seeds in both arms; only the copula differs.
# --------------------------------------------------------------------------------------- #
rows = NamedTuple[];
for (qi, (q_start, q_end)) in enumerate(quarter_bounds)
    n_days = q_end - q_start + 1;
    R_quarter = R_full[q_start:q_end, :];

    win_end = q_start - 1;
    win_start = max(1, win_end - WINDOW + 1);
    R_window = R_full[win_start:win_end, :];

    Random.seed!(SEED + 1000 + qi);
    copula_q = build(MyStudentTCopulaModel,
        (returns = R_window, tickers = ASSETS, marginals = chmms));

    Random.seed!(SEED + 2000 + qi);
    sim_s = simulate(copula_static, n_days, N_PATHS);
    cor_s = correlation_reproduction(R_quarter, sim_s);

    Random.seed!(SEED + 2000 + qi);
    sim_r = simulate(copula_q, n_days, N_PATHS);
    cor_r = correlation_reproduction(R_quarter, sim_r);

    push!(rows, (
        qi = qi,
        q_start = q_start, q_end = q_end, n_days = n_days,
        win_start = win_start, win_end = win_end,
        nu_static = copula_static.nu, nu_roll = copula_q.nu,
        mae_static = cor_s.offdiag_mae, mae_roll = cor_r.offdiag_mae,
        diff = cor_s.offdiag_mae - cor_r.offdiag_mae,
    ));
    println("  q=$qi  [$(q_start),$(q_end)] ($(n_days)d)  static=$(round(cor_s.offdiag_mae, digits=4))  rolling=$(round(cor_r.offdiag_mae, digits=4))  diff=$(round(cor_s.offdiag_mae - cor_r.offdiag_mae, digits=4))  ν*_roll=$(copula_q.nu)");
end

# Paired inference across quarters (time-block uncertainty, not Monte-Carlo path noise).
diffs = [r.diff for r in rows];
d_mean = mean(diffs);
d_sd   = std(diffs);
d_se   = d_sd / sqrt(n_q);
tdist  = TDist(n_q - 1);
t_stat = d_mean / d_se;
p_two  = 2 * (1 - cdf(tdist, abs(t_stat)));
t_crit = quantile(tdist, 0.975);
ci_lo, ci_hi = d_mean - t_crit * d_se, d_mean + t_crit * d_se;
n_pos = count(>(0), diffs);

mae_static_mean = mean([r.mae_static for r in rows]);
mae_roll_mean   = mean([r.mae_roll   for r in rows]);

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "Rolling_Copula_OoS.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "Paired static-vs-rolling Student-t copula comparison, six-asset SPY cross-section (third-review item 2)");
    println(io, "="^110);
    println(io, "");
    println(io, "Setup:");
    println(io, "  - Universe: $(join(ASSETS, ", "))");
    println(io, "  - Per-asset CHMM-N marginals at K = $K (paper headline K*), fitted once on IS, held fixed in BOTH arms.");
    println(io, "  - Quarter partition: $(n_q) OoS blocks of $STEP days; the final $(rows[end].n_days)-day partial block is included.");
    println(io, "  - STATIC arm: copula fit once on IS; per-quarter simulation scored on that quarter's realised matrix.");
    println(io, "  - ROLLING arm: copula refit each quarter on the trailing $WINDOW days; scored on the SAME quarter target.");
    println(io, "  - Paths per simulation: $N_PATHS.  Per-quarter seeds shared across arms (seed-level CRN).");
    println(io, "  - Seed root: $SEED");
    println(io, "");
    println(io, "Headline (paired across $(n_q) quarters):");
    println(io, "  Mean per-quarter off-diag MAE, static copula   : $(round(mae_static_mean, digits=4))");
    println(io, "  Mean per-quarter off-diag MAE, rolling refit   : $(round(mae_roll_mean, digits=4))");
    println(io, "  Mean paired difference (static - rolling)      : $(round(d_mean, digits=4))  (sd $(round(d_sd, digits=4)), se $(round(d_se, digits=4)))");
    println(io, "  Paired t ($(n_q - 1) df)                        : t = $(round(t_stat, digits=2)),  two-sided p = $(round(p_two, digits=4))");
    println(io, "  95% t-interval for the difference              : [$(round(ci_lo, digits=4)), $(round(ci_hi, digits=4))]");
    println(io, "  Quarters favouring the rolling refit           : $n_pos / $n_q");
    println(io, "");
    println(io, "Descriptive anchor (not part of the paired comparison):");
    println(io, "  Static full-window OoS off-diag MAE (one $(n_oos)-day panel vs the single full-OoS matrix,");
    println(io, "  the main-text Table construction): $(round(cor_full.offdiag_mae, digits=4))   nu* = $(copula_static.nu)");
    println(io, "");
    println(io, "Per-quarter detail:");
    println(io, "");
    println(io, rpad("q", 3), " | ", rpad("forecast", 14), " | ", rpad("days", 5), " | ",
                rpad("win", 14), " | ", rpad("ν*roll", 7), " | ",
                rpad("MAE static", 11), " | ", rpad("MAE rolling", 12), " | ", rpad("diff", 8));
    println(io, "-"^96);
    for r in rows
        println(io, rpad(r.qi, 3), " | ",
                    rpad("[$(r.q_start),$(r.q_end)]", 14), " | ",
                    rpad(r.n_days, 5), " | ",
                    rpad("[$(r.win_start),$(r.win_end)]", 14), " | ",
                    rpad(string(r.nu_roll), 7), " | ",
                    rpad(round(r.mae_static, digits=4), 11), " | ",
                    rpad(round(r.mae_roll, digits=4), 12), " | ",
                    rpad(round(r.diff, digits=4), 8));
    end
    println(io, "");
    println(io, "Notes:");
    println(io, "  - Both arms score the identical per-quarter targets with identical marginals and horizons, so the");
    println(io, "    paired difference isolates the copula refit; quarter-to-quarter target difficulty cancels in d_q.");
    println(io, "  - The CHMM-N marginals are not refit in either arm; only the dependence layer moves.");
    println(io, "  - Uncertainty is across quarters (paired t over $(n_q) blocks). Monte-Carlo path noise is secondary");
    println(io, "    and partially cancelled by the shared per-quarter seeds.");
    println(io, "  - The rolling refit can move ν* across the discrete grid {2,3,4,5,6,8,10,15,20,30} per quarter.");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "rolling_copula_oos.csv"), "w") do io
    println(io, "qi,q_start,q_end,n_days,win_start,win_end,nu_static,nu_roll,mae_static,mae_roll,diff");
    for r in rows
        println(io, "$(r.qi),$(r.q_start),$(r.q_end),$(r.n_days),$(r.win_start),$(r.win_end),$(r.nu_static),$(r.nu_roll),$(round(r.mae_static, digits=5)),$(round(r.mae_roll, digits=5)),$(round(r.diff, digits=5))");
    end
    println(io, "summary,mean,,,,,$(copula_static.nu),,$(round(mae_static_mean, digits=5)),$(round(mae_roll_mean, digits=5)),$(round(d_mean, digits=5))");
    println(io, "summary,paired_t,,,,,,,t=$(round(t_stat, digits=3)),p=$(round(p_two, digits=4)),ci=[$(round(ci_lo, digits=5));$(round(ci_hi, digits=5))]");
    println(io, "summary,anchor_full_window,$(n_is + 1),$(n_full),$(n_oos),,,$(copula_static.nu),$(round(cor_full.offdiag_mae, digits=5)),,");
end

println("\n" * "="^72);
println("Headline (paired across $(n_q) quarters):");
println("  static mean $(round(mae_static_mean, digits=4))  rolling mean $(round(mae_roll_mean, digits=4))  diff $(round(d_mean, digits=4))  t=$(round(t_stat, digits=2)) p=$(round(p_two, digits=4))");
println("  full-window static anchor: $(round(cor_full.offdiag_mae, digits=4))");
println("  Outputs:");
println("    $out_path");
println("    $(joinpath(PAPER_ROBUSTNESS_DIR, "rolling_copula_oos.csv"))");
println("="^72);
