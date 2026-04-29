# ========================================================================================= #
# run_cross_asset_rolling_copula.jl
#
# Rolling-window refit of the Pipeline-B Student-t copula on the six-asset SPY universe.
# Addresses peer-review item R1-Q4 / R1-RE4: the static-IS-fit OoS off-diagonal MAE of
# 0.209 in Table tab:cross_asset is much larger than the IS value of 0.027. R1 asked
# whether a quarterly rolling refit reduces the OoS gap.
#
# Setup:
# - Per-asset CHMM-N marginals are held fixed at the IS fits (the same fits used for
#   the headline cross-asset Table 3 in the paper).
# - The Student-t copula correlation matrix Σ and degrees-of-freedom ν* are refit on a
#   rolling 252-day window (1 trading year), sliding by 63 days (1 quarter).
# - For each refit window ending at t in the OoS span, we simulate forward 63 days under
#   the new copula and compare the path-averaged simulated correlation matrix to the
#   realised next-quarter correlation matrix.
# - The metric is the off-diagonal MAE between simulated and realised correlations,
#   averaged across rolling sub-windows.
#
# Outputs:
#   results/cross_asset/Rolling_Copula_OoS.txt
#   ../CHMM-paper/results/robustness/rolling_copula_oos.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using LinearAlgebra
using Dates

const SEED      = 20260422;
const RISK_FREE = 0.0;
const DT        = 1/252;
const K         = 18;
const MAX_ITER  = 60;
const N_PATHS   = 200;
const ASSETS    = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const WINDOW    = 252;        # rolling-window length (1 trading year)
const STEP      = 63;         # quarterly refit cadence (1 trading quarter)

const OUT_DIR             = joinpath(_ROOT, "results", "cross_asset");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72);
println("  Rolling-window cross-asset Student-t copula refit  (R1-Q4 / R1-RE4)");
println("  Window $WINDOW d  Step $STEP d  Seed $SEED");
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
# Per-asset CHMM-N marginals at K = 18 (held fixed across rolling refits)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Per-asset CHMM-N marginals at K = $K on IS (held fixed for rolling refits)...");
chmms = Vector{AbstractMarkovModel}(undef, d);
for j in 1:d
    Random.seed!(SEED + j);
    chmms[j] = build(MyContinuousHiddenMarkovModel,
        (observations=R_is[:, j], number_of_states=K, max_iter=MAX_ITER));
    println("  $(ASSETS[j]) done.");
end

# --------------------------------------------------------------------------------------- #
# Rolling refit + per-quarter forecast
# --------------------------------------------------------------------------------------- #
function _offdiag_mae(A::Matrix{Float64}, B::Matrix{Float64})
    @assert size(A) == size(B);
    d = size(A, 1);
    mask = .!I(d);
    return mean(abs.((A .- B)[mask]));
end

# Static-IS-fit baseline: refit on the IS window only, simulate, compare to OoS realised.
println("\n[baseline] Static IS-fit copula -> OoS off-diag MAE...");
Random.seed!(SEED);
copula_static = build(MyStudentTCopulaModel,
    (returns = R_is, tickers = ASSETS, marginals = chmms));
sim_static = simulate(copula_static, n_oos, N_PATHS);
cor_static = correlation_reproduction(R_oos, sim_static);
println("  Static-IS  OoS off-diag MAE = $(round(cor_static.offdiag_mae, digits=4))   ν* = $(copula_static.nu)");

# Rolling refits across OoS quarters. The window ends at t in the OoS span and contains the
# preceding 252 days (mostly OoS by mid-OoS). The forecast horizon is the next 63 OoS days.
oos_quarter_starts = collect((n_is + 1):STEP:(n_full - STEP));
println("\n[rolling] Refit cadence quarterly (every $STEP days), $(length(oos_quarter_starts)) sub-windows in OoS");

rows = NamedTuple[];
for (qi, q_start) in enumerate(oos_quarter_starts)
    q_end = min(q_start + STEP - 1, n_full);
    win_end = q_start - 1;
    win_start = max(1, win_end - WINDOW + 1);

    R_window = R_full[win_start:win_end, :];
    R_quarter = R_full[q_start:q_end, :];

    Random.seed!(SEED + 1000 + qi);
    copula_q = build(MyStudentTCopulaModel,
        (returns = R_window, tickers = ASSETS, marginals = chmms));
    sim_q = simulate(copula_q, q_end - q_start + 1, N_PATHS);
    cor_q = correlation_reproduction(R_quarter, sim_q);

    push!(rows, (
        qi = qi,
        win_start = win_start, win_end = win_end,
        q_start = q_start, q_end = q_end,
        nu_star = copula_q.nu,
        offdiag_mae = cor_q.offdiag_mae,
        frob = cor_q.frob_mean,
    ));
    println("  q=$qi  win=[$(win_start),$(win_end)] -> [$(q_start),$(q_end)]  ν* = $(copula_q.nu)  off-diag MAE = $(round(cor_q.offdiag_mae, digits=4))");
end

mae_rolling_mean = mean([r.offdiag_mae for r in rows]);
mae_rolling_med  = median([r.offdiag_mae for r in rows]);

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "Rolling_Copula_OoS.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "Rolling-window Student-t copula refit on six-asset SPY cross-section  (R1-Q4 / R1-RE4)");
    println(io, "="^110);
    println(io, "");
    println(io, "Setup:");
    println(io, "  - Universe: $(join(ASSETS, ", "))");
    println(io, "  - Per-asset CHMM-N marginals at K = $K, fitted once on IS, held fixed for rolling refits.");
    println(io, "  - Rolling window length: $WINDOW days (1 trading year).");
    println(io, "  - Refit cadence:        $STEP days  (1 trading quarter).");
    println(io, "  - Forecast horizon:     $STEP days per quarter.");
    println(io, "  - Paths per refit:      $N_PATHS");
    println(io, "  - Seed root:            $SEED");
    println(io, "");
    println(io, "Headline:");
    println(io, "  Static-IS-fit OoS off-diag MAE   : $(round(cor_static.offdiag_mae, digits=4))");
    println(io, "  Rolling-refit OoS off-diag MAE   : mean $(round(mae_rolling_mean, digits=4))   median $(round(mae_rolling_med, digits=4))");
    println(io, "  N rolling sub-windows           : $(length(rows))");
    println(io, "");
    println(io, "Per-quarter detail:");
    println(io, "");
    println(io, rpad("q",  3), " | ",
                rpad("win", 14), " | ",
                rpad("forecast", 14), " | ",
                rpad("ν*", 5), " | ",
                rpad("offdiag MAE", 12), " | ",
                rpad("Frob",     8));
    println(io, "-"^80);
    for r in rows
        println(io, rpad(r.qi, 3), " | ",
                    rpad("[$(r.win_start),$(r.win_end)]", 14), " | ",
                    rpad("[$(r.q_start),$(r.q_end)]",     14), " | ",
                    rpad(string(r.nu_star), 5), " | ",
                    rpad(round(r.offdiag_mae, digits=4), 12), " | ",
                    rpad(round(r.frob, digits=4), 8));
    end
    println(io, "");
    println(io, "Notes:");
    println(io, "  - The CHMM-N marginals are not refit; only the copula. This isolates the dependence-layer");
    println(io, "    refit benefit from the marginal refit benefit.");
    println(io, "  - The forecast horizon ($STEP days) per quarter is short, which inflates the sampling-variance");
    println(io, "    component of the per-quarter MAE; the headline is the mean across rolling sub-windows.");
    println(io, "  - The rolling refit can move ν* across the discrete grid {2,3,4,5,6,8,10,15,20,30} from");
    println(io, "    quarter to quarter; the per-quarter ν* values are reported above.");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "rolling_copula_oos.csv"), "w") do io
    println(io, "qi,win_start,win_end,q_start,q_end,nu_star,offdiag_mae,frob");
    for r in rows
        println(io, "$(r.qi),$(r.win_start),$(r.win_end),$(r.q_start),$(r.q_end),$(r.nu_star),$(round(r.offdiag_mae, digits=5)),$(round(r.frob, digits=5))");
    end
    println(io, "static_is,$(n_is),$(n_is),$(n_is + 1),$(n_full),$(copula_static.nu),$(round(cor_static.offdiag_mae, digits=5)),$(round(cor_static.frob_mean, digits=5))");
end

println("\n" * "="^72);
println("Headline:");
println("  Static-IS-fit OoS off-diag MAE  : $(round(cor_static.offdiag_mae, digits=4))");
println("  Rolling-refit OoS off-diag MAE  : mean $(round(mae_rolling_mean, digits=4))   median $(round(mae_rolling_med, digits=4))");
println("  Sub-windows: $(length(rows))");
println("  Outputs:");
println("    $out_path");
println("    $(joinpath(PAPER_ROBUSTNESS_DIR, "rolling_copula_oos.csv"))");
println("="^72);
