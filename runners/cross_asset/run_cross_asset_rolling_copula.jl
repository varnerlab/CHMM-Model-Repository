# ========================================================================================= #
# run_cross_asset_rolling_copula.jl
#
# 2x2 family/refit copula experiment on the six-asset SPY universe (fourth-review item 6).
# The third-review rebuild made the static-vs-rolling Student-t comparison paired, but the
# paper's broader conclusion ("refitting the dependence layer, not choosing its family,
# drives OoS dependence error") still compared a family effect measured on one full-window
# matrix with a refit effect measured on quarter-level targets - two different estimands.
# This version crosses family and refit status on IDENTICAL quarter-level targets:
#
#     arm 1: static  Gaussian copula      arm 3: static  Student-t copula
#     arm 2: rolling Gaussian copula      arm 4: rolling Student-t copula
#
# Design (paired, like-for-like):
# - Per-asset CHMM-N marginals at K* = 3 (Table tab:cross_asset headline state count),
#   fitted once on IS and held fixed in ALL FOUR arms. Only the copula differs.
# - The OoS span is partitioned into quarters of STEP = 63 trading days plus the final
#   5-day partial block. Every OoS observation is scored, but the partial block is
#   reported separately and EXCLUDED from the paired inference (stated convention): a
#   six-asset sample correlation matrix from 5 observations is rank-deficient and
#   target-noise-dominated, and equal-weighting it against 63-day blocks distorts the
#   paired t.
# - STATIC arms: copulas fit once on the IS window. ROLLING arms: copulas refit each
#   quarter on the trailing WINDOW = 252 days ending the session before the quarter
#   starts. Each quarter's simulated panel is scored against that quarter's realised
#   correlation matrix.
# - All four arms use the STRICT common-random-number simulate interface
#   (simulate(model, T, n_paths, crn_seed) in src/CrossAsset.jl) with the same per-quarter
#   crn_seed: base normals and per-asset marginal draws are identical across all four arms,
#   and the Student-t chi-square mixing draws come from their own component stream that the
#   Gaussian arms never touch. Seed-level resets alone (the fifth-review finding) are only a
#   paired-seed design: _sample_t_copula consumes an extra chi-square stream that shifts
#   every draw after it, so Gaussian and Student-t arms would not see identical marginals.
# - Inference: paired per-quarter loss differences over the COMPLETE blocks for four
#   contrasts - the refit effect within each family and the family effect within each
#   refit status - each with a paired t interval and sign counts. With only nine complete
#   quarters and overlapping 252-day estimation windows across consecutive refits, these
#   paired t results are SUGGESTIVE, not confirmatory; the per-quarter difference table
#   is the primary evidence.
# - Full-window static anchors (one 572-day panel against the single full-OoS matrix, the
#   main-text Table construction) are reported once per family, descriptively, so the
#   paper can reconcile against the main-text numbers; they take no part in the paired
#   comparison.
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
println("  2x2 family/refit cross-asset copula experiment");
println("  {static, rolling} x {Gaussian, Student-t}");
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
# Per-asset CHMM-N marginals at K* = 3 (held fixed in all four arms)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Per-asset CHMM-N marginals at K = $K on IS (held fixed in all arms)...");
chmms = Vector{AbstractMarkovModel}(undef, d);
for j in 1:d
    Random.seed!(SEED + j);
    chmms[j] = build(MyContinuousHiddenMarkovModel,
        (observations=R_is[:, j], number_of_states=K, max_iter=MAX_ITER));
    println("  $(ASSETS[j]) done.");
end

# --------------------------------------------------------------------------------------- #
# Quarter partition of the OoS span: complete 63-day blocks plus the final partial block.
# Every OoS observation is scored; the partial block is excluded from paired inference.
# --------------------------------------------------------------------------------------- #
quarter_bounds = Tuple{Int,Int}[];
let q_start = n_is + 1
    while q_start <= n_full
        push!(quarter_bounds, (q_start, min(q_start + STEP - 1, n_full)));
        q_start += STEP;
    end
end
n_q = length(quarter_bounds);
complete_idx = [qi for (qi, (a, b)) in enumerate(quarter_bounds) if (b - a + 1) == STEP];
n_c = length(complete_idx);
println("\n[quarters] $n_q OoS blocks ($n_c complete $STEP-day blocks; final block $(quarter_bounds[end][2] - quarter_bounds[end][1] + 1) days, excluded from paired inference)");

# --------------------------------------------------------------------------------------- #
# Static copulas: fit once on IS (one per family).
# --------------------------------------------------------------------------------------- #
println("\n[static] Fitting the static copulas on the IS window...");
Random.seed!(SEED);
copula_static_t = build(MyStudentTCopulaModel,
    (returns = R_is, tickers = ASSETS, marginals = chmms));
copula_static_g = build(MyGaussianCopulaModel,
    (returns = R_is, tickers = ASSETS, marginals = chmms));
println("  static Student-t nu* = $(copula_static_t.nu)");

# Descriptive full-window anchors (NOT part of the paired comparison): one n_oos-day panel
# per family against the single full-OoS correlation matrix, the main-text Table
# construction. These deliberately keep the legacy three-argument simulate + global-seed
# convention so they reconcile against the main-text table; they enter no cross-family
# contrast.
Random.seed!(SEED);
sim_full_t = simulate(copula_static_t, n_oos, N_PATHS);
cor_full_t = correlation_reproduction(R_oos, sim_full_t);
Random.seed!(SEED);
sim_full_g = simulate(copula_static_g, n_oos, N_PATHS);
cor_full_g = correlation_reproduction(R_oos, sim_full_g);
println("  [anchor] static full-window OoS off-diag MAE: t = $(round(cor_full_t.offdiag_mae, digits=4)), Gaussian = $(round(cor_full_g.offdiag_mae, digits=4))");

# --------------------------------------------------------------------------------------- #
# Paired per-quarter 2x2: identical targets, horizons, marginals, and per-quarter seeds in
# all four arms; only the copula (family x refit status) differs.
# --------------------------------------------------------------------------------------- #
rows = NamedTuple[];
for (qi, (q_start, q_end)) in enumerate(quarter_bounds)
    n_days = q_end - q_start + 1;
    R_quarter = R_full[q_start:q_end, :];

    win_end = q_start - 1;
    win_start = max(1, win_end - WINDOW + 1);
    R_window = R_full[win_start:win_end, :];

    # Rolling refits on the trailing window (Student-t seed convention unchanged from the
    # third-review runner so the t arms reproduce; the Gaussian build is deterministic).
    Random.seed!(SEED + 1000 + qi);
    copula_roll_t = build(MyStudentTCopulaModel,
        (returns = R_window, tickers = ASSETS, marginals = chmms));
    Random.seed!(SEED + 3000 + qi);
    copula_roll_g = build(MyGaussianCopulaModel,
        (returns = R_window, tickers = ASSETS, marginals = chmms));

    # Strict CRN: all four arms draw through the four-argument simulate interface with the
    # same per-quarter crn_seed, so base normals and marginal paths are identical across
    # arms (the t arms' chi-square mixing stream is separate and untouched by the Gaussian).
    crn = SEED + 2000 + qi;
    mae_st = correlation_reproduction(R_quarter, simulate(copula_static_t, n_days, N_PATHS, crn)).offdiag_mae;
    mae_rt = correlation_reproduction(R_quarter, simulate(copula_roll_t, n_days, N_PATHS, crn)).offdiag_mae;
    mae_sg = correlation_reproduction(R_quarter, simulate(copula_static_g, n_days, N_PATHS, crn)).offdiag_mae;
    mae_rg = correlation_reproduction(R_quarter, simulate(copula_roll_g, n_days, N_PATHS, crn)).offdiag_mae;

    push!(rows, (
        qi = qi,
        q_start = q_start, q_end = q_end, n_days = n_days,
        win_start = win_start, win_end = win_end,
        nu_static = copula_static_t.nu, nu_roll = copula_roll_t.nu,
        mae_static_t = mae_st, mae_roll_t = mae_rt,
        mae_static_g = mae_sg, mae_roll_g = mae_rg,
    ));
    println("  q=$qi [$(q_start),$(q_end)] ($(n_days)d)  t: static=$(round(mae_st, digits=4)) roll=$(round(mae_rt, digits=4))  G: static=$(round(mae_sg, digits=4)) roll=$(round(mae_rg, digits=4))  ν*_roll=$(copula_roll_t.nu)");
end

# --------------------------------------------------------------------------------------- #
# Paired inference across the COMPLETE quarters (time-block uncertainty). Four contrasts.
# --------------------------------------------------------------------------------------- #
function _paired_t(diffs::Vector{Float64})
    n = length(diffs);
    m = mean(diffs); s = std(diffs); se = s / sqrt(n);
    tdist = TDist(n - 1);
    t = m / se;
    p = 2 * (1 - cdf(tdist, abs(t)));
    tc = quantile(tdist, 0.975);
    return (n=n, mean=m, sd=s, se=se, t=t, p=p, ci=(m - tc*se, m + tc*se),
            n_pos=count(>(0), diffs));
end

crows = [rows[qi] for qi in complete_idx];
contrasts = [
    ("refit effect, Student-t (static - rolling)",
        [r.mae_static_t - r.mae_roll_t for r in crows]),
    ("refit effect, Gaussian (static - rolling)",
        [r.mae_static_g - r.mae_roll_g for r in crows]),
    ("family effect, static (t - Gaussian)",
        [r.mae_static_t - r.mae_static_g for r in crows]),
    ("family effect, rolling (t - Gaussian)",
        [r.mae_roll_t - r.mae_roll_g for r in crows]),
];
contrast_stats = [(name=name, stat=_paired_t(diffs)) for (name, diffs) in contrasts];

arm_means = (
    static_t = mean([r.mae_static_t for r in crows]),
    roll_t   = mean([r.mae_roll_t   for r in crows]),
    static_g = mean([r.mae_static_g for r in crows]),
    roll_g   = mean([r.mae_roll_g   for r in crows]),
);

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "Rolling_Copula_OoS.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "2x2 family/refit cross-asset copula experiment, six-asset SPY cross-section");
    println(io, "="^110);
    println(io, "");
    println(io, "Setup:");
    println(io, "  - Universe: $(join(ASSETS, ", "))");
    println(io, "  - Per-asset CHMM-N marginals at K = $K (paper headline K*), fitted once on IS, held fixed in ALL FOUR arms.");
    println(io, "  - Arms: {static, rolling-quarterly-refit} x {Gaussian, Student-t} copulas.");
    println(io, "  - Quarter partition: $n_q OoS blocks of $STEP days; the final $(rows[end].n_days)-day partial block is scored and");
    println(io, "    reported but EXCLUDED from paired inference (rank-deficient 5-observation correlation target).");
    println(io, "  - ROLLING arms: copula refit each quarter on the trailing $WINDOW days; scored on the SAME quarter target.");
    println(io, "  - Paths per simulation: $N_PATHS.  STRICT common random numbers across all four arms via the");
    println(io, "    four-argument simulate(model, T, n_paths, crn_seed) interface: identical base normals and");
    println(io, "    identical per-asset marginal draws per quarter; the Student-t chi-square mixing stream is");
    println(io, "    separate, so it does not shift the draws the arms share.");
    println(io, "  - Seed root: $SEED");
    println(io, "");
    println(io, "Arm means over the $n_c complete quarters (off-diagonal MAE, lower is better):");
    println(io, "                       Gaussian    Student-t");
    println(io, "  static             $(rpad(round(arm_means.static_g, digits=4), 11)) $(round(arm_means.static_t, digits=4))");
    println(io, "  rolling refit      $(rpad(round(arm_means.roll_g, digits=4), 11)) $(round(arm_means.roll_t, digits=4))");
    println(io, "");
    println(io, "Paired contrasts over the $n_c complete quarters (suggestive, not confirmatory: n = $n_c time");
    println(io, "blocks, overlapping $WINDOW-day estimation windows across consecutive refits, no dependence");
    println(io, "correction; the per-quarter table below is the primary evidence):");
    println(io, "");
    for c in contrast_stats
        s = c.stat;
        println(io, "  $(c.name):");
        println(io, "    mean diff $(round(s.mean, digits=4))  (sd $(round(s.sd, digits=4)), se $(round(s.se, digits=4)))   paired t($(s.n - 1)) = $(round(s.t, digits=2)), two-sided p = $(round(s.p, digits=4))");
        println(io, "    95% t-interval [$(round(s.ci[1], digits=4)), $(round(s.ci[2], digits=4))]   blocks with diff > 0: $(s.n_pos) / $(s.n)");
    end
    println(io, "");
    println(io, "Descriptive full-window anchors (main-text Table construction; not part of the paired comparison):");
    println(io, "  Student-t : off-diag MAE $(round(cor_full_t.offdiag_mae, digits=4))   (nu* = $(copula_static_t.nu))");
    println(io, "  Gaussian  : off-diag MAE $(round(cor_full_g.offdiag_mae, digits=4))");
    println(io, "");
    println(io, "Per-quarter detail (partial block marked *):");
    println(io, "");
    println(io, rpad("q", 3), " | ", rpad("forecast", 14), " | ", rpad("days", 5), " | ",
                rpad("win", 14), " | ", rpad("ν*roll", 7), " | ",
                rpad("t static", 9), " | ", rpad("t roll", 7), " | ",
                rpad("G static", 9), " | ", rpad("G roll", 7));
    println(io, "-"^100);
    for r in rows
        star = r.n_days == STEP ? "" : "*";
        println(io, rpad("$(r.qi)$star", 3), " | ",
                    rpad("[$(r.q_start),$(r.q_end)]", 14), " | ",
                    rpad(r.n_days, 5), " | ",
                    rpad("[$(r.win_start),$(r.win_end)]", 14), " | ",
                    rpad(string(r.nu_roll), 7), " | ",
                    rpad(round(r.mae_static_t, digits=4), 9), " | ",
                    rpad(round(r.mae_roll_t, digits=4), 7), " | ",
                    rpad(round(r.mae_static_g, digits=4), 9), " | ",
                    rpad(round(r.mae_roll_g, digits=4), 7));
    end
    println(io, "");
    println(io, "Notes:");
    println(io, "  - All four arms score the identical per-quarter targets with identical marginals, horizons,");
    println(io, "    base normals, and marginal draws (strict CRN), so paired differences isolate the design");
    println(io, "    factor (refit or family) and quarter-to-quarter target difficulty cancels in d_q.");
    println(io, "  - The full-window anchors keep the legacy three-argument simulate + global-seed convention");
    println(io, "    so they reconcile with the main-text table; they enter no cross-family contrast.");
    println(io, "  - The CHMM-N marginals are not refit in any arm; only the dependence layer moves.");
    println(io, "  - The refit and family effects are measured on the SAME quarter-level estimand, so their");
    println(io, "    magnitudes are directly comparable (this was not true of the earlier full-window family");
    println(io, "    comparison vs quarter-level refit comparison).");
    println(io, "  - The rolling t refit can move ν* across the discrete grid {2,3,4,5,6,8,10,15,20,30} per quarter.");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "rolling_copula_oos.csv"), "w") do io
    println(io, "qi,q_start,q_end,n_days,win_start,win_end,nu_static,nu_roll,mae_static_t,mae_roll_t,mae_static_g,mae_roll_g,in_inference");
    for r in rows
        println(io, "$(r.qi),$(r.q_start),$(r.q_end),$(r.n_days),$(r.win_start),$(r.win_end),$(r.nu_static),$(r.nu_roll),$(round(r.mae_static_t, digits=5)),$(round(r.mae_roll_t, digits=5)),$(round(r.mae_static_g, digits=5)),$(round(r.mae_roll_g, digits=5)),$(r.n_days == STEP ? 1 : 0)");
    end
    println(io, "arm_means_complete,,,,,,$(copula_static_t.nu),,$(round(arm_means.static_t, digits=5)),$(round(arm_means.roll_t, digits=5)),$(round(arm_means.static_g, digits=5)),$(round(arm_means.roll_g, digits=5)),");
    for c in contrast_stats
        s = c.stat;
        key = replace(c.name, r"[ ,()\-]+" => "_");
        println(io, "contrast,$key,,,,,,,mean=$(round(s.mean, digits=5)),t=$(round(s.t, digits=3)),p=$(round(s.p, digits=4)),ci=[$(round(s.ci[1], digits=5));$(round(s.ci[2], digits=5))],npos=$(s.n_pos)/$(s.n)");
    end
    println(io, "anchor_full_window,,,,,,$(copula_static_t.nu),,$(round(cor_full_t.offdiag_mae, digits=5)),,$(round(cor_full_g.offdiag_mae, digits=5)),,");
end

println("\n" * "="^72);
println("Arm means (complete quarters): t static $(round(arm_means.static_t, digits=4))  t roll $(round(arm_means.roll_t, digits=4))  G static $(round(arm_means.static_g, digits=4))  G roll $(round(arm_means.roll_g, digits=4))");
for c in contrast_stats
    println("  $(c.name): mean $(round(c.stat.mean, digits=4)) t=$(round(c.stat.t, digits=2)) p=$(round(c.stat.p, digits=4))");
end
println("  full-window anchors: t $(round(cor_full_t.offdiag_mae, digits=4)), G $(round(cor_full_g.offdiag_mae, digits=4))");
println("  Outputs:");
println("    $out_path");
println("    $(joinpath(PAPER_ROBUSTNESS_DIR, "rolling_copula_oos.csv"))");
println("="^72);
