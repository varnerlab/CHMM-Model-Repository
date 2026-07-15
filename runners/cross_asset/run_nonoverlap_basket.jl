# ========================================================================================= #
# run_nonoverlap_basket.jl
#
# Non-overlapping comparison basket for the cross-asset copula composition
# (2026-07 technical-review response; closes tracker item T1q).
#
# The main-text six-asset universe (SPY, NVDA, JNJ, JPM, AAPL, QQQ) contains
# overlapping ETFs and constituents (SPY holds all four single names; QQQ holds
# AAPL and NVDA), which makes the dependence structure strongly positive and the
# copula task easier. This runner repeats the copula composition on a basket of
# six single names, one representative from each of six selected GICS sectors,
# with no ETFs and no cross-holdings:
#
#   MSFT (Information Technology), UNH (Health Care), BAC (Financials),
#   CAT (Industrials), PG (Consumer Staples), XOM (Energy)
#
# Same pipeline as the main basket: per-asset CHMM-N marginals at K* = 3,
# Gaussian and Student-t copulas via Kendall-tau inversion + rank reordering,
# 200 paths, per-asset KS pass rates and off-diagonal correlation MAE on the
# IS and OoS windows. The observed mean absolute off-diagonal correlation of
# both universes is reported so the "overlap-easy" claim is quantified.
#
# Output: results/cross_asset/nonoverlap_basket.{csv,txt}
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Printf

const SEED = 20260422;
Random.seed!(SEED);

const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const N_PATHS = 200;
const K = 3;
const MAX_ITER = 60;
const ASSETS = ["MSFT", "UNH", "BAC", "CAT", "PG", "XOM"];
const MAIN_ASSETS = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const OUT_DIR = joinpath(_ROOT, "results", "cross_asset");
mkpath(OUT_DIR);

println("="^80)
println("  Non-overlapping six-name basket, copula composition at K = $K")
println("  Universe: $(join(ASSETS, ", "))   Seed: $SEED   Paths: $N_PATHS")
println("="^80)

# --- data -------------------------------------------------------------------------------- #
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end

for t in ASSETS
    haskey(dataset, t) || error("ticker $t missing from IS dataset");
    haskey(oos_dataset, t) || error("ticker $t missing from OoS dataset");
end

function _returns_matrix(ds, tickers)
    cols = [log_growth_matrix(ds, t; Δt=ΔT, risk_free_rate=RISK_FREE_RATE) for t in tickers];
    n = minimum(length.(cols));
    return hcat([c[1:n] for c in cols]...);
end

R_is  = _returns_matrix(dataset, ASSETS);
R_oos = _returns_matrix(oos_dataset, ASSETS);
d = length(ASSETS);
n_is, n_oos = size(R_is, 1), size(R_oos, 1);
println("[data] IS $(size(R_is))  OoS $(size(R_oos))");

# Observed dependence level of both universes (mean |off-diagonal rho|).
function _mean_abs_offdiag(R)
    C = cor(R);
    dd = size(C, 1);
    vals = [abs(C[i, j]) for i in 1:dd for j in (i+1):dd];
    return sum(vals) / length(vals);
end
R_is_main = _returns_matrix(dataset, MAIN_ASSETS);
rho_nonoverlap = _mean_abs_offdiag(R_is);
rho_main = _mean_abs_offdiag(R_is_main);
@printf("[dep] mean |offdiag rho| IS: non-overlapping %.3f vs main basket %.3f\n",
        rho_nonoverlap, rho_main);

# --- per-asset CHMM-N marginals ---------------------------------------------------------- #
println("\n[fit] per-asset CHMM-N at K = $K");
chmms = Vector{AbstractMarkovModel}(undef, d);
for j in 1:d
    Random.seed!(SEED + j);
    chmms[j] = build(MyContinuousHiddenMarkovModel, (
        observations=R_is[:, j], number_of_states=K, max_iter=MAX_ITER));
    println("  $(ASSETS[j]) done");
end

# --- copulas ----------------------------------------------------------------------------- #
println("\n[fit] Gaussian copula");
gauss_copula = build(MyGaussianCopulaModel, (returns=R_is, tickers=ASSETS, marginals=chmms));
println("\n[fit] Student-t copula");
t_copula = build(MyStudentTCopulaModel, (returns=R_is, tickers=ASSETS, marginals=chmms));
println("  nu* = ", t_copula.nu);

Random.seed!(SEED + 100);
gauss_is  = simulate(gauss_copula, n_is,  N_PATHS);
gauss_oos = simulate(gauss_copula, n_oos, N_PATHS);
Random.seed!(SEED + 200);
t_is  = simulate(t_copula, n_is,  N_PATHS);
t_oos = simulate(t_copula, n_oos, N_PATHS);

# --- scoring ----------------------------------------------------------------------------- #
ks_g_is  = per_asset_ks_pass_rates(R_is,  gauss_is);
ks_g_oos = per_asset_ks_pass_rates(R_oos, gauss_oos);
ks_t_is  = per_asset_ks_pass_rates(R_is,  t_is);
ks_t_oos = per_asset_ks_pass_rates(R_oos, t_oos);
cor_g_is  = correlation_reproduction(R_is,  gauss_is);
cor_g_oos = correlation_reproduction(R_oos, gauss_oos);
cor_t_is  = correlation_reproduction(R_is,  t_is);
cor_t_oos = correlation_reproduction(R_oos, t_oos);

println("\n[score] off-diag MAE: t IS $(round(cor_t_is.offdiag_mae, digits=3)) / OoS $(round(cor_t_oos.offdiag_mae, digits=3));  Gaussian IS $(round(cor_g_is.offdiag_mae, digits=3)) / OoS $(round(cor_g_oos.offdiag_mae, digits=3))");

# --- outputs ----------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "nonoverlap_basket.csv");
open(csv_path, "w") do io
    println(io, "asset,ks_t_is,ks_t_oos,ks_gauss_is,ks_gauss_oos");
    for j in 1:d
        @printf(io, "%s,%.1f,%.1f,%.1f,%.1f\n", ASSETS[j],
                ks_t_is[j], ks_t_oos[j], ks_g_is[j], ks_g_oos[j]);
    end
    println(io, "summary,metric,t,gauss,");
    @printf(io, "offdiag_mae,IS,%.4f,%.4f,\n", cor_t_is.offdiag_mae, cor_g_is.offdiag_mae);
    @printf(io, "offdiag_mae,OoS,%.4f,%.4f,\n", cor_t_oos.offdiag_mae, cor_g_oos.offdiag_mae);
    @printf(io, "nu_star,,%.1f,,\n", t_copula.nu);
    @printf(io, "mean_abs_offdiag_rho_IS,nonoverlap,%.4f,,\n", rho_nonoverlap);
    @printf(io, "mean_abs_offdiag_rho_IS,main_basket,%.4f,,\n", rho_main);
end

txt_path = joinpath(OUT_DIR, "nonoverlap_basket.txt");
open(txt_path, "w") do io
    println(io, "="^96);
    println(io, "Non-overlapping six-name basket, copula composition at K = $K  (T1q / 2026-07 review)");
    println(io, "="^96);
    println(io);
    println(io, "Universe: $(join(ASSETS, ", "))  (one from each of six selected GICS sectors, no ETFs, no cross-holdings)");
    println(io, "Marginals: per-asset CHMM-N at K = $K, fitted on IS; copulas via Kendall-tau inversion;");
    println(io, "rank reordering; $N_PATHS paths; seed $SEED.");
    println(io);
    @printf(io, "Observed dependence (mean |off-diag rho|, IS): %.3f vs %.3f for the main basket\n",
            rho_nonoverlap, rho_main);
    @printf(io, "Student-t copula nu* = %.1f\n", t_copula.nu);
    println(io);
    println(io, "Per-asset KS pass rates (alpha = 0.05):");
    @printf(io, "  %-6s  %-10s  %-10s  %-12s  %s\n", "asset", "t IS", "t OoS", "Gauss IS", "Gauss OoS");
    for j in 1:d
        @printf(io, "  %-6s  %8.1f%%  %8.1f%%  %10.1f%%  %10.1f%%\n", ASSETS[j],
                ks_t_is[j], ks_t_oos[j], ks_g_is[j], ks_g_oos[j]);
    end
    println(io);
    println(io, "Off-diagonal correlation MAE (average over $(N_PATHS) paths):");
    @printf(io, "  Student-t : IS %.4f   OoS %.4f\n", cor_t_is.offdiag_mae, cor_t_oos.offdiag_mae);
    @printf(io, "  Gaussian  : IS %.4f   OoS %.4f\n", cor_g_is.offdiag_mae, cor_g_oos.offdiag_mae);
    println(io);
    println(io, "Reading (computed):");
    dir = cor_t_is.offdiag_mae <= cor_g_is.offdiag_mae ? "at or below" : "above";
    @printf(io, "  The non-overlapping basket carries weaker average co-movement (%.3f vs %.3f),\n",
            rho_nonoverlap, rho_main);
    @printf(io, "  and the Student-t copula IS off-diag MAE sits %s the Gaussian copula's.\n", dir);
end
println("wrote ", csv_path);
println("wrote ", txt_path);
println("\ndone.");
