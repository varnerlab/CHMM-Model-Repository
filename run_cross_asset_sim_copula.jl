# ========================================================================================= #
# run_cross_asset_sim_copula.jl
#
# Pipeline B (cross-asset dependence extension). Each asset's marginal comes from a fitted
# CHMM-N (the same single-index marginals used in Pipeline A / Table T2); dependence is
# injected via three constructions:
#   1. Single Index Model (SIM) with SPY as the market factor.
#   2. Gaussian copula on CHMM marginals (rank reordering).
#   3. Student-t copula on CHMM marginals (rank reordering), nu selected by profile MLE.
#   4. Truncated level-1 C-vine on CHMM marginals, pair family chosen edge-wise by AIC.
#
# Output files (results/cross_asset/):
#   - Table-T3-Cross-Asset-Dependence.txt           (Table T3 in the paper)
#   - Fig-Cross-Asset-Correlation.svg / .pdf        (Fig 7 in the paper)
#   - Fig-Cross-Asset-KS-Dist.svg / .pdf            (per-asset KS sanity check for T3)
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

const SEED = 20260422;
Random.seed!(SEED);

const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const N_PATHS = 200;
const K = 18;
const MAX_ITER = 60;
const ASSETS = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
const MARKET = "SPY";
const RESULTS_DIR = joinpath(@__DIR__, "results");


# ========================================================================================= #
# LOAD DATA
# ========================================================================================= #
println("Loading data...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];

max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) ∈ train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end

available = [t for t in ASSETS if haskey(dataset, t) && haskey(oos_dataset, t)];
println("Available assets: ", available);

R_is_list = [log_growth_matrix(dataset, t; Δt=ΔT, risk_free_rate=RISK_FREE_RATE) for t in available];
R_oos_list = [log_growth_matrix(oos_dataset, t; Δt=ΔT, risk_free_rate=RISK_FREE_RATE) for t in available];

n_is = minimum(length.(R_is_list));
n_oos = minimum(length.(R_oos_list));
R_is = hcat([r[1:n_is] for r in R_is_list]...);
R_oos = hcat([r[1:n_oos] for r in R_oos_list]...);
d = length(available);
idx_m = findfirst(x -> x == MARKET, available);
println("IS matrix: $(size(R_is)), OoS matrix: $(size(R_oos))");


# ========================================================================================= #
# FIT PER-ASSET CHMM MARGINALS
# ========================================================================================= #
println("\nFitting per-asset CHMMs (K=$K)...");
chmms = Vector{AbstractMarkovModel}(undef, d);
for j in 1:d
    print("  $(available[j])...");
    chmms[j] = build(MyContinuousHiddenMarkovModel, (
        observations=R_is[:, j], number_of_states=K, max_iter=MAX_ITER));
    println(" done.");
end


# ========================================================================================= #
# BUILD DEPENDENCE MODELS
# ========================================================================================= #
println("\nFitting SIM (market=$MARKET)...");
M_is = R_is[:, idx_m];
non_market_idx = [j for j in 1:d if j != idx_m];
sim_model = build(MySingleIndexModel, (
    returns = R_is[:, non_market_idx],
    market = M_is,
    tickers = available[non_market_idx]));
println("  Median R²: ", round(median(sim_model.r2), digits=3),
        "  β range: [", round(minimum(sim_model.betas), digits=2),
        ", ", round(maximum(sim_model.betas), digits=2), "]");

println("\nFitting Gaussian copula...");
gauss_copula = build(MyGaussianCopulaModel, (
    returns=R_is, tickers=available, marginals=chmms));
println("  Σ condition number: ", round(cond(gauss_copula.Sigma), digits=2));

println("\nFitting Student-t copula...");
t_copula = build(MyStudentTCopulaModel, (
    returns=R_is, tickers=available, marginals=chmms));
println("  ν* = ", t_copula.nu);

println("\nFitting truncated C-vine copula...");
vine_copula = build(MyTruncatedCVineCopulaModel, (
    returns=R_is, tickers=available, marginals=chmms));
println("  root asset = ", available[vine_copula.root_index]);
println("  pair families = ", Dict(string(available[j]) => string(vine_copula.families[k]) for (k, j) in enumerate(vine_copula.children)));


# ========================================================================================= #
# GENERATE MARKET PATHS VIA SPY CHMM (for SIM)
# ========================================================================================= #
println("\nSimulating SIM paths (n_paths=$N_PATHS)...");
market_chmm = chmms[idx_m];
start_m = _stationary(market_chmm);

market_paths_is = zeros(n_is, N_PATHS);
market_paths_oos = zeros(n_oos, N_PATHS);
for p in 1:N_PATHS
    market_paths_is[:, p] = _simulate_chmm_marginal(market_chmm, start_m, n_is);
    market_paths_oos[:, p] = _simulate_chmm_marginal(market_chmm, start_m, n_oos);
end

sim_paths_is = simulate(sim_model, market_paths_is);
sim_paths_oos = simulate(sim_model, market_paths_oos);

# Rebuild full d-asset tensor so SPY column contains the market factor paths
sim_full_is = zeros(n_is, d, N_PATHS);
sim_full_oos = zeros(n_oos, d, N_PATHS);
for (k, j) in enumerate(non_market_idx)
    sim_full_is[:, j, :] = sim_paths_is[:, k, :];
    sim_full_oos[:, j, :] = sim_paths_oos[:, k, :];
end
for p in 1:N_PATHS
    sim_full_is[:, idx_m, p] = market_paths_is[:, p];
    sim_full_oos[:, idx_m, p] = market_paths_oos[:, p];
end


println("Simulating Gaussian copula paths...");
gauss_paths_is = simulate(gauss_copula, n_is, N_PATHS);
gauss_paths_oos = simulate(gauss_copula, n_oos, N_PATHS);

println("Simulating Student-t copula paths...");
t_paths_is = simulate(t_copula, n_is, N_PATHS);
t_paths_oos = simulate(t_copula, n_oos, N_PATHS);

println("Simulating truncated C-vine paths...");
vine_paths_is = simulate(vine_copula, n_is, N_PATHS);
vine_paths_oos = simulate(vine_copula, n_oos, N_PATHS);


# ========================================================================================= #
# EVALUATION
# ========================================================================================= #
println("\nEvaluating cross-asset fidelity...");

ks_sim_is = per_asset_ks_pass_rates(R_is, sim_full_is);
ks_gauss_is = per_asset_ks_pass_rates(R_is, gauss_paths_is);
ks_t_is = per_asset_ks_pass_rates(R_is, t_paths_is);
ks_vine_is = per_asset_ks_pass_rates(R_is, vine_paths_is);
ks_sim_oos = per_asset_ks_pass_rates(R_oos, sim_full_oos);
ks_gauss_oos = per_asset_ks_pass_rates(R_oos, gauss_paths_oos);
ks_t_oos = per_asset_ks_pass_rates(R_oos, t_paths_oos);
ks_vine_oos = per_asset_ks_pass_rates(R_oos, vine_paths_oos);

cor_sim_is = correlation_reproduction(R_is, sim_full_is);
cor_gauss_is = correlation_reproduction(R_is, gauss_paths_is);
cor_t_is = correlation_reproduction(R_is, t_paths_is);
cor_vine_is = correlation_reproduction(R_is, vine_paths_is);
cor_sim_oos = correlation_reproduction(R_oos, sim_full_oos);
cor_gauss_oos = correlation_reproduction(R_oos, gauss_paths_oos);
cor_t_oos = correlation_reproduction(R_oos, t_paths_oos);
cor_vine_oos = correlation_reproduction(R_oos, vine_paths_oos);


# ========================================================================================= #
# WRITE TABLE
# ========================================================================================= #
mkpath(joinpath(RESULTS_DIR, "cross_asset"));
outpath = joinpath(RESULTS_DIR, "cross_asset", "Table-T3-Cross-Asset-Dependence.txt");
open(outpath, "w") do io
    println(io, "=" ^ 100);
    println(io, "TABLE T3. Cross-asset dependence: SIM, Gaussian copula, Student-t copula, truncated C-vine on CHMM marginals.");
    println(io, "          Pipeline B (cross-asset extension): marginals come from Pipeline A; dependence is added.");
    println(io, "=" ^ 100);
    println(io, "");
    println(io, "Pipeline      : B. Cross-asset extension on top of Pipeline A marginals.");
    println(io, "Tickers       : $(join(available, ", "))");
    println(io, "Marginals     : Per-asset CHMM-N (Gaussian emissions, K = $K), trained independently per ticker");
    println(io, "                (same fits as the CHMM-N rows of Table T2).");
    println(io, "Dependence    : Four competing constructions");
    println(io, "                  SIM              : Single Index Model regressing each non-market ticker on $MARKET.");
    println(io, "                  Gaussian copula  : rank-reordering with Kendall-tau-based correlation matrix.");
    println(io, "                  Student-t copula : rank-reordering with profile-MLE nu* = $(t_copula.nu).");
    println(io, "                  Truncated C-vine : root-centered pair-copula vine, edge family chosen by AIC.");
    println(io, "Market factor : $MARKET (drives the SIM factor path).");
    println(io, "Paths / alpha : $N_PATHS simulated paths per dependence model; KS thresholded at alpha = 0.05.");
    println(io, "");
    println(io, "What this table tests:");
    println(io, "  Given per-ticker CHMM marginals that already pass univariate fidelity (Table T2), which");
    println(io, "  dependence mechanism best reproduces the empirical cross-asset correlation matrix?");
    println(io, "  Two metrics are reported: per-asset KS pass rates (sanity check on marginals after");
    println(io, "  dependence is injected) and correlation-matrix distance (the core dependence metric).");
    println(io, "");
    println(io, "How this differs from Table T2:");
    println(io, "  Table T2 (results/SPY/Table-T2-Per-Ticker-Emission-Families.txt) evaluates the per-ticker");
    println(io, "  marginal only, across THREE EMISSION FAMILIES, with NO dependence. Table T3 holds the");
    println(io, "  emission family fixed (Gaussian) and varies the DEPENDENCE MECHANISM.");
    println(io, "=" ^ 110);
    println(io, "");
    println(io, "1. Per-asset KS pass rates (%)");
    println(io, "-" ^ 110);
    println(io, "Ticker    SIM IS   SIM OoS   Gauss IS   Gauss OoS    t IS      t OoS    Vine IS   Vine OoS");
    for j in 1:d
        println(io, "  $(rpad(available[j],7))  $(lpad(round(ks_sim_is[j],digits=1),6))   $(lpad(round(ks_sim_oos[j],digits=1),7))    $(lpad(round(ks_gauss_is[j],digits=1),7))    $(lpad(round(ks_gauss_oos[j],digits=1),7))   $(lpad(round(ks_t_is[j],digits=1),7))    $(lpad(round(ks_t_oos[j],digits=1),7))    $(lpad(round(ks_vine_is[j],digits=1),7))    $(lpad(round(ks_vine_oos[j],digits=1),8))");
    end
    println(io, "  Mean     $(lpad(round(mean(ks_sim_is),digits=1),6))   $(lpad(round(mean(ks_sim_oos),digits=1),7))    $(lpad(round(mean(ks_gauss_is),digits=1),7))    $(lpad(round(mean(ks_gauss_oos),digits=1),7))   $(lpad(round(mean(ks_t_is),digits=1),7))    $(lpad(round(mean(ks_t_oos),digits=1),7))    $(lpad(round(mean(ks_vine_is),digits=1),7))    $(lpad(round(mean(ks_vine_oos),digits=1),8))");
    println(io, "  Median   $(lpad(round(median(ks_sim_is),digits=1),6))   $(lpad(round(median(ks_sim_oos),digits=1),7))    $(lpad(round(median(ks_gauss_is),digits=1),7))    $(lpad(round(median(ks_gauss_oos),digits=1),7))   $(lpad(round(median(ks_t_is),digits=1),7))    $(lpad(round(median(ks_t_oos),digits=1),7))    $(lpad(round(median(ks_vine_is),digits=1),7))    $(lpad(round(median(ks_vine_oos),digits=1),8))");
    println(io, "");
    println(io, "2. Cross-asset correlation reproduction (||Sigma_sim - Sigma_obs||_F over $N_PATHS paths)");
    println(io, "-" ^ 110);
    println(io, "Model              Frobenius IS        Off-diag MAE IS   Frobenius OoS       Off-diag MAE OoS");
    println(io, "SIM                $(lpad(round(cor_sim_is.frob_mean,digits=3),8))            $(lpad(round(cor_sim_is.offdiag_mae,digits=3),8))          $(lpad(round(cor_sim_oos.frob_mean,digits=3),8))            $(lpad(round(cor_sim_oos.offdiag_mae,digits=3),8))");
    println(io, "Gaussian copula    $(lpad(round(cor_gauss_is.frob_mean,digits=3),8))            $(lpad(round(cor_gauss_is.offdiag_mae,digits=3),8))          $(lpad(round(cor_gauss_oos.frob_mean,digits=3),8))            $(lpad(round(cor_gauss_oos.offdiag_mae,digits=3),8))");
    println(io, "Student-t copula   $(lpad(round(cor_t_is.frob_mean,digits=3),8))            $(lpad(round(cor_t_is.offdiag_mae,digits=3),8))          $(lpad(round(cor_t_oos.frob_mean,digits=3),8))            $(lpad(round(cor_t_oos.offdiag_mae,digits=3),8))");
    println(io, "Truncated C-vine   $(lpad(round(cor_vine_is.frob_mean,digits=3),8))            $(lpad(round(cor_vine_is.offdiag_mae,digits=3),8))          $(lpad(round(cor_vine_oos.frob_mean,digits=3),8))            $(lpad(round(cor_vine_oos.offdiag_mae,digits=3),8))");
    println(io, "");
    println(io, "3a. Truncated C-vine edge summary (root = $(available[vine_copula.root_index]))");
    println(io, "-" ^ 110);
    println(io, "Child      family       rho_hat    nu_hat");
    for (k, j) in enumerate(vine_copula.children)
        ν_str = isfinite(vine_copula.nus[k]) ? string(round(vine_copula.nus[k], digits=1)) : "Inf";
        println(io, "  $(rpad(available[j],8))  $(rpad(String(vine_copula.families[k]),11))  $(lpad(round(vine_copula.rhos[k],digits=3),7))  $(lpad(ν_str,7))");
    end
    println(io, "");
    println(io, "3b. SIM regression summary (non-market assets, $MARKET as market factor)");
    println(io, "-" ^ 110);
    println(io, "Ticker     alpha_hat   beta_hat   R2");
    for (k, j) in enumerate(non_market_idx)
        println(io, "  $(rpad(available[j],8))  $(lpad(round(sim_model.alphas[k],digits=4),7))  $(lpad(round(sim_model.betas[k],digits=3),6))  $(lpad(round(sim_model.r2[k],digits=3),5))");
    end
    println(io, "=" ^ 110);
end
println("\nSaved: $outpath");
println(read(outpath, String));


# ========================================================================================= #
# FIGURES (save only if Plots is working; otherwise skip gracefully)
# ========================================================================================= #
figs_dir = joinpath(RESULTS_DIR, "cross_asset");
mkpath(figs_dir);

try
    # Observed correlation heatmap vs. average simulated correlation heatmap
    Σ_obs = cor(R_is);
    Σ_gauss_avg = zeros(d, d);
    Σ_t_avg = zeros(d, d);
    Σ_vine_avg = zeros(d, d);
    Σ_sim_avg = zeros(d, d);
    for p in 1:N_PATHS
        Σ_gauss_avg .+= cor(gauss_paths_is[:, :, p]);
        Σ_t_avg .+= cor(t_paths_is[:, :, p]);
        Σ_vine_avg .+= cor(vine_paths_is[:, :, p]);
        Σ_sim_avg .+= cor(sim_full_is[:, :, p]);
    end
    Σ_gauss_avg ./= N_PATHS;
    Σ_t_avg ./= N_PATHS;
    Σ_vine_avg ./= N_PATHS;
    Σ_sim_avg ./= N_PATHS;

    # 5 split panels (no top titles; (a)-(e) come from LaTeX subcaptions).
    _ps_corr = (550, 500);
    _heatmap_kw = (c=:RdBu, clims=(-1,1), aspect_ratio=1,
        xticks=(1:d, available), yticks=(1:d, available), xrotation=45, size=_ps_corr);

    p1 = heatmap(Σ_obs; _heatmap_kw...);
    savefig(p1, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-a.svg"));
    savefig(p1, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-a.pdf"));

    p2 = heatmap(Σ_sim_avg; _heatmap_kw...);
    savefig(p2, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-b.svg"));
    savefig(p2, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-b.pdf"));

    p3 = heatmap(Σ_gauss_avg; _heatmap_kw...);
    savefig(p3, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-c.svg"));
    savefig(p3, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-c.pdf"));

    p4 = heatmap(Σ_t_avg; _heatmap_kw...);
    savefig(p4, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-d.svg"));
    savefig(p4, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-d.pdf"));

    p5 = heatmap(Σ_vine_avg; _heatmap_kw...);
    savefig(p5, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-e.svg"));
    savefig(p5, joinpath(figs_dir, "Fig-Cross-Asset-Correlation-e.pdf"));
    println("Saved correlation heatmap (5 split panels).");

    # Bar chart of per-asset IS KS pass rates across models
    gx = 1:d;
    w = 0.25;
    p_is = bar(gx .- w, ks_sim_is, bar_width=w, label="SIM", legend=:bottomright);
    bar!(p_is, gx, ks_gauss_is, bar_width=w, label="Gaussian copula");
    bar!(p_is, gx .+ w, ks_t_is, bar_width=w, label="Student-t copula (nu*=$(Int(t_copula.nu)))");
    scatter!(p_is, gx, ks_vine_is, ms=5, mc=:black, label="Truncated C-vine");
    xticks!(p_is, gx, available);
    ylabel!(p_is, "In-sample KS pass rate (%), alpha=0.05");
    xlabel!(p_is, "Ticker");
    title!(p_is, "Table T3 sanity check (Pipeline B): per-asset IS KS after dependence injection\n" *
                 "$N_PATHS paths | CHMM-N marginals, K=$K | market factor = $MARKET",
           titlefontsize=9);
    ylims!(p_is, 0, 105);
    savefig(p_is, joinpath(figs_dir, "Fig-Cross-Asset-KS-Dist.svg"));
    savefig(p_is, joinpath(figs_dir, "Fig-Cross-Asset-KS-Dist.pdf"));
    println("Saved KS bar chart.");
catch err
    println("Plotting skipped: ", err);
end

println("\nCross-asset run complete. Output directory: $figs_dir");

# ========================================================================================= #
# Copy paper-relevant figures into the paper's figs directory.
# ========================================================================================= #
const PAPER_FIGS = abspath(joinpath(@__DIR__, "..", "CHMM-paper", "figs"));
if isdir(PAPER_FIGS)
    for letter in ("a", "b", "c", "d", "e"), ext in ("pdf",)
        src = joinpath(figs_dir, "Fig-Cross-Asset-Correlation-$letter.$ext");
        if isfile(src); cp(src, joinpath(PAPER_FIGS, "Fig-Cross-Asset-Correlation-$letter.$ext"); force=true); end
    end
    println("Copied Fig-Cross-Asset-Correlation-{a..e} to: $PAPER_FIGS")
else
    println("(Skipped paper copy: $PAPER_FIGS not found)")
end
