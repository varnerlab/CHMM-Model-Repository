# ========================================================================================= #
# run_cross_asset_large_universe.jl
#
# Large-universe dependence-layer scaling: compare Gaussian copula, flat
# Student-t copula, and truncated C-vine on a 50-asset universe using exact
# empirical marginals. Isolates the scalability of the dependence layer.
#
# Outputs:
#   results/cross_asset_large/large_universe_table.txt
#   results/cross_asset_large/summary.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const SEED = 20260422;
Random.seed!(SEED);

const RISK_FREE = 0.0;
const DT = 1/252;
const N_ASSETS = 50;
const N_PATHS = 100;
const OUT_DIR = joinpath(_ROOT, "results", "cross_asset_large");
mkpath(OUT_DIR);

println("="^72)
println("  Large-universe dependence-layer scaling (50-asset universe)")
println("  Seed=$SEED, assets=$N_ASSETS, paths=$N_PATHS")
println("="^72)

println("\n[data] Loading broad-universe training + OoS datasets...");
train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset = MyOutOfSamplePortfolioDataSet()["dataset"];

common = sort(collect(intersect(keys(train_dataset), keys(oos_dataset))));
max_days_is = maximum(nrow(train_dataset[t]) for t in common);
max_days_oos = maximum(nrow(oos_dataset[t]) for t in common);
common = [t for t in common if nrow(train_dataset[t]) == max_days_is && nrow(oos_dataset[t]) == max_days_oos];

function avg_dollar_volume(df::DataFrame)
    return mean(df.volume .* df.volume_weighted_average_price);
end

ranked = sort(common; by=t -> avg_dollar_volume(train_dataset[t]), rev=true);
assets = ranked[1:N_ASSETS];
println("  selected top-$N_ASSETS by avg dollar volume.");
println("  first 10: ", join(assets[1:10], ", "));

R_is = hcat([log_growth_matrix(train_dataset, t; Δt=DT, risk_free_rate=RISK_FREE) for t in assets]...);
R_oos = hcat([log_growth_matrix(oos_dataset, t; Δt=DT, risk_free_rate=RISK_FREE) for t in assets]...);
n_is, d = size(R_is);
n_oos = size(R_oos, 1);
println("  IS $(size(R_is)), OoS $(size(R_oos))");

println("\n[fit] Fitting dependence models...");
gauss = build(MyGaussianCopulaModel, (
    returns=R_is, tickers=assets, marginals=AbstractMarkovModel[]));
t_cop = build(MyStudentTCopulaModel, (
    returns=R_is, tickers=assets, marginals=AbstractMarkovModel[]));
vine = build(MyTruncatedCVineCopulaModel, (
    returns=R_is, tickers=assets, marginals=AbstractMarkovModel[]));
println("  Gaussian cond(Σ) = ", round(cond(gauss.Sigma), digits=2));
println("  Student-t ν* = ", t_cop.nu);
println("  Vine root = ", assets[vine.root_index], " | t edges = ", count(==(Symbol("student_t")), vine.families), " / ", length(vine.families));

function simulate_empirical_marginal(U::Matrix{Float64}, empirical::Matrix{Float64})
    T, d = size(U);
    out = zeros(T, d);
    for j in 1:d
        col = empirical[:, j];
        ord_col = sortperm(col);
        ord_u = ordinalrank(U[:, j]);
        reordered = Vector{Float64}(undef, T);
        reordered[1:T] = col[ord_col];
        out[:, j] = reordered[ord_u];
    end
    return out;
end

function simulate_empirical_archive(model, empirical::Matrix{Float64}, n_paths::Int)
    T, d = size(empirical);
    out = Array{Float64,3}(undef, T, d, n_paths);
    for p in 1:n_paths
        U = if model isa MyGaussianCopulaModel
            _sample_gaussian_copula(model.Sigma, T)
        elseif model isa MyStudentTCopulaModel
            _sample_t_copula(model.Sigma, model.nu, T)
        else
            _sample_truncated_cvine(model, T)
        end
        out[:, :, p] = simulate_empirical_marginal(U, empirical);
    end
    return out;
end

println("\n[sim] Simulating large-universe archives...");
gauss_is = simulate_empirical_archive(gauss, R_is, N_PATHS);
gauss_oos = simulate_empirical_archive(gauss, R_oos, N_PATHS);
t_is = simulate_empirical_archive(t_cop, R_is, N_PATHS);
t_oos = simulate_empirical_archive(t_cop, R_oos, N_PATHS);
vine_is = simulate_empirical_archive(vine, R_is, N_PATHS);
vine_oos = simulate_empirical_archive(vine, R_oos, N_PATHS);

println("\n[eval] Correlation reproduction and equal-weight portfolio VaR...");
cor_gauss_is = correlation_reproduction(R_is, gauss_is);
cor_t_is = correlation_reproduction(R_is, t_is);
cor_vine_is = correlation_reproduction(R_is, vine_is);
cor_gauss_oos = correlation_reproduction(R_oos, gauss_oos);
cor_t_oos = correlation_reproduction(R_oos, t_oos);
cor_vine_oos = correlation_reproduction(R_oos, vine_oos);

portfolio_is = vec(mean(R_is; dims=2));
portfolio_oos = vec(mean(R_oos; dims=2));

function portfolio_var_stats(sim::Array{Float64,3}, portfolio_oos::Vector{Float64})
    np = size(sim, 3);
    port = zeros(size(sim, 1), np);
    for p in 1:np
        port[:, p] = vec(mean(sim[:, :, p]; dims=2));
    end
    v01 = quantile(vec(port), 0.01);
    v05 = quantile(vec(port), 0.05);
    br01 = portfolio_oos .<= v01;
    br05 = portfolio_oos .<= v05;
    return (
        v01=v01, v05=v05,
        k01=kupiec_lr(br01, 0.01), k05=kupiec_lr(br05, 0.05),
        c01=christoffersen_lr(br01), c05=christoffersen_lr(br05)
    );
end

var_gauss = portfolio_var_stats(gauss_is, portfolio_oos);
var_t = portfolio_var_stats(t_is, portfolio_oos);
var_vine = portfolio_var_stats(vine_is, portfolio_oos);

table_path = joinpath(OUT_DIR, "large_universe_table.txt");
open(table_path, "w") do io
    println(io, "="^130);
    println(io, "Large-universe dependence scaling (exact empirical marginals, $N_ASSETS assets)");
    println(io, "="^130);
    println(io, "Universe    : top-$N_ASSETS common tickers by in-sample average dollar volume.");
    println(io, "Marginals   : exact empirical marginals via rank reordering (dependence layer isolated).");
    println(io, "IS          : $(size(R_is))   OoS: $(size(R_oos))   Paths: $N_PATHS");
    println(io, "Student-t ν*: $(t_cop.nu)");
    println(io, "Vine root   : $(assets[vine.root_index])");
    println(io, "");
    println(io, rpad("Model", 18), " | ", rpad("MAE IS", 8), " | ", rpad("MAE OoS", 8), " | ", rpad("Frob IS", 8), " | ", rpad("Frob OoS", 9), " | ", rpad("Port LR01", 9), " | ", rpad("Port LR05", 9));
    println(io, "-"^130);
    println(io, rpad("Gaussian copula", 18), " | ", rpad(round(cor_gauss_is.offdiag_mae, digits=3), 8), " | ", rpad(round(cor_gauss_oos.offdiag_mae, digits=3), 8), " | ", rpad(round(cor_gauss_is.frob_mean, digits=3), 8), " | ", rpad(round(cor_gauss_oos.frob_mean, digits=3), 9), " | ", rpad(round(var_gauss.k01.LR, digits=2), 9), " | ", rpad(round(var_gauss.k05.LR, digits=2), 9));
    println(io, rpad("Student-t copula", 18), " | ", rpad(round(cor_t_is.offdiag_mae, digits=3), 8), " | ", rpad(round(cor_t_oos.offdiag_mae, digits=3), 8), " | ", rpad(round(cor_t_is.frob_mean, digits=3), 8), " | ", rpad(round(cor_t_oos.frob_mean, digits=3), 9), " | ", rpad(round(var_t.k01.LR, digits=2), 9), " | ", rpad(round(var_t.k05.LR, digits=2), 9));
    println(io, rpad("Truncated C-vine", 18), " | ", rpad(round(cor_vine_is.offdiag_mae, digits=3), 8), " | ", rpad(round(cor_vine_oos.offdiag_mae, digits=3), 8), " | ", rpad(round(cor_vine_is.frob_mean, digits=3), 8), " | ", rpad(round(cor_vine_oos.frob_mean, digits=3), 9), " | ", rpad(round(var_vine.k01.LR, digits=2), 9), " | ", rpad(round(var_vine.k05.LR, digits=2), 9));
    println(io, "");
    println(io, "Top 10 assets: ", join(assets[1:10], ", "));
end

open(joinpath(OUT_DIR, "summary.txt"), "w") do io
    n_t_edges = count(==(Symbol("student_t")), vine.families);
    println(io, "Large-universe dependence-scaling summary")
    println(io, "="^80);
    println(io, "");
    println(io, "Universe: top-$N_ASSETS common tickers by avg dollar volume.");
    println(io, "Dependence layer isolated via exact empirical marginals.");
    println(io, "");
    println(io, "IS off-diag correlation MAE:");
    println(io, "  Gaussian   $(round(cor_gauss_is.offdiag_mae, digits=3))");
    println(io, "  Student-t  $(round(cor_t_is.offdiag_mae, digits=3))");
    println(io, "  C-vine     $(round(cor_vine_is.offdiag_mae, digits=3))");
    println(io, "");
    println(io, "OoS off-diag correlation MAE:");
    println(io, "  Gaussian   $(round(cor_gauss_oos.offdiag_mae, digits=3))");
    println(io, "  Student-t  $(round(cor_t_oos.offdiag_mae, digits=3))");
    println(io, "  C-vine     $(round(cor_vine_oos.offdiag_mae, digits=3))");
    println(io, "");
    println(io, "Vine root: $(assets[vine.root_index])");
    println(io, "Student-t pair edges: $n_t_edges / $(length(vine.families))");
    println(io, "");
    println(io, "Interpretation: this is the dependence-layer scaling test for C2.");
    println(io, "If the vine beats the flat Student-t copula here, the C2 finance-venue story strengthens.");
end

println("\nSaved: $table_path");
println(read(table_path, String));
