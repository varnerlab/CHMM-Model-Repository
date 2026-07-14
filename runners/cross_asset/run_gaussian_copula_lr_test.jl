# =========================================================================== #
# run_gaussian_copula_lr_test.jl
#
# 2026-07 audit item (P0): the parametric bootstrap in
# run_copula_profile_ci_halfunit.jl simulates from the fitted Student-t copula
# and refits nu on a finite grid, so it quantifies sampling variability around
# the fitted finite nu; it is not a test against the Gaussian-copula limit
# (nu -> infinity), which lies outside the refit grid.
#
# This runner performs the boundary-appropriate comparison the audit asks for:
#
#   H0 : Gaussian copula with correlation Sigma (nu = infinity boundary)
#   H1 : Student-t copula, nu maximised over an extended grid
#
#   LR = 2 * ( max_nu logL_t(U; Sigma_hat, nu) - logL_gauss(U; Sigma_hat) )
#
# The null distribution of LR is simulated parametrically: B samples are drawn
# from the fitted GAUSSIAN copula, Sigma is re-estimated from each sample via
# Kendall's tau (matching the observed-data estimator), and LR is recomputed.
# Because H0 sits on the boundary of the H1 family, the usual chi-squared
# calibration does not apply; the simulated null distribution replaces it.
#
# Output: results/copula_profile_ci/gaussian_copula_lr_test.csv
#         results/copula_profile_ci/gaussian_copula_lr_test.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Printf, Distributions;

const SEED      = 20260420;
const DT        = 1/252;
const RISK_FREE = 0.0;
const B         = 500;
const OUT_DIR   = joinpath(_ROOT, "results", "copula_profile_ci");
mkpath(OUT_DIR);

# nu grid for the H1 maximisation: the paper's half-unit [3, 12] grid extended
# upward so that, under the Gaussian null, the maximiser is free to run toward
# the boundary instead of being clipped at 12.
const NU_GRID = vcat(collect(3.0:0.5:12.0), collect(13.0:1.0:20.0),
                     [25.0, 30.0, 40.0, 60.0, 100.0, 200.0]);

println("="^70)
println("  Gaussian-copula boundary LR test (2026-07 audit, P0)")
println("="^70)

# --- data: identical construction to run_copula_profile_ci_halfunit.jl ------ #
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

cross_tickers = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
cross_idx = [findfirst(==(t), all_tickers) for t in cross_tickers];
R_cross = all_R[:, cross_idx];

U = _pit_ranks(R_cross);

# --- Gaussian copula log-likelihood ----------------------------------------- #
function _gcopula_loglik(U::Matrix{Float64}, Σ::Matrix{Float64})::Float64
    T, d = size(U);
    X = similar(U);
    for j in 1:d
        X[:, j] = quantile.(Normal(), U[:, j]);
    end
    Σchol = cholesky(Symmetric(Σ));
    logdetΣ = 2.0 * sum(log.(diag(Σchol.U)));
    Σinv = inv(Symmetric(Σ));
    ll = -0.5 * T * logdetΣ;
    for t in 1:T
        x = X[t, :];
        ll += -0.5 * (dot(x, Σinv * x) - dot(x, x));
    end
    return ll;
end

function _sigma_from_tau(X::Matrix{Float64})::Matrix{Float64}
    τ = _kendall_tau_matrix(X);
    return _nearest_psd(sin.((π/2) .* τ));
end

function _lr_stat(U::Matrix{Float64})
    Σ = _sigma_from_tau(U);
    ll_g = _gcopula_loglik(U, Σ);
    ll_t = [_tcopula_profile_loglik(U, Σ, ν) for ν in NU_GRID];
    i = argmax(ll_t);
    return (lr = 2.0 * (ll_t[i] - ll_g), nu_hat = NU_GRID[i],
            ll_t = ll_t[i], ll_g = ll_g);
end

obs = _lr_stat(U);
@printf("[obs] nu_hat = %.1f   logL_t = %.3f   logL_gauss = %.3f   LR = %.3f\n",
        obs.nu_hat, obs.ll_t, obs.ll_g, obs.lr)

# --- simulate the null: Gaussian copula at the observed Sigma_hat ----------- #
Σ0 = _sigma_from_tau(U);
L0 = cholesky(Symmetric(Σ0)).L;
T_obs, d = size(U);

rng = Random.MersenneTwister(SEED + 11);
lr_null = Float64[];
nu_null = Float64[];
for b in 1:B
    Z = randn(rng, T_obs, d) * L0';
    U_b = similar(Z);
    for j in 1:d
        U_b[:, j] = cdf.(Normal(), Z[:, j] ./ sqrt(Σ0[j, j]));
    end
    r = _lr_stat(U_b);
    push!(lr_null, r.lr);
    push!(nu_null, r.nu_hat);
    if b % 50 == 0
        p_run = (1 + count(>=(obs.lr), lr_null)) / (b + 1);
        @printf("  [null] %3d/%d   running p = %.4f   null LR median = %.3f\n",
                b, B, p_run, median(lr_null));
    end
end

p_val = (1 + count(>=(obs.lr), lr_null)) / (B + 1);
q = quantile(lr_null, [0.5, 0.9, 0.95, 0.99]);

println();
@printf("[result] LR_obs = %.3f   null q50/q90/q95/q99 = %.3f / %.3f / %.3f / %.3f\n",
        obs.lr, q[1], q[2], q[3], q[4]);
@printf("[result] Monte-Carlo p-value = %.4f  (B = %d)\n", p_val, B);

# --- outputs ----------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "gaussian_copula_lr_test.csv");
open(csv_path, "w") do io
    println(io, "stat,value");
    @printf(io, "nu_hat,%.2f\n", obs.nu_hat);
    @printf(io, "logL_t,%.4f\n", obs.ll_t);
    @printf(io, "logL_gauss,%.4f\n", obs.ll_g);
    @printf(io, "LR_obs,%.4f\n", obs.lr);
    @printf(io, "null_q50,%.4f\n", q[1]);
    @printf(io, "null_q90,%.4f\n", q[2]);
    @printf(io, "null_q95,%.4f\n", q[3]);
    @printf(io, "null_q99,%.4f\n", q[4]);
    @printf(io, "p_value,%.4f\n", p_val);
    @printf(io, "B,%d\n", B);
end

txt_path = joinpath(OUT_DIR, "gaussian_copula_lr_test.txt");
open(txt_path, "w") do io
    println(io, "="^90);
    println(io, "Gaussian-copula boundary LR test on the six-asset IS universe (2026-07 audit, P0)");
    println(io, "="^90);
    println(io);
    println(io, "H0: Gaussian copula (nu -> infinity boundary), Sigma from Kendall tau.");
    println(io, "H1: Student-t copula, nu maximised over the extended grid");
    println(io, "    [3, 12] half-unit + {13..20, 25, 30, 40, 60, 100, 200}.");
    println(io, "Null distribution: $B parametric simulations from the fitted Gaussian copula,");
    println(io, "Sigma re-estimated per replicate via Kendall tau (matching the data estimator).");
    println(io);
    @printf(io, "  nu_hat (observed)      : %.1f\n", obs.nu_hat);
    @printf(io, "  logL_t at nu_hat       : %.4f\n", obs.ll_t);
    @printf(io, "  logL_gauss             : %.4f\n", obs.ll_g);
    @printf(io, "  LR_obs = 2*(Lt - Lg)   : %.4f\n", obs.lr);
    @printf(io, "  null LR q50/q90/q95/q99: %.4f / %.4f / %.4f / %.4f\n", q[1], q[2], q[3], q[4]);
    @printf(io, "  null nu_hat at grid top: %.1f%% of replicates\n",
            100 * count(>=(maximum(NU_GRID)), nu_null) / B);
    @printf(io, "  Monte-Carlo p-value    : %.4f   (B = %d)\n", p_val, B);
    println(io);
    if p_val < 0.05
        println(io, "Reading (computed): the observed LR exceeds the simulated Gaussian-null");
        println(io, "distribution at the 5% level, so the Student-t copula improvement over the");
        println(io, "Gaussian copula on this window is larger than Gaussian sampling variability");
        println(io, "under the matched Kendall-tau estimation pipeline.");
    else
        println(io, "Reading (computed): the observed LR is inside the simulated Gaussian-null");
        println(io, "distribution at the 5% level; on this window the Student-t improvement is");
        println(io, "not distinguishable from Gaussian sampling variability.");
    end
end
println("wrote ", csv_path);
println("wrote ", txt_path);
println("\ndone.");
