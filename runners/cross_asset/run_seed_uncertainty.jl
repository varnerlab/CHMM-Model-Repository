# ========================================================================================= #
# run_seed_uncertainty.jl
#
# Across-seed uncertainty for the Student-t versus Gaussian copula OoS gap
# (2026-07 technical-review rerun, finding 4: "path-to-path noise" was
# asserted without quantification).
#
# For each basket (the main six-asset universe and the non-overlapping
# six-name basket) the marginals and both copulas are fitted ONCE (the fits
# are deterministic given the data and the fit seeds); the simulation is then
# repeated over 20 fixed seeds. Both copulas are simulated from the SAME seed
# in each replicate (common random numbers), so the per-seed difference in
# OoS off-diagonal correlation MAE is a paired observation. We report the
# per-copula mean and standard deviation, the paired per-seed gap
# (Student-t minus Gaussian), its mean, sd, and a 95% t-interval on 19
# degrees of freedom, and whether that interval covers zero.
#
# Output: results/cross_asset/seed_uncertainty.{csv,txt}
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Printf

const FIT_SEED = 20260422;            # same fit-seed convention as run_nonoverlap_basket.jl
const SIM_SEEDS = collect(20260501:20260520);
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const N_PATHS = 200;
const K = 3;
const MAX_ITER = 60;
const BASKETS = [
    ("main",       ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"]),
    ("nonoverlap", ["MSFT", "UNH", "BAC", "CAT", "PG", "XOM"]),
];
const OUT_DIR = joinpath(_ROOT, "results", "cross_asset");
mkpath(OUT_DIR);

println("="^80)
println("  Across-seed OoS off-diag MAE uncertainty, Student-t vs Gaussian copulas")
println("  $(length(SIM_SEEDS)) simulation seeds x $(N_PATHS) paths; marginals CHMM-N K = $K")
println("="^80)

# --- data -------------------------------------------------------------------------------- #
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end

function _returns_matrix(ds, tickers)
    cols = [log_growth_matrix(ds, t; Δt=ΔT, risk_free_rate=RISK_FREE_RATE) for t in tickers];
    n = minimum(length.(cols));
    return hcat([c[1:n] for c in cols]...);
end

# --- per-basket loop ---------------------------------------------------------------------- #
results = Dict{String,Any}();
for (b, (basket, assets)) in enumerate(BASKETS)
    println("\n[basket] $basket: $(join(assets, ", "))");
    for t in assets
        haskey(dataset, t) || error("ticker $t missing from IS dataset");
        haskey(oos_dataset, t) || error("ticker $t missing from OoS dataset");
    end
    R_is  = _returns_matrix(dataset, assets);
    R_oos = _returns_matrix(oos_dataset, assets);
    d = length(assets);
    n_oos = size(R_oos, 1);

    println("[fit] per-asset CHMM-N at K = $K");
    chmms = Vector{AbstractMarkovModel}(undef, d);
    for j in 1:d
        Random.seed!(FIT_SEED + 1000*b + j);
        chmms[j] = build(MyContinuousHiddenMarkovModel, (
            observations=R_is[:, j], number_of_states=K, max_iter=MAX_ITER));
    end
    gauss_copula = build(MyGaussianCopulaModel, (returns=R_is, tickers=assets, marginals=chmms));
    t_copula = build(MyStudentTCopulaModel, (returns=R_is, tickers=assets, marginals=chmms));
    println("[fit] nu* = ", t_copula.nu);

    mae_t = zeros(length(SIM_SEEDS));
    mae_g = zeros(length(SIM_SEEDS));
    for (i, s) in enumerate(SIM_SEEDS)
        Random.seed!(s);
        sim_t = simulate(t_copula, n_oos, N_PATHS);
        Random.seed!(s);
        sim_g = simulate(gauss_copula, n_oos, N_PATHS);
        mae_t[i] = correlation_reproduction(R_oos, sim_t).offdiag_mae;
        mae_g[i] = correlation_reproduction(R_oos, sim_g).offdiag_mae;
        @printf("  seed %d: t %.4f  gauss %.4f  gap %+.4f\n", s, mae_t[i], mae_g[i], mae_t[i] - mae_g[i]);
    end

    gap = mae_t .- mae_g;
    n = length(gap);
    gap_mean = mean(gap);
    gap_sd = std(gap);
    tq = quantile(TDist(n - 1), 0.975);
    ci_lo = gap_mean - tq * gap_sd / sqrt(n);
    ci_hi = gap_mean + tq * gap_sd / sqrt(n);
    results[basket] = (assets=assets, nu=t_copula.nu, mae_t=mae_t, mae_g=mae_g,
                       gap_mean=gap_mean, gap_sd=gap_sd, ci=(ci_lo, ci_hi),
                       covers_zero=(ci_lo <= 0.0 <= ci_hi));
    @printf("[stat] %s: t %.4f +/- %.4f, gauss %.4f +/- %.4f; paired gap %+.4f (sd %.4f), 95%% CI [%+.4f, %+.4f], covers 0: %s\n",
            basket, mean(mae_t), std(mae_t), mean(mae_g), std(mae_g),
            gap_mean, gap_sd, ci_lo, ci_hi, results[basket].covers_zero);
end

# --- outputs ----------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "seed_uncertainty.csv");
open(csv_path, "w") do io
    println(io, "basket,seed,oos_offdiag_mae_t,oos_offdiag_mae_gauss,gap_t_minus_gauss");
    for (basket, _) in BASKETS
        r = results[basket];
        for (i, s) in enumerate(SIM_SEEDS)
            @printf(io, "%s,%d,%.6f,%.6f,%+.6f\n", basket, s, r.mae_t[i], r.mae_g[i], r.mae_t[i] - r.mae_g[i]);
        end
    end
    println(io, "basket,stat,value,,");
    for (basket, _) in BASKETS
        r = results[basket];
        @printf(io, "%s,gap_mean,%+.6f,,\n", basket, r.gap_mean);
        @printf(io, "%s,gap_sd,%.6f,,\n", basket, r.gap_sd);
        @printf(io, "%s,gap_ci95_lo,%+.6f,,\n", basket, r.ci[1]);
        @printf(io, "%s,gap_ci95_hi,%+.6f,,\n", basket, r.ci[2]);
        @printf(io, "%s,covers_zero,%s,,\n", basket, r.covers_zero);
    end
end

txt_path = joinpath(OUT_DIR, "seed_uncertainty.txt");
open(txt_path, "w") do io
    println(io, "="^96);
    println(io, "Across-seed OoS off-diagonal MAE uncertainty, Student-t vs Gaussian  (2026-07 rerun, finding 4)");
    println(io, "="^96);
    println(io);
    println(io, "Marginals: per-asset CHMM-N at K = $K (fit seeds $FIT_SEED + 1000b + j); copulas fitted once");
    println(io, "per basket; $(length(SIM_SEEDS)) simulation seeds ($(SIM_SEEDS[1])..$(SIM_SEEDS[end])), $N_PATHS paths each,");
    println(io, "common random numbers across the two copulas within each seed (paired design).");
    for (basket, _) in BASKETS
        r = results[basket];
        println(io);
        println(io, "-"^96);
        println(io, "Basket: $basket  ($(join(r.assets, ", ")))   nu* = $(r.nu)");
        @printf(io, "  Student-t OoS off-diag MAE: mean %.4f  sd %.4f\n", mean(r.mae_t), std(r.mae_t));
        @printf(io, "  Gaussian  OoS off-diag MAE: mean %.4f  sd %.4f\n", mean(r.mae_g), std(r.mae_g));
        @printf(io, "  Paired gap (t - Gauss): mean %+.4f  sd %.4f  95%% t-CI [%+.4f, %+.4f]  covers 0: %s\n",
                r.gap_mean, r.gap_sd, r.ci[1], r.ci[2], r.covers_zero);
    end
    println(io);
    println(io, "Reading (computed):");
    for (basket, _) in BASKETS
        r = results[basket];
        verdict = r.covers_zero ?
            "the 95% interval covers zero, so the gap is within across-seed simulation noise" :
            "the 95% interval excludes zero, so the gap is resolved beyond across-seed simulation noise";
        @printf(io, "  %s: paired gap %+.4f, %s.\n", basket, r.gap_mean, verdict);
    end
end
println("wrote ", csv_path);
println("wrote ", txt_path);
println("\ndone.");
