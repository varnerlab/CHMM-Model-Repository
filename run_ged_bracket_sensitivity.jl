# ========================================================================================= #
# run_ged_bracket_sensitivity.jl
#
# CHMM-GED bracket sensitivity analog of `tab:nu_bracket` for CHMM-t. Two sweeps
# at K = 18 on SPY:
#
#   (i)  p_max sweep: vary p_max ∈ {2.0, 2.5, 3.0, 3.5, 4.0} with p_min = 0.5 fixed.
#        Quantifies the LL cost of tightening the upper bracket (some bulk states
#        want p > 3 in the unrestricted fit) and confirms whether the bimodal
#        partition is robust to upper-bracket choice.
#   (ii) p_min sweep: vary p_min ∈ {0.5, 0.7, 0.85} with p_max = 3.0 fixed.
#        Probes whether the heavy-shape states pile against the lower bracket.
#        At the canonical p_min = 0.5 the smallest fitted p_k on SPY is ≈ 0.86,
#        so we expect the partition to be insensitive to p_min lifts up to that
#        level.
#
# Output: results/SPY/ged_bracket/
#   Table-p-Bracket-Sensitivity.txt    paper-ready text dump
#   p_bracket_sweep.csv                machine-readable
#
# Usage: julia --project=. run_ged_bracket_sensitivity.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");
using Random;
using SpecialFunctions: gamma;

const TICKER = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT = 1/252;
const K = 18;
const MAX_ITER = 60;
const N_PATHS = 1000;
const L = 252;
const SEED = 20260420;

const PMAX_GRID = [2.0, 2.5, 3.0, 3.5, 4.0];
const PMIN_GRID = [0.5, 0.7, 0.85];
const PMIN_DEFAULT = 0.5;
const PMAX_DEFAULT = 3.0;

const RESULTS_DIR = joinpath(_ROOT, "results", "SPY", "ged_bracket");
mkpath(RESULTS_DIR);

println("="^72)
println("  CHMM-GED bracket sensitivity  |  $TICKER  |  K = $K  |  seed = $SEED")
println("  Sweep (i):  p_max ∈ $PMAX_GRID with p_min = $PMIN_DEFAULT")
println("  Sweep (ii): p_min ∈ $PMIN_GRID with p_max = $PMAX_DEFAULT")
println("="^72)

# ========================================================================================= #
# Data
# ========================================================================================= #
println("\n[data] Loading SPY IS + OoS...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
ds = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; ds[t] = data; end; end
all_tickers = keys(ds) |> collect |> sort;
all_R = log_growth_matrix(ds, all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
idx_spy = findfirst(==(TICKER), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
R_oos = Vector{Float64}(log_growth_matrix(oos, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE));
n_is = length(R_is); n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

obs_kurt_is = (sum(((R_is .- mean(R_is)) ./ std(R_is)).^4) / n_is) - 3.0;
obs_kurt_oos = (sum(((R_oos .- mean(R_oos)) ./ std(R_oos)).^4) / n_oos) - 3.0;

# ========================================================================================= #
# Helpers
# ========================================================================================= #
function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return T_mat, Categorical(π_stat);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos::Int, n_paths::Int)
    sim_is = Array{Float64,2}(undef, n_is, n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist);
        st = model(s0, n_is);
        for j in 1:n_is; sim_is[j,i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist);
        st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

function eval_metrics(observed, sim_archive)
    np = size(sim_archive, 2); n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    L_use = min(L, n_o - 1);
    acf_o = autocor(abs.(observed), 1:L_use);

    ks_pass = 0; ad_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    w1_s = 0.0; hell_s = 0.0;

    for i in 1:np
        sim = sim_archive[:, i];
        if pvalue(ApproximateTwoSampleKSTest(observed, sim)) > 0.05; ks_pass += 1; end
        if pvalue(KSampleADTest(observed, sim)) > 0.05; ad_pass += 1; end

        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;
        acf_mae_s += mean(abs.(acf_o .- autocor(abs.(sim), 1:L_use)));

        obs_sorted = sort(observed); sim_sorted = sort(sim);
        n_min = min(length(obs_sorted), length(sim_sorted));
        obs_q = [obs_sorted[max(1, round(Int, k*length(obs_sorted)/n_min))] for k in 1:n_min];
        sim_q = [sim_sorted[max(1, round(Int, k*length(sim_sorted)/n_min))] for k in 1:n_min];
        w1_s += mean(abs.(obs_q .- sim_q));

        lo = min(minimum(observed), minimum(sim)) - 10;
        hi = max(maximum(observed), maximum(sim)) + 10;
        edges = range(lo, hi, length=101);
        h_o = fit(Histogram, observed, edges).weights ./ n_o;
        h_s = fit(Histogram, sim, edges).weights ./ length(sim);
        hell_s += sqrt(sum((sqrt.(h_o) .- sqrt.(h_s)).^2)) / sqrt(2);
    end

    return (ks=100*ks_pass/np, ad=100*ad_pass/np,
            kurt=kurt_s/np,
            acf_mae=acf_mae_s/np,
            w1=w1_s/np, hell=hell_s/np);
end

function classify_p(p_vec)
    n_g = count(p -> p >= 1.85, p_vec);
    n_i = count(p -> 1.30 <= p < 1.85, p_vec);
    n_l = count(p -> 0.85 <= p < 1.30, p_vec);
    n_s = count(p -> p < 0.85, p_vec);
    return (gauss=n_g, inter=n_i, lapl=n_l, super=n_s);
end

# ========================================================================================= #
# Sweeps
# ========================================================================================= #
function fit_and_eval(p_min::Float64, p_max::Float64)
    Random.seed!(SEED);
    m = build(MyGEDHiddenMarkovModel, (
        observations=R_is, number_of_states=K, max_iter=MAX_ITER,
        p_bounds=(p_min, p_max)));

    p_k = [m.emission[k].p for k in 1:K];
    cls = classify_p(p_k);
    ll = m.log_likelihood_history[end];

    Random.seed!(SEED + 1);
    _, start_dist = _stationary(m, K);
    sim_is, sim_oos = _simulate_paths(m, start_dist, n_is, n_oos, N_PATHS);
    m_is = eval_metrics(R_is, sim_is);
    m_oos = eval_metrics(R_oos, sim_oos);

    return (
        p_min=p_min, p_max=p_max, ll=ll,
        p_min_fit=minimum(p_k), p_med_fit=median(p_k), p_max_fit=maximum(p_k),
        n_gauss=cls.gauss, n_inter=cls.inter, n_lapl=cls.lapl, n_super=cls.super,
        ks_is=m_is.ks, ad_is=m_is.ad, ks_oos=m_oos.ks, ad_oos=m_oos.ad,
        kurt_is=m_is.kurt, kurt_oos=m_oos.kurt,
        acf_mae=m_is.acf_mae, w1=m_is.w1, hell=m_is.hell,
    );
end

println("\n[sweep i] p_max sweep with p_min = $PMIN_DEFAULT...");
pmax_rows = NamedTuple[];
for pmax in PMAX_GRID
    println("  p_max = $pmax ...");
    push!(pmax_rows, fit_and_eval(PMIN_DEFAULT, pmax));
end

println("\n[sweep ii] p_min sweep with p_max = $PMAX_DEFAULT...");
pmin_rows = NamedTuple[];
for pmin in PMIN_GRID
    println("  p_min = $pmin ...");
    push!(pmin_rows, fit_and_eval(pmin, PMAX_DEFAULT));
end

# ========================================================================================= #
# Outputs
# ========================================================================================= #
println("\n[write] Output files...");

# Paper-ready text dump.
open(joinpath(RESULTS_DIR, "Table-p-Bracket-Sensitivity.txt"), "w") do io
    println(io, "CHMM-GED bracket sensitivity at K = $K on $TICKER  (seed = $SEED, $N_PATHS sim paths)");
    println(io, "Observed IS excess kurtosis $(round(obs_kurt_is, digits=2)); OoS $(round(obs_kurt_oos, digits=2)).");
    println(io, "Bimodality counts: nG = Gaussian-like (p >= 1.85), nI = intermediate (1.30 <= p < 1.85),");
    println(io, "                   nL = Laplace-like (0.85 <= p < 1.30), nS = super-Laplace (p < 0.85).");
    println(io, "");
    println(io, "Sweep (i): p_max sweep with p_min = $PMIN_DEFAULT");
    println(io, "="^130);
    println(io, "p_max | LL       | p_min_fit | p_med_fit | p_max_fit | nG | nI | nL | nS | KS_is | AD_is | KS_oos | AD_oos | kurt_is | ACF-MAE | W1   | Hell");
    println(io, "-"^130);
    for r in pmax_rows
        println(io, "$(lpad(round(r.p_max, digits=2), 5)) | $(lpad(round(r.ll, digits=1), 8)) | $(lpad(round(r.p_min_fit, digits=2), 8)) | $(lpad(round(r.p_med_fit, digits=2), 8)) | $(lpad(round(r.p_max_fit, digits=2), 8)) | $(lpad(r.n_gauss, 2)) | $(lpad(r.n_inter, 2)) | $(lpad(r.n_lapl, 2)) | $(lpad(r.n_super, 2)) | $(lpad(round(r.ks_is, digits=1), 5)) | $(lpad(round(r.ad_is, digits=1), 5)) | $(lpad(round(r.ks_oos, digits=1), 6)) | $(lpad(round(r.ad_oos, digits=1), 6)) | $(lpad(round(r.kurt_is, digits=2), 7)) | $(lpad(round(r.acf_mae, digits=4), 7)) | $(lpad(round(r.w1, digits=3), 4)) | $(round(r.hell, digits=4))");
    end
    println(io, "="^130);
    println(io, "");

    println(io, "Sweep (ii): p_min sweep with p_max = $PMAX_DEFAULT");
    println(io, "="^130);
    println(io, "p_min | LL       | p_min_fit | p_med_fit | p_max_fit | nG | nI | nL | nS | KS_is | AD_is | KS_oos | AD_oos | kurt_is | ACF-MAE | W1   | Hell");
    println(io, "-"^130);
    for r in pmin_rows
        println(io, "$(lpad(round(r.p_min, digits=2), 5)) | $(lpad(round(r.ll, digits=1), 8)) | $(lpad(round(r.p_min_fit, digits=2), 8)) | $(lpad(round(r.p_med_fit, digits=2), 8)) | $(lpad(round(r.p_max_fit, digits=2), 8)) | $(lpad(r.n_gauss, 2)) | $(lpad(r.n_inter, 2)) | $(lpad(r.n_lapl, 2)) | $(lpad(r.n_super, 2)) | $(lpad(round(r.ks_is, digits=1), 5)) | $(lpad(round(r.ad_is, digits=1), 5)) | $(lpad(round(r.ks_oos, digits=1), 6)) | $(lpad(round(r.ad_oos, digits=1), 6)) | $(lpad(round(r.kurt_is, digits=2), 7)) | $(lpad(round(r.acf_mae, digits=4), 7)) | $(lpad(round(r.w1, digits=3), 4)) | $(round(r.hell, digits=4))");
    end
    println(io, "="^130);

    # Headline read.
    pmax_partitions = [(r.n_gauss, r.n_inter, r.n_lapl, r.n_super) for r in pmax_rows];
    pmin_partitions = [(r.n_gauss, r.n_inter, r.n_lapl, r.n_super) for r in pmin_rows];
    all_have_heavy_pmax = all(r -> (r.n_lapl + r.n_super) >= 1, pmax_rows);
    all_have_heavy_pmin = all(r -> (r.n_lapl + r.n_super) >= 1, pmin_rows);

    println(io, "");
    println(io, "Headline read:");
    println(io, "  - Every p_max in $PMAX_GRID produces >= 1 heavy-shape state: $(all_have_heavy_pmax)");
    println(io, "  - Every p_min in $PMIN_GRID produces >= 1 heavy-shape state: $(all_have_heavy_pmin)");
    println(io, "  - p_max sweep partitions (G, I, L, S): $(pmax_partitions)");
    println(io, "  - p_min sweep partitions (G, I, L, S): $(pmin_partitions)");
    canonical_ll = pmax_rows[findfirst(r -> r.p_max == PMAX_DEFAULT, pmax_rows)].ll;
    full_ll = pmax_rows[findfirst(r -> r.p_max == 4.0, pmax_rows)].ll;
    println(io, "  - LL cost of tightening p_max from 4.0 to 3.0 (canonical): $(round(canonical_ll - full_ll, digits=2)) nats");
end

# Machine-readable CSV.
open(joinpath(RESULTS_DIR, "p_bracket_sweep.csv"), "w") do io
    println(io, "sweep,p_min,p_max,ll,p_min_fit,p_med_fit,p_max_fit,n_gauss,n_inter,n_lapl,n_super,ks_is,ad_is,ks_oos,ad_oos,kurt_is,kurt_oos,acf_mae,w1,hell");
    for r in pmax_rows
        println(io, "pmax,$(r.p_min),$(r.p_max),$(r.ll),$(r.p_min_fit),$(r.p_med_fit),$(r.p_max_fit),$(r.n_gauss),$(r.n_inter),$(r.n_lapl),$(r.n_super),$(r.ks_is),$(r.ad_is),$(r.ks_oos),$(r.ad_oos),$(r.kurt_is),$(r.kurt_oos),$(r.acf_mae),$(r.w1),$(r.hell)");
    end
    for r in pmin_rows
        println(io, "pmin,$(r.p_min),$(r.p_max),$(r.ll),$(r.p_min_fit),$(r.p_med_fit),$(r.p_max_fit),$(r.n_gauss),$(r.n_inter),$(r.n_lapl),$(r.n_super),$(r.ks_is),$(r.ad_is),$(r.ks_oos),$(r.ad_oos),$(r.kurt_is),$(r.kurt_oos),$(r.acf_mae),$(r.w1),$(r.hell)");
    end
end

println("\nDone. Outputs in: $RESULTS_DIR")
println("  - Table-p-Bracket-Sensitivity.txt")
println("  - p_bracket_sweep.csv")
