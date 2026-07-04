# ========================================================================================= #
# run_arch_clustering_test.jl
#
# Formal volatility-clustering test, the hypothesis-test companion to the ACF-MAE distance.
# ACF-MAE measures how close a generator's |G_t| autocorrelation is to the observed one, but
# it is a distance, not a test (no null, no critical value). This runner adds a genuine
# portmanteau test:
#
#   - Ljung-Box test on |G_t| at lag 20 (the Project I diagnostic; Ljung & Box 1978), and the
#     McLeod-Li (1983) squared-return variant on G_t^2. Both test H0: no autocorrelation, i.e.
#     no volatility clustering / no ARCH effect.
#
# Reported:
#   1. Observed row: the |G_t| and G_t^2 Ljung-Box Q-statistic and p-value on the SPY IS and
#      OoS windows. Confirms the stylized fact is statistically significant in the data
#      (reproduces the Project I result).
#   2. Per-generator ARCH retention: over the 1000 simulated paths, the median Q-statistic
#      (a graded measure, since a binary reject-rate saturates for any clustered series) and
#      the fraction of paths that reject the no-autocorrelation null at alpha = 0.05.
#
# Generators: the i.i.d. bootstrap and GARCH(1,1) are read from the canonical baselines
# archive (both are state-count-independent, canonical seed 20260420); CHMM-N and the
# penalised CHMM-t are fit fresh at K* = 3 (the headline state count), reusing the exact
# build / seed / simulate protocol of run_kstar3_headline.jl.
#
# Output: results/diagnostics/arch_clustering_test.csv (+ arch_clustering_test.txt)
# Usage:  julia --project=. runners/diagnostics/run_arch_clustering_test.jl
# ========================================================================================= #

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."));
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
using Random, Statistics, StatsBase, HypothesisTests, FileIO, Printf;

const TICKER          = "SPY";
const RISK_FREE_RATE  = 0.0;
const ΔT              = 1/252;
const N_PATHS         = 1000;
const K_STAR          = 3;
const MAX_ITER        = 60;
const SEED            = 20260420;        # paper-canonical global seed
const LAMBDA_T        = 20.0;            # penalised CHMM-t 1/ν_k shrinkage rate
const LB_LAG          = 20;              # Ljung-Box lag, matching Project I
const RESULTS_DIR     = joinpath(_ROOT, "results");
const ARCHIVE_PATH    = joinpath(RESULTS_DIR, "baselines_archive", "sim_archive_cache.jld2");
const OUT_DIR         = joinpath(RESULTS_DIR, "diagnostics");
mkpath(OUT_DIR);

println("="^72)
println("  Formal volatility-clustering test (Ljung-Box |G_t| / McLeod-Li G_t^2), lag $LB_LAG")
println("  Seed: $SEED   Paths: $N_PATHS   CHMM state count: K* = $K_STAR")
println("="^72)

# ----------------------------------------------------------------------------------------- #
# Test helpers (reuse HypothesisTests.LjungBoxTest, already a repo dependency)
# ----------------------------------------------------------------------------------------- #
lb_absG(x; lag=LB_LAG) = (t = LjungBoxTest(abs.(x), lag); (Q = t.Q, p = pvalue(t)));
ml_sqG(x;  lag=LB_LAG) = (t = LjungBoxTest(x .^ 2,  lag); (Q = t.Q, p = pvalue(t)));

# Per-generator retention: median Q across paths + reject-rate at alpha
function retention(mat; lag=LB_LAG, α=0.05)
    np = size(mat, 2);
    Qa = Vector{Float64}(undef, np); Qs = Vector{Float64}(undef, np);
    ra = 0; rs = 0;
    for i in 1:np
        x = mat[:, i];
        a = lb_absG(x; lag); Qa[i] = a.Q; ra += (a.p < α ? 1 : 0);
        s = ml_sqG(x;  lag); Qs[i] = s.Q; rs += (s.p < α ? 1 : 0);
    end
    return (Qa_med = median(Qa), rej_a = 100*ra/np, Qs_med = median(Qs), rej_s = 100*rs/np);
end

# ----------------------------------------------------------------------------------------- #
# CHMM build / simulate helpers (verbatim from run_kstar3_headline.jl)
# ----------------------------------------------------------------------------------------- #
function _train(family::Symbol, obs::Vector{Float64}, K::Int, max_iter::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t_pen
        return build(MyStudentTHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter, ν_shrink_rate=LAMBDA_T));
    else
        error("Unknown emission family: $family")
    end
end

function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    return Categorical((T_mat^1000)[1, :]);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos::Int, n_paths::Int)
    sim_is = Array{Float64,2}(undef, n_is, n_paths);
    sim_oos = Array{Float64,2}(undef, n_oos, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, n_is);
        for j in 1:n_is;  sim_is[j,i]  = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, n_oos);
        for j in 1:n_oos; sim_oos[j,i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

# ----------------------------------------------------------------------------------------- #
# Load SPY observed data (identical to the headline runner)
# ----------------------------------------------------------------------------------------- #
println("\n[1/4] Loading SPY data...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) ∈ train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
list_of_all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, list_of_all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
idx_spy = findfirst(x -> x == TICKER, list_of_all_tickers);
R_is = all_R[:, idx_spy];
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, TICKER; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
println("  IS: $(length(R_is)) obs | OoS: $(length(R_oos)) obs")

# ----------------------------------------------------------------------------------------- #
# Observed formal test
# ----------------------------------------------------------------------------------------- #
println("\n[2/4] Observed Ljung-Box / McLeod-Li test...")
obs = Dict(
    ("Observed", "IS")  => (a=lb_absG(R_is),  s=ml_sqG(R_is)),
    ("Observed", "OoS") => (a=lb_absG(R_oos), s=ml_sqG(R_oos)),
);
for w in ("IS", "OoS")
    o = obs[("Observed", w)];
    @printf("  %-4s  |G| Q=%.1f p=%.2e   G^2 Q=%.1f p=%.2e\n", w, o.a.Q, o.a.p, o.s.Q, o.s.p)
end

# ----------------------------------------------------------------------------------------- #
# Generators: bootstrap + GARCH from the canonical archive; CHMM-N / -t fit at K* = 3
# ----------------------------------------------------------------------------------------- #
println("\n[3/4] Loading archive generators + fitting CHMM at K* = $K_STAR...")
archive = load(ARCHIVE_PATH, "archive");
gens = Vector{Tuple{String, Matrix{Float64}, Matrix{Float64}}}();   # (label, is, oos)
push!(gens, ("i.i.d. bootstrap", archive["Bootstrap"].is, archive["Bootstrap"].oos));
push!(gens, ("GARCH(1,1)",       archive["GARCH"].is,     archive["GARCH"].oos));

for (fam, label) in ((:gaussian, "CHMM-N"), (:student_t_pen, "CHMM-t (pen., lambda=$(Int(LAMBDA_T)))"))
    Random.seed!(SEED);
    model = _train(fam, R_is, K_STAR, MAX_ITER);
    sd = _stationary(model, K_STAR);
    Random.seed!(SEED + 1);
    sis, soos = _simulate_paths(model, sd, length(R_is), length(R_oos), N_PATHS);
    push!(gens, (label, sis, soos));
    println("  fit $label at K* = $K_STAR")
end

# ----------------------------------------------------------------------------------------- #
# Per-generator retention + write outputs
# ----------------------------------------------------------------------------------------- #
println("\n[4/4] Computing ARCH retention and writing outputs...")
csv_path = joinpath(OUT_DIR, "arch_clustering_test.csv");
txt_path = joinpath(OUT_DIR, "arch_clustering_test.txt");

open(csv_path, "w") do io
    println(io, "generator,window,n_paths,absG_Q,absG_pval,absG_reject_pct,G2_Q,G2_pval,G2_reject_pct")
    for w in ("IS", "OoS")
        o = obs[("Observed", w)];
        @printf(io, "Observed,%s,1,%.4f,%.3e,%.1f,%.4f,%.3e,%.1f\n",
            w, o.a.Q, o.a.p, (o.a.p < 0.05 ? 100.0 : 0.0),
               o.s.Q, o.s.p, (o.s.p < 0.05 ? 100.0 : 0.0))
    end
    for (label, sis, soos) in gens
        for (w, mat) in (("IS", sis), ("OoS", soos))
            r = retention(mat);
            @printf(io, "%s,%s,%d,%.4f,NaN,%.1f,%.4f,NaN,%.1f\n",
                label, w, size(mat, 2), r.Qa_med, r.rej_a, r.Qs_med, r.rej_s)
        end
    end
end

open(txt_path, "w") do io
    println(io, "Formal volatility-clustering test  (Ljung-Box |G_t| / McLeod-Li G_t^2, lag $LB_LAG)")
    println(io, "SPY, seed $SEED, $N_PATHS paths, CHMM at K* = $K_STAR.")
    println(io, "chi^2_$LB_LAG (0.95) critical value = 31.41; reject H0 (no ARCH) if Q above it.\n")
    @printf(io, "%-26s %-4s %12s %10s %12s %10s\n",
        "generator", "win", "|G| Q", "|G| rej%", "G^2 Q", "G^2 rej%")
    for w in ("IS", "OoS")
        o = obs[("Observed", w)];
        @printf(io, "%-26s %-4s %12.1f %10s %12.1f %10s\n",
            "Observed (target)", w, o.a.Q, @sprintf("p=%.1e", o.a.p), o.s.Q, @sprintf("p=%.1e", o.s.p))
    end
    for (label, sis, soos) in gens
        for (w, mat) in (("IS", sis), ("OoS", soos))
            r = retention(mat);
            @printf(io, "%-26s %-4s %12.1f %10.1f %12.1f %10.1f\n",
                label, w, r.Qa_med, r.rej_a, r.Qs_med, r.rej_s)
        end
    end
    println(io, "\nColumns: median Q across paths (graded), reject% = fraction of paths rejecting")
    println(io, "the no-autocorrelation null at alpha=0.05. Observed row shows the single-series Q and p.")
end

println("  wrote $csv_path")
println("  wrote $txt_path")
println("\nDone.")
